"""
Candidate Generator — Serving Layer
=====================================
Combines two-tower ANN retrieval + Mult-VAE re-scoring into a single
get_candidates(user_id, top_k) call.

ML System Design decisions documented inline.

Architecture:
    Two-tower user tower → 64-dim vector → Qdrant ANN → top-200 candidates
    Mult-VAE → re-score top-200 → re-rank → top-K to reranker

Cold start handling:
    New user (no interaction history) → skip Mult-VAE re-scoring
    → return two-tower results directly
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range

from src.recsys.models.mult_vae import MultVAE
from src.recsys.models.two_tower import TwoTowerModel


@dataclass
class Candidate:
    item_idx: int
    movie_id: int
    title: str
    genres: list
    year: int
    plot: str
    score: float
    retrieval_score: float
    recsys_score: float
    is_cold_start: bool


class CandidateGenerator:
    """
    Serving layer: wraps two-tower + Mult-VAE + Qdrant for candidate retrieval.

    Design decision: load both models at init, keep in memory.
    At serving time, model loading latency is unacceptable.
    Both models are small enough to keep resident in GPU/CPU memory.

    Design decision: keep interaction matrix in memory (sparse).
    Needed to construct user interaction vectors for Mult-VAE input.
    At scale: replace with feature store (Feast, Tecton) for real-time
    user feature serving without loading the full matrix.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
        self._load_data()
        self.client = QdrantClient(url=config["retrieval"]["qdrant_url"])
        self.collection_name = config["retrieval"]["collection_name"]
        print("CandidateGenerator ready.")

    def _load_models(self):
        ckpt_dir = Path(self.config["training"]["checkpoint_dir"])

        # Load two-tower
        print("Loading two-tower model...")
        tt_ckpt = torch.load(ckpt_dir / "two_tower_best.pt", map_location=self.device)
        self.two_tower = TwoTowerModel(
            num_users=tt_ckpt["n_users"],
            item_feature_dim=tt_ckpt["item_feature_dim"],
            embed_dim=tt_ckpt["config"]["two_tower"]["embed_dim"],
        ).to(self.device)
        self.two_tower.load_state_dict(tt_ckpt["model_state_dict"])
        self.two_tower.eval()
        self.n_items = tt_ckpt["n_items"]

        # Load Mult-VAE
        print("Loading Mult-VAE model...")
        vae_ckpt = torch.load(ckpt_dir / "mult_vae_best.pt", map_location=self.device)
        vae_cfg = vae_ckpt["config"]
        self.mult_vae = MultVAE(
            num_items=self.n_items,
            hidden_dims=vae_cfg["recsys"]["hidden_dims"],
            latent_dim=vae_cfg["recsys"]["latent_dim"],
            dropout=vae_cfg["recsys"]["dropout"],
        ).to(self.device)
        self.mult_vae.load_state_dict(vae_ckpt["model_state_dict"])
        self.mult_vae.eval()

    def _load_data(self):
        """
        Load interaction matrix + ID maps.

        Design decision: sparse matrix in memory (~50MB for MovieLens).
        At scale: replace with feature store serving pre-computed user vectors.
        """
        print("Loading interaction matrix...")
        self.train_matrix = sp.load_npz("data/processed/train.npz")
        with open("data/processed/user2idx.json") as f:
            self.user2idx = json.load(f)
        with open("data/processed/item2idx.json") as f:
            self.item2idx = json.load(f)

    def _get_user_interaction_vector(self, user_id: str):
        """Returns interaction vector for warm users, None for cold start."""
        if user_id not in self.user2idx:
            return None
        user_idx = self.user2idx[user_id]
        row = self.train_matrix[user_idx].toarray().squeeze(0)
        if row.sum() == 0:
            return None
        return torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def get_candidates(
        self,
        user_id: str,
        top_k: int = 20,
        ann_top_k: int = 200,
        genre_filter: list = None,
        year_min: int = None,
        year_max: int = None,
    ) -> list:
        """
        Main serving method.

        Design decision: two-tower for ANN, Mult-VAE for re-scoring.
        Two-tower → single dense vector → ANN-compatible.
        Mult-VAE → score distribution → not ANN-compatible, but cheap
        to score 200 candidates vs 62K full catalog.

        Design decision: metadata filtering DURING ANN search.
        Genre/year filters applied inside Qdrant graph traversal —
        not post-hoc. Preserves retrieval budget.

        Design decision: weighted score fusion (alpha=0.5).
        Simple, interpretable, no additional training.
        At scale: learn fusion weights with a small MLP (late fusion).
        """
        is_cold_start = user_id not in self.user2idx

        # ── Step 1: Two-tower ANN retrieval ──────────────────────────────────
        if is_cold_start:
            user_emb = torch.zeros(
                1, self.config["two_tower"]["embed_dim"]
            ).to(self.device)
        else:
            user_idx = self.user2idx[user_id]
            uid_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
            user_emb = self.two_tower.user_tower(uid_tensor)

        query_vector = user_emb.squeeze(0).cpu().numpy().tolist()

        # Build metadata filter
        qdrant_filter = None
        conditions = []
        if genre_filter:
            conditions.append(FieldCondition(key="genres", match=MatchAny(any=genre_filter)))
        if year_min or year_max:
            conditions.append(FieldCondition(
                key="year",
                range=Range(gte=year_min, lte=year_max)
            ))
        if conditions:
            qdrant_filter = Filter(must=conditions)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=ann_top_k,
            with_payload=True,
        )

        if not hits:
            return []

        # ── Step 2: Mult-VAE re-scoring (warm users only) ─────────────────────
        interaction_vec = self._get_user_interaction_vector(user_id)
        vae_scores = None

        if interaction_vec is not None:
            recon, _, _ = self.mult_vae(interaction_vec)
            vae_scores = recon.squeeze(0).cpu().numpy()
            # Mask seen items
            seen = self.train_matrix[self.user2idx[user_id]].indices
            vae_scores[seen] = -np.inf

        # ── Step 3: Build + re-rank candidates ───────────────────────────────
        alpha = self.config.get("serving", {}).get("recsys_alpha", 0.5)
        candidates = []

        for hit in hits:
            retrieval_score = float(hit.score)
            recsys_score = float(vae_scores[hit.id]) if vae_scores is not None else 0.0

            if vae_scores is not None:
                combined = (
                    alpha * retrieval_score +
                    (1 - alpha) * (1 / (1 + math.exp(-recsys_score)))
                )
            else:
                combined = retrieval_score

            p = hit.payload
            candidates.append(Candidate(
                item_idx=hit.id,
                movie_id=p.get("movie_id", -1),
                title=p.get("title", ""),
                genres=p.get("genres", []),
                year=p.get("year"),
                plot=p.get("plot", ""),
                score=combined,
                retrieval_score=retrieval_score,
                recsys_score=recsys_score,
                is_cold_start=is_cold_start,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def get_candidates_as_dicts(self, user_id: str, top_k: int = 20, **kwargs) -> list:
        """Returns dicts instead of dataclasses — for agent layer."""
        return [
            {k: v for k, v in c.__dict__.items()}
            for c in self.get_candidates(user_id, top_k, **kwargs)
        ]


def load_candidate_generator(config_path: str = "configs/config.yaml") -> CandidateGenerator:
    """Convenience loader for agent layer and eval harness."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return CandidateGenerator(config)