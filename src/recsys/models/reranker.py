"""
LightGBM Reranker
==================
Re-ranks ANN candidates using handcrafted features.

Design decision: LightGBM over neural reranker.
LightGBM: fast inference (<10ms), interpretable features, no GPU needed.
Neural reranker: higher accuracy, slower, GPU required, harder to debug.
For our latency budget (<500ms), LightGBM keeps reranking within 10ms.

Features:
    - retrieval_score: two-tower cosine similarity
    - recsys_score: Mult-VAE reconstruction probability
    - popularity: log interaction count (global)
    - recency: normalized release year
    - genre_match: overlap between query genres and item genres
    - diversity_penalty: similarity to already-selected items

ML System Design decisions documented inline.
"""

import numpy as np
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RankedCandidate:
    item_idx: int
    title: str
    genres: list
    year: Optional[int]
    plot: str
    score: float           # final reranker score
    features: dict         # feature values for interpretability


class LightGBMReranker:
    """
    LightGBM reranker with diversity penalty.

    Design decision: pointwise ranking (score each item independently).
    Pairwise/listwise ranking gives better results but requires
    pairwise training labels — expensive to collect. Pointwise is
    sufficient for our candidate set size (20-200 items).

    Design decision: diversity penalty in features.
    Without diversity penalty, reranker surfaces similar items.
    Penalty = max cosine similarity to already-selected items.
    Controlled by lambda_diversity hyperparameter.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lambda_diversity: float = 0.3,
    ):
        self.lambda_diversity = lambda_diversity
        self._model = None

        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, path: str):
        import lightgbm as lgb
        self._model = lgb.Booster(model_file=path)
        print(f"Reranker loaded from {path}")

    def _build_features(
        self,
        candidate: dict,
        query_genres: list[str],
        item_popularity: dict[int, float],
        selected_embeddings: list[np.ndarray],
        candidate_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Build feature vector for a single candidate.

        Feature engineering decisions:
        - log(popularity): raw count skewed, log normalizes
        - year normalized: (year - 1900) / 100, matches training preprocessing
        - genre_match: Jaccard overlap — handles multi-genre items fairly
        - diversity_penalty: max sim to selected items — discourages redundancy
        """
        features = {}

        # Retrieval scores from candidate generator
        features["retrieval_score"] = candidate.get("retrieval_score", 0.0)
        features["recsys_score"] = float(
            1 / (1 + np.exp(-candidate.get("recsys_score", 0.0)))
        )

        # Popularity (log-normalized)
        pop = item_popularity.get(candidate.get("item_idx", -1), 1)
        features["log_popularity"] = float(np.log1p(pop))

        # Recency
        year = candidate.get("year") or 2000
        features["year_normalized"] = (year - 1900) / 100.0

        # Genre match with query
        item_genres = set(candidate.get("genres", []))
        query_genre_set = set(query_genres)
        if item_genres or query_genre_set:
            intersection = len(item_genres & query_genre_set)
            union = len(item_genres | query_genre_set)
            features["genre_match"] = intersection / union if union > 0 else 0.0
        else:
            features["genre_match"] = 0.0

        # Diversity penalty: max cosine similarity to already-selected items
        if selected_embeddings and candidate_embedding is not None:
            sims = [
                float(np.dot(candidate_embedding, sel) /
                      (np.linalg.norm(candidate_embedding) * np.linalg.norm(sel) + 1e-8))
                for sel in selected_embeddings
            ]
            features["diversity_penalty"] = max(sims)
        else:
            features["diversity_penalty"] = 0.0

        # Cold start flag
        features["is_cold_start"] = float(candidate.get("is_cold_start", False))

        return features

    def rerank(
        self,
        candidates: list[dict],
        query_genres: list[str] = None,
        item_popularity: dict = None,
        item_embeddings: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> list[RankedCandidate]:
        """
        Re-rank candidates using LightGBM or feature-weighted fallback.

        Design decision: feature-weighted fallback when model not trained.
        During development, we don't always have a trained reranker.
        Fallback uses manually weighted feature combination — same
        features, simple linear combination. Enables testing the full
        pipeline before reranker training is complete.
        """
        query_genres = query_genres or []
        item_popularity = item_popularity or {}
        selected_embeddings = []

        ranked = []
        for i, candidate in enumerate(candidates):
            emb = item_embeddings[candidate["item_idx"]] if item_embeddings is not None else None

            features = self._build_features(
                candidate=candidate,
                query_genres=query_genres,
                item_popularity=item_popularity,
                selected_embeddings=selected_embeddings,
                candidate_embedding=emb,
            )

            if self._model is not None:
                # LightGBM scoring
                feature_vec = np.array(list(features.values())).reshape(1, -1)
                score = float(self._model.predict(feature_vec)[0])
            else:
                # Feature-weighted fallback
                score = (
                    0.35 * features["retrieval_score"] +
                    0.25 * features["recsys_score"] +
                    0.15 * features["genre_match"] +
                    0.10 * (1.0 - features["log_popularity"] / 10.0) +  # novelty
                    0.05 * features["year_normalized"] -
                    self.lambda_diversity * features["diversity_penalty"]
                )

            ranked.append(RankedCandidate(
                item_idx=candidate.get("item_idx", -1),
                title=candidate.get("title", ""),
                genres=candidate.get("genres", []),
                year=candidate.get("year"),
                plot=candidate.get("plot", ""),
                score=score,
                features=features,
            ))

            # Add to selected for diversity penalty in next iterations
            if emb is not None:
                selected_embeddings.append(emb)

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_n]

    def train(
        self,
        train_data: list[dict],
        val_data: list[dict],
        out_path: str = "checkpoints/reranker.lgb",
    ):
        """
        Train LightGBM reranker on labeled (candidate, relevance) pairs.

        Training data format:
            {"features": {...}, "label": 1.0}  # label = implicit relevance score
        """
        import lightgbm as lgb

        X_train = np.array([list(d["features"].values()) for d in train_data])
        y_train = np.array([d["label"] for d in train_data])
        X_val = np.array([list(d["features"].values()) for d in val_data])
        y_val = np.array([d["label"] for d in val_data])

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        params = {
            "objective": "regression",
            "metric": "ndcg",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1,
        }

        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
        )

        self._model.save_model(out_path)
        print(f"Reranker saved to {out_path}")