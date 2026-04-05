"""
Two-Tower Training Script
==========================
Trains user tower + item tower on MovieLens 25M interaction pairs.

ML System Design decisions documented inline.

Usage:
    python scripts/train_two_tower.py --config configs/config.yaml

Outputs:
    checkpoints/two_tower_best.pt       — best val Recall@K checkpoint
    checkpoints/two_tower_final.pt      — end of training
    checkpoints/item_embeddings.npy     — item tower embeddings (ready for Qdrant)
    checkpoints/item2idx.json           — item index mapping
    checkpoints/training_config.json    — hyperparams snapshot
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

# Ensure repo root is on path — fixes circular import on Kaggle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.optim as optim
import wandb
import yaml
from torch.utils.data import DataLoader, Dataset

from src.recsys.models.two_tower import TwoTowerModel


# ── Item Feature Engineering ──────────────────────────────────────────────────

# All 20 genres in MovieLens 25M
GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)",
]
GENRE2IDX = {g: i for i, g in enumerate(GENRES)}


def build_item_features(
    movies_path: str,
    item2idx: dict,
) -> torch.Tensor:
    """
    Build item feature matrix from MovieLens movies.csv.

    Features per item:
        - Genre multi-hot (20-dim): which genres the movie belongs to
        - Release year normalized (1-dim): (year - 1900) / 100

    Design decision: no text encoder for title yet — adding bge-large
    plot embeddings is the planned TMDB extension. Current features
    give enough signal for a baseline two-tower without heavy dependencies.

    Returns:
        item_features: (n_items, feature_dim) float32 tensor
    """
    movies = pd.read_csv(movies_path)

    n_items = len(item2idx)
    feature_dim = len(GENRES) + 1  # 20 genre dims + 1 year dim
    features = np.zeros((n_items, feature_dim), dtype=np.float32)

    for _, row in movies.iterrows():
        movie_id = str(row["movieId"])
        if movie_id not in item2idx:
            continue  # filtered out in k-core

        idx = item2idx[movie_id]

        # Genre multi-hot
        genres = str(row["genres"]).split("|")
        for g in genres:
            if g in GENRE2IDX:
                features[idx, GENRE2IDX[g]] = 1.0

        # Year from title e.g. "Toy Story (1995)" → 1995
        title = str(row["title"])
        year = 0.0
        if "(" in title and ")" in title:
            try:
                year = float(title[title.rfind("(") + 1: title.rfind(")")])
                year = (year - 1900) / 100  # normalize to ~[0, 1.5]
            except ValueError:
                pass
        features[idx, -1] = year

    print(f"Item features: {features.shape}, "
          f"genre sparsity: {(features[:, :20] == 0).mean():.2%}")
    return torch.tensor(features, dtype=torch.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionPairDataset(Dataset):
    """
    Samples positive (user, item) pairs from the sparse interaction matrix.

    Design decision: sample-based dataset over full pair extraction.
    Extracting all 20M pairs upfront into tensors consumes ~300MB RAM
    and causes DataLoader to hang on Kaggle during shuffle.
    Instead: keep sparse matrix, sample random users per __getitem__,
    pick a random positive item for each user. Equivalent training signal,
    much lower memory pressure and no shuffle hang.

    Design decision: in-batch negatives.
    For a batch of B pairs, each user positive item is correct.
    All other B-1 items serve as negatives. B^2 training signals per batch.
    """
    def __init__(self, sparse_matrix: sp.csr_matrix, n_samples: int = None):
        self.matrix = sparse_matrix
        self.n_users = sparse_matrix.shape[0]
        self.n_samples = n_samples or min(sparse_matrix.nnz, 2_000_000)
        # Cache non-zero item indices per user for fast sampling
        print("Building user-item index...")
        cx = sparse_matrix.tocsr()
        self.user_items = {}
        for u in range(self.n_users):
            items = cx[u].indices
            if len(items) > 0:
                self.user_items[u] = items
        self.valid_users = np.array(list(self.user_items.keys()))
        print(f"InteractionPairDataset: {self.n_samples:,} samples/epoch, "
              f"{len(self.valid_users):,} active users")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # Sample random user, then random positive item for that user
        u = int(self.valid_users[idx % len(self.valid_users)])
        items = self.user_items[u]
        item = int(items[np.random.randint(len(items))])
        return torch.tensor(u, dtype=torch.long), torch.tensor(item, dtype=torch.long)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: TwoTowerModel,
    train_matrix: sp.csr_matrix,
    val_matrix: sp.csr_matrix,
    item_features: torch.Tensor,
    val_user_indices: np.ndarray,
    k: int,
    device: torch.device,
) -> dict:
    """
    Evaluate Recall@K on fixed val user panel.

    Design decision: pre-compute all item embeddings once per eval,
    then score val users via dot product. This is O(n_items) per eval
    rather than O(n_items * n_val_users) forward passes.

    Key detail: mask training items from predictions — same as Mult-VAE eval.
    We evaluate on val interactions only, using train interactions as input context.

    Design decision: Recall@K (not NDCG@K) as primary two-tower metric.
    Two-tower is a RETRIEVAL model — we care about recall (did the relevant
    item make it into the candidate set?), not ranking precision.
    NDCG is the reranker's job downstream.
    """
    model.eval()

    # Pre-compute all item embeddings — O(n_items) forward passes
    # Design decision: batch item embedding computation to avoid OOM
    n_items = item_features.shape[0]
    all_item_embs = []
    batch_size = 512
    for i in range(0, n_items, batch_size):
        batch = item_features[i:i + batch_size].to(device)
        emb = model.item_tower(batch)
        all_item_embs.append(emb.cpu())
    all_item_embs = torch.cat(all_item_embs, dim=0)  # (n_items, D)

    recall_scores = []
    user_ids_tensor = torch.tensor(val_user_indices, dtype=torch.long).to(device)

    # Embed all val users in one batch
    user_embs = model.user_tower(user_ids_tensor).cpu()  # (n_val, D)

    # Score: (n_val, n_items) dot product
    scores = torch.matmul(user_embs, all_item_embs.T).numpy()  # (n_val, n_items)

    for i, user_idx in enumerate(val_user_indices):
        # Mask training items
        train_items = train_matrix[user_idx].indices
        scores[i, train_items] = -np.inf

        # Ground truth val items
        val_items = set(val_matrix[user_idx].indices)
        if not val_items:
            continue

        top_k = np.argsort(scores[i])[::-1][:k]
        hits = sum(1 for item in top_k if item in val_items)
        recall_scores.append(hits / len(val_items))

    model.train()
    return {"recall": float(np.mean(recall_scores))}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(config["training"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if config["training"]["log_wandb"]:
        wandb.init(
            project=config["eval"]["wandb_project"],
            name=f"two_tower_{int(time.time())}",
            config=config,
        )

    # Load data
    print("Loading processed data...")
    train_matrix = sp.load_npz(config["training"]["train_path"])
    val_matrix = sp.load_npz(config["training"]["val_path"])
    n_users, n_items = train_matrix.shape
    print(f"  {n_users:,} users, {n_items:,} items")

    # Load ID maps
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)

    # Build item features
    item_features = build_item_features(
        movies_path="data/raw/ml-25m/movies.csv",
        item2idx=item2idx,
    ).to(device)
    item_feature_dim = item_features.shape[1]

    # Fixed val user panel — same design decision as Mult-VAE
    rng = np.random.RandomState(42)
    n_val_users = min(config["training"]["n_val_users"], n_users)
    val_user_indices = rng.choice(n_users, size=n_val_users, replace=False)

    # DataLoader
    # Design decision: large batch size for in-batch negatives.
    # 512 pairs → 511 negatives per user. 1024 → 1023 negatives.
    # Memory cost: 2 * batch_size * embed_dim * 4 bytes (negligible).
    dataset = InteractionPairDataset(train_matrix)
    loader = DataLoader(
        dataset,
        batch_size=config["two_tower"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = TwoTowerModel(
        num_users=n_users,
        item_feature_dim=item_feature_dim,
        embed_dim=config["two_tower"]["embed_dim"],
    ).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["two_tower"]["lr"],
        weight_decay=config["two_tower"]["weight_decay"],
    )

    # LR scheduler — cosine decay
    # Design decision: cosine LR decay (vs step decay).
    # Smooth decay avoids sudden loss spikes. Standard for contrastive training.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["two_tower"]["n_epochs"],
    )

    best_recall = 0.0
    global_step = 0
    k = config["eval"]["recsys_k"]

    print(f"\nStarting training for {config['two_tower']['n_epochs']} epochs...")
    for epoch in range(1, config["two_tower"]["n_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for user_ids, item_ids in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            # Get item features for this batch
            batch_item_features = item_features[item_ids]  # (B, feature_dim)

            optimizer.zero_grad()
            loss = model.in_batch_loss(user_ids, batch_item_features)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0 and config["training"]["log_wandb"]:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": global_step,
                })

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        # Validation
        if epoch % config["training"]["eval_every"] == 0:
            metrics = evaluate(
                model, train_matrix, val_matrix,
                item_features, val_user_indices, k, device,
            )
            print(f"  → Val Recall@{k}={metrics['recall']:.4f}")

            if config["training"]["log_wandb"]:
                wandb.log({"val/recall": metrics["recall"], "epoch": epoch})

            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "recall": best_recall,
                    "config": config,
                    "n_users": n_users,
                    "n_items": n_items,
                    "item_feature_dim": item_feature_dim,
                }, out_dir / "two_tower_best.pt")
                print(f"  ✓ New best checkpoint (Recall@{k}={best_recall:.4f})")

    # Save final checkpoint
    torch.save({
        "epoch": config["two_tower"]["n_epochs"],
        "model_state_dict": model.state_dict(),
        "config": config,
        "n_users": n_users,
        "n_items": n_items,
    }, out_dir / "two_tower_final.pt")

    # Export item embeddings → ready for Qdrant indexing
    # Design decision: export item embeddings at training end, not on-the-fly.
    # Item catalog is static (or updated infrequently). Pre-computing and
    # indexing all item embeddings once is far cheaper than computing them
    # at query time. User embeddings are computed at query time (dynamic).
    print("\nExporting item embeddings for Qdrant indexing...")
    model.eval()
    all_item_embs = []
    with torch.no_grad():
        for i in range(0, n_items, 512):
            batch = item_features[i:i + 512]
            emb = model.item_tower(batch)
            all_item_embs.append(emb.cpu().numpy())
    item_embeddings = np.concatenate(all_item_embs, axis=0)  # (n_items, embed_dim)
    np.save(out_dir / "item_embeddings.npy", item_embeddings)
    print(f"  ✓ Item embeddings saved: {item_embeddings.shape}")

    # Save config snapshot
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete. Best val Recall@{k}: {best_recall:.4f}")
    if config["training"]["log_wandb"]:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
