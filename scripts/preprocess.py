"""
MovieLens 25M Preprocessing Pipeline
=====================================
ML System Design decisions documented inline.

Run:
    python scripts/preprocess.py --data_dir data/raw/ml-25m --out_dir data/processed

Output:
    data/processed/
        user2idx.json       -- user_id → contiguous int index
        item2idx.json       -- movie_id → contiguous int index
        train.npz           -- scipy sparse matrix (users × items)
        val.npz
        test.npz
        stats.json          -- dataset statistics for logging
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── Config ────────────────────────────────────────────────────────────────────

# Design decision: k-core thresholds.
# Users with <20 interactions give too little signal for the VAE encoder
# to learn a meaningful latent representation.
# Items with <5 interactions are statistically unreliable for any model.
# These are hyperparameters — document and revisit after ablation.
MIN_USER_INTERACTIONS = 20
MIN_ITEM_INTERACTIONS = 5

# Design decision: temporal split ratios.
# 80/10/10 assumes enough interactions in the val and test windows.
# For MovieLens 25M this is fine — 2.5M test interactions is plenty.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (implicit)

# Design decision: implicit feedback.
# We treat this as an implicit feedback problem — we don't use rating
# values (1.0-5.0). Any interaction (regardless of rating) is treated
# as a positive signal. This is standard for VAE-CF and two-tower training.
# Rationale: a user who rates a movie 2/5 still watched it — that's signal.
BINARIZE = True


# ── K-core filtering ──────────────────────────────────────────────────────────

def kcore_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """
    Iteratively remove users and items below interaction thresholds.

    Why iterate?
    Removing sparse users reduces item interaction counts, which may push
    some items below the item threshold. Removing those items further reduces
    user counts. We iterate until the dataset is stable (no more removals).
    Typically converges in 3-5 iterations on MovieLens.
    """
    print(f"\nK-core filtering (min_user={min_user}, min_item={min_item})")
    iteration = 0
    while True:
        iteration += 1
        n_before = len(df)

        # Remove users below threshold
        user_counts = df["userId"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["userId"].isin(valid_users)]

        # Remove items below threshold
        item_counts = df["movieId"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["movieId"].isin(valid_items)]

        n_after = len(df)
        removed = n_before - n_after
        print(f"  Iteration {iteration}: removed {removed:,} interactions "
              f"({len(valid_users):,} users, {len(valid_items):,} items remain)")

        if removed == 0:
            print(f"  Converged after {iteration} iterations.")
            break

    return df


# ── Temporal split ─────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions by timestamp — not randomly.

    ML System Design decision:
    Random splitting leaks future interactions into training. The model would
    see ratings made AFTER a user's early interactions, which it wouldn't have
    at inference time. This inflates offline metrics and causes underperformance
    in production. Temporal split simulates real deployment: train on the past,
    evaluate on the future.

    Implementation: sort globally by timestamp, split by percentile.
    Alternative: per-user last-N-items as test set (leave-one-out).
    We use global split for simplicity — leave-one-out is more rigorous
    and worth exploring in ablation.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"\nTemporal split:")
    print(f"  Train: {len(train):,} interactions "
          f"({pd.to_datetime(train['timestamp'].min(), unit='s').date()} → "
          f"{pd.to_datetime(train['timestamp'].max(), unit='s').date()})")
    print(f"  Val:   {len(val):,} interactions "
          f"({pd.to_datetime(val['timestamp'].min(), unit='s').date()} → "
          f"{pd.to_datetime(val['timestamp'].max(), unit='s').date()})")
    print(f"  Test:  {len(test):,} interactions "
          f"({pd.to_datetime(test['timestamp'].min(), unit='s').date()} → "
          f"{pd.to_datetime(test['timestamp'].max(), unit='s').date()})")

    return train, val, test


# ── ID mapping ────────────────────────────────────────────────────────────────

def build_id_maps(
    train: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Build contiguous integer mappings for users and items.

    Why contiguous indices?
    Embedding layers and sparse matrices require integer indices starting
    from 0. Raw MovieLens IDs are not contiguous — we remap them.

    Important: mappings are built from TRAIN set only.
    Users/items that appear only in val/test are out-of-vocabulary (OOV).
    This is intentional — it simulates cold start at eval time.
    """
    unique_users = sorted(train["userId"].unique())
    unique_items = sorted(train["movieId"].unique())

    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item2idx = {iid: idx for idx, iid in enumerate(unique_items)}

    print(f"\nID maps built from train set:")
    print(f"  Users: {len(user2idx):,}")
    print(f"  Items: {len(item2idx):,}")

    return user2idx, item2idx


# ── Sparse matrix construction ─────────────────────────────────────────────────

def build_sparse_matrix(
    df: pd.DataFrame,
    user2idx: dict,
    item2idx: dict,
    binarize: bool = BINARIZE,
) -> sp.csr_matrix:
    """
    Build a user × item sparse interaction matrix.

    Design decision: implicit feedback binarization.
    We set all observed interactions to 1.0 regardless of rating value.
    Mult-VAE is trained with multinomial likelihood — it treats the
    interaction vector as a distribution over items, not a regression target.

    Out-of-vocabulary users/items (not in train maps) are dropped silently.
    This correctly handles val/test users who were cold start at train time.
    """
    # Filter to known users and items
    df = df[df["userId"].isin(user2idx) & df["movieId"].isin(item2idx)].copy()

    rows = df["userId"].map(user2idx).values
    cols = df["movieId"].map(item2idx).values
    values = np.ones(len(df), dtype=np.float32) if binarize else df["rating"].values.astype(np.float32)

    n_users = len(user2idx)
    n_items = len(item2idx)

    matrix = sp.csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    density = matrix.nnz / (n_users * n_items) * 100
    print(f"  Matrix shape: {matrix.shape}, density: {density:.4f}%")

    return matrix


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(data_dir: str, out_dir: str):
    t0 = time.time()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading ratings.csv...")
    ratings_path = Path(data_dir) / "ratings.csv"
    df = pd.read_csv(ratings_path)
    print(f"  Loaded {len(df):,} interactions, "
          f"{df['userId'].nunique():,} users, "
          f"{df['movieId'].nunique():,} items")

    # 2. K-core filtering
    df = kcore_filter(df, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)

    # 3. Temporal split — before any statistics are computed on the data
    train_df, val_df, test_df = temporal_split(df)

    # 4. Build ID maps from train only
    user2idx, item2idx = build_id_maps(train_df)

    # 5. Build sparse matrices
    print("\nBuilding sparse matrices...")
    print("  Train:")
    train_mat = build_sparse_matrix(train_df, user2idx, item2idx)
    print("  Val:")
    val_mat = build_sparse_matrix(val_df, user2idx, item2idx)
    print("  Test:")
    test_mat = build_sparse_matrix(test_df, user2idx, item2idx)

    # 6. Save outputs
    print(f"\nSaving to {out_dir}/...")
    sp.save_npz(f"{out_dir}/train.npz", train_mat)
    sp.save_npz(f"{out_dir}/val.npz", val_mat)
    sp.save_npz(f"{out_dir}/test.npz", test_mat)

    # Save ID maps (convert keys to str for JSON)
    with open(f"{out_dir}/user2idx.json", "w") as f:
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    with open(f"{out_dir}/item2idx.json", "w") as f:
        json.dump({str(k): v for k, v in item2idx.items()}, f)

    # Save dataset statistics
    stats = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "n_train": int(train_mat.nnz),
        "n_val": int(val_mat.nnz),
        "n_test": int(test_mat.nnz),
        "train_density": float(train_mat.nnz / (train_mat.shape[0] * train_mat.shape[1])),
        "min_user_interactions": MIN_USER_INTERACTIONS,
        "min_item_interactions": MIN_ITEM_INTERACTIONS,
        "binarized": BINARIZE,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2)},
    }
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Done in {time.time() - t0:.1f}s")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/ml-25m")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
