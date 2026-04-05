"""
Qdrant Indexing Script
=======================
Loads item tower embeddings into Qdrant with HNSW indexing.
Attaches rich payload for metadata filtering and RAG grounding.

ML System Design decisions documented inline.

Usage:
    # Start Qdrant first
    docker run -p 6333:6333 qdrant/qdrant

    python scripts/build_index.py --config configs/config.yaml

Inputs:
    checkpoints/item_embeddings.npy   — from train_two_tower.py
    data/processed/item2idx.json      — item ID mapping
    data/raw/ml-25m/movies.csv        — title, genres
    data/raw/tmdb/metadata.json       — plot, poster_url (optional)

Output:
    Qdrant collection "content_db" with HNSW index
    Payload per point: movie_id, title, genres, year, plot, poster_url
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)


# ── Item payload builder ───────────────────────────────────────────────────────

def build_payloads(
    item2idx: dict,
    movies_path: str,
    tmdb_path: str | None = None,
) -> dict[int, dict]:
    """
    Build payload dict: item_idx → metadata dict.

    Design decision: store rich payload in Qdrant alongside embeddings.
    This enables two things:
    1. Metadata filtering during ANN search — "top-K similar WHERE year > 1990
       AND genre contains Thriller" — applied DURING search, not after.
       Post-filtering (retrieve 1000, filter down) wastes retrieval budget.
       Qdrant's filtered HNSW applies filters during graph traversal.
    2. RAG grounding — plot summary is retrieved with the embedding and
       passed to the explainer agent. No separate DB lookup needed.

    Fields:
        movie_id    — links back to interaction matrix
        title       — display name
        genres      — list of genre strings (filterable)
        year        — release year int (range filterable)
        plot        — plot summary for RAG (from TMDB if available)
        poster_url  — for VLM cold start (from TMDB if available)
    """
    movies = pd.read_csv(movies_path)

    # Load TMDB metadata if available
    tmdb = {}
    if tmdb_path and Path(tmdb_path).exists():
        with open(tmdb_path) as f:
            tmdb = json.load(f)
        print(f"  Loaded TMDB metadata for {len(tmdb)} movies")
    else:
        print("  No TMDB metadata found — plot/poster_url will be empty")

    # Invert item2idx: idx → movie_id
    idx2movie = {v: k for k, v in item2idx.items()}

    payloads = {}
    for idx, movie_id_str in idx2movie.items():
        movie_id = int(movie_id_str)

        # Get movie row from movies.csv
        row = movies[movies["movieId"] == movie_id]
        if row.empty:
            continue
        row = row.iloc[0]

        # Parse title and year
        title = str(row["title"])
        year = None
        if "(" in title and ")" in title:
            try:
                year = int(title[title.rfind("(") + 1: title.rfind(")")])
                title = title[:title.rfind("(")].strip()
            except ValueError:
                pass

        # Parse genres
        genres = [g for g in str(row["genres"]).split("|") if g != "(no genres listed)"]

        # TMDB fields
        tmdb_data = tmdb.get(str(movie_id), {})
        plot = tmdb_data.get("overview", "")
        poster_path = tmdb_data.get("poster_path", "")
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path else ""
        )

        payloads[idx] = {
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            "year": year,
            "plot": plot,
            "poster_url": poster_url,
            # Concatenated text for BM25 sparse retrieval
            "text": f"{title} {' '.join(genres)} {plot}".strip(),
        }

    print(f"  Built payloads for {len(payloads):,} items")
    return payloads


# ── Qdrant collection setup ────────────────────────────────────────────────────

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
):
    """
    Create Qdrant collection with HNSW config.

    Design decision: HNSW over IVF-PQ.
    HNSW: high recall (~98%), higher memory (full float32 vectors).
    IVF-PQ: compressed vectors, lower memory, lower recall (~90%).
    At our scale (62K items, ~64-dim embeddings), memory is not a constraint
    — HNSW gives better recall for the same latency budget.

    At scale (10M+ items): switch to IVF-PQ or Qdrant's built-in quantization.

    HNSW parameters:
        m=16: number of bidirectional links per node.
              Higher m → better recall, more memory, slower index build.
              16 is the standard starting point.
        ef_construction=200: search depth during index build.
              Higher → better index quality, slower build.
              200 is standard for high-recall requirements.

    Distance: Cosine
    Why: both towers output L2-normalized vectors (F.normalize in training).
    On the unit sphere, cosine similarity = dot product.
    Cosine distance matches the training objective exactly.
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            print(f"  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"  Collection '{collection_name}' already exists. Use --recreate to rebuild.")
            return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,  # use HNSW for collections > 10K points
        ),
    )
    print(f"  ✓ Collection '{collection_name}' created "
          f"(dim={vector_size}, distance=Cosine, m=16, ef_construction=200)")
    return True


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_items(
    client: QdrantClient,
    collection_name: str,
    embeddings: np.ndarray,
    payloads: dict[int, dict],
    batch_size: int = 256,
):
    """
    Upsert item embeddings + payloads into Qdrant in batches.

    Design decision: batch upserts (256 per batch).
    Single-point upserts have high per-request overhead.
    Full-dataset upsert in one call may exceed gRPC message size limits.
    256 per batch balances throughput and reliability.

    Design decision: use item_idx as Qdrant point ID.
    This makes lookup O(1) — given an item index from the interaction
    matrix, we can directly retrieve its Qdrant payload without a
    secondary mapping. Qdrant point IDs must be unsigned integers.
    """
    n_items = embeddings.shape[0]
    n_batches = (n_items + batch_size - 1) // batch_size
    total_indexed = 0

    print(f"\nIndexing {n_items:,} items in {n_batches} batches...")
    t0 = time.time()

    for batch_start in range(0, n_items, batch_size):
        batch_end = min(batch_start + batch_size, n_items)
        points = []

        for idx in range(batch_start, batch_end):
            if idx not in payloads:
                continue  # skip items with no metadata
            points.append(
                PointStruct(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload=payloads[idx],
                )
            )

        if points:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,  # wait for indexing to complete before next batch
            )
            total_indexed += len(points)

        if (batch_start // batch_size) % 10 == 0:
            print(f"  {total_indexed:,}/{n_items:,} indexed...")

    elapsed = time.time() - t0
    print(f"  ✓ Indexed {total_indexed:,} items in {elapsed:.1f}s "
          f"({total_indexed/elapsed:.0f} items/sec)")
    return total_indexed


# ── Verification ───────────────────────────────────────────────────────────────

def verify_index(client: QdrantClient, collection_name: str, embeddings: np.ndarray):
    """
    Sanity check: query the index with a random embedding, print results.
    Verifies the collection is queryable and payloads are attached correctly.
    """
    print("\nVerification — test query:")
    query_idx = np.random.randint(0, len(embeddings))
    query_vec = embeddings[query_idx].tolist()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=5,
        with_payload=True,
    )

    print(f"  Query item: {results[0].payload.get('title')} "
          f"(genres: {results[0].payload.get('genres')})")
    print("  Top 5 similar:")
    for r in results:
        print(f"    [{r.score:.3f}] {r.payload.get('title')} "
              f"({r.payload.get('year')}) — {r.payload.get('genres')}")

    # Test metadata filter
    print("\n  Filtered query (Thriller, year >= 2000):")
    from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
    filtered = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        query_filter=Filter(
            must=[
                FieldCondition(key="genres", match=MatchAny(any=["Thriller"])),
                FieldCondition(key="year", range=Range(gte=2000)),
            ]
        ),
        limit=5,
        with_payload=True,
    )
    for r in filtered:
        print(f"    [{r.score:.3f}] {r.payload.get('title')} "
              f"({r.payload.get('year')}) — {r.payload.get('genres')}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(config: dict, recreate: bool = False):
    print("Building Qdrant index...")

    # Load embeddings
    emb_path = Path(config["training"]["checkpoint_dir"]) / "item_embeddings.npy"
    print(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    print(f"  Embeddings: {embeddings.shape}")

    # Load item2idx
    with open("data/processed/item2idx.json") as f:
        item2idx = json.load(f)

    # Build payloads
    print("Building payloads...")
    payloads = build_payloads(
        item2idx=item2idx,
        movies_path="data/raw/ml-25m/movies.csv",
        tmdb_path="data/raw/tmdb/metadata.json",
    )

    # Connect to Qdrant
    client = QdrantClient(url=config["retrieval"]["qdrant_url"])
    collection_name = config["retrieval"]["collection_name"]

    # Create collection
    create_collection(
        client=client,
        collection_name=collection_name,
        vector_size=embeddings.shape[1],
        recreate=recreate,
    )

    # Index items
    n_indexed = index_items(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        payloads=payloads,
    )

    # Verify
    verify_index(client, collection_name, embeddings)

    # Save index stats
    collection_info = client.get_collection(collection_name)
    stats = {
        "collection_name": collection_name,
        "n_indexed": n_indexed,
        "vector_size": embeddings.shape[1],
        "distance": "Cosine",
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
        "points_count": collection_info.points_count,
    }
    stats_path = Path(config["training"]["checkpoint_dir"]) / "index_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Index built. Stats saved to {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate collection if it exists")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config, recreate=args.recreate)