"""
RecSys Evaluation Metrics

Beyond accuracy metrics matter as much as NDCG/Recall for real systems.
Netflix, Spotify, and LinkedIn all evaluate on: coverage, serendipity, novelty.
"""

import numpy as np
from typing import Any


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def mrr(recommended: list, relevant: set) -> float:
    """Mean Reciprocal Rank."""
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1 / (i + 1)
    return 0.0


def coverage(all_recommendations: list[list], catalog_size: int) -> float:
    """Fraction of catalog items ever recommended — diversity signal."""
    recommended_items = {item for recs in all_recommendations for item in recs}
    return len(recommended_items) / catalog_size


def serendipity(
    recommended: list,
    expected: list,   # what a popularity baseline would recommend
    relevant: set,
    k: int,
) -> float:
    """
    Fraction of recommendations that are both relevant AND unexpected.
    Unexpected = not in the popularity baseline top-K.
    """
    expected_set = set(expected[:k])
    return sum(
        1 for item in recommended[:k]
        if item in relevant and item not in expected_set
    ) / k


def novelty(recommended: list, item_popularity: dict[Any, float], k: int) -> float:
    """
    Average self-information of recommended items.
    Less popular items have higher novelty (higher -log(p)).
    """
    scores = [
        -np.log2(item_popularity.get(item, 1e-6))
        for item in recommended[:k]
    ]
    return np.mean(scores) if scores else 0.0


def compute_recsys_metrics(pipeline, n_users: int, k: int) -> dict:
    """
    Run full RecSys eval over n_users.
    pipeline must expose: get_recommendations(user_id) -> list, get_relevant(user_id) -> set
    """
    ndcg_scores, recall_scores, serendip_scores, novelty_scores = [], [], [], []
    all_recs = []

    for user_id in pipeline.sample_users(n_users):
        recs = pipeline.get_recommendations(user_id, k=k)
        relevant = pipeline.get_relevant_items(user_id)
        popularity_baseline = pipeline.get_popularity_baseline(k=k)
        item_pop = pipeline.get_item_popularity_map()

        ndcg_scores.append(ndcg_at_k(recs, relevant, k))
        recall_scores.append(recall_at_k(recs, relevant, k))
        serendip_scores.append(serendipity(recs, popularity_baseline, relevant, k))
        novelty_scores.append(novelty(recs, item_pop, k))
        all_recs.append(recs)

    return {
        "ndcg": float(np.mean(ndcg_scores)),
        "recall": float(np.mean(recall_scores)),
        "serendipity": float(np.mean(serendip_scores)),
        "novelty": float(np.mean(novelty_scores)),
        "coverage": coverage(all_recs, pipeline.catalog_size),
    }
