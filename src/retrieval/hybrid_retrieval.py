"""
Hybrid Retrieval: Dense (bge-large) + Sparse (BM25) + Reciprocal Rank Fusion
Cross-encoder reranker (BGE-Reranker-v2) on top.

Design decision: Dense retrieval alone fails on exact title/genre queries.
BM25 alone misses semantic similarity. RRF fusion recovers best of both.
Validated with RAGAS context precision metrics (see eval/rag/).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, NamedVector
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagModel, FlagReranker


@dataclass
class RetrievalResult:
    item_id: str
    title: str
    score: float
    metadata: dict


class HybridRetriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "content_db",
        embed_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_k: int = 60,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedder = FlagModel(embed_model, use_fp16=True)
        self.reranker = FlagReranker(reranker_model, use_fp16=True)
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[dict] = []

    def build_bm25_index(self, corpus: list[dict]):
        """corpus: list of {'id': ..., 'text': ..., 'metadata': ...}"""
        self._corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_emb = self.embedder.encode([query])[0].tolist()
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            limit=top_k,
        )
        return [
            RetrievalResult(
                item_id=str(h.id),
                title=h.payload.get("title", ""),
                score=h.score,
                metadata=h.payload,
            )
            for h in hits
        ]

    def _sparse_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25_index first.")
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalResult(
                item_id=self._corpus[i]["id"],
                title=self._corpus[i].get("title", ""),
                score=float(scores[i]),
                metadata=self._corpus[i].get("metadata", {}),
            )
            for i in top_indices
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """RRF score = 1/(k + rank) summed across retrieval methods."""
        scores: dict[str, float] = {}
        metadata_map: dict[str, dict] = {}

        for rank, r in enumerate(dense_results):
            scores[r.item_id] = scores.get(r.item_id, 0) + 1 / (self.rrf_k + rank + 1)
            metadata_map[r.item_id] = r.metadata

        for rank, r in enumerate(sparse_results):
            scores[r.item_id] = scores.get(r.item_id, 0) + 1 / (self.rrf_k + rank + 1)
            metadata_map.setdefault(r.item_id, r.metadata)

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return [
            RetrievalResult(
                item_id=iid,
                title=metadata_map[iid].get("title", ""),
                score=scores[iid],
                metadata=metadata_map[iid],
            )
            for iid in sorted_ids
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_n: int = 5,
    ) -> list[RetrievalResult]:
        dense = self._dense_search(query, top_k)
        sparse = self._sparse_search(query, top_k)
        fused = self._reciprocal_rank_fusion(dense, sparse)[:top_k]

        # Cross-encoder reranking on top candidates
        if rerank_top_n and len(fused) > 0:
            pairs = [[query, r.metadata.get("text", r.title)] for r in fused[:rerank_top_n * 2]]
            rerank_scores = self.reranker.compute_score(pairs)
            reranked = sorted(
                zip(fused[:len(pairs)], rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            fused = [r for r, _ in reranked[:rerank_top_n]] + fused[len(pairs):]

        return fused[:rerank_top_n]
