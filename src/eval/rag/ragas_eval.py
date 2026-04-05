"""
RAGAS Evaluation Wrapper
=========================
Evaluates RAG quality on 4 dimensions:
    faithfulness        — are claims in response supported by context?
    answer_relevancy    — does response address the query?
    context_precision   — are retrieved chunks relevant?
    context_recall      — did retrieval capture all necessary information?

Design decision: decouple retrieval quality from generation quality.
Faithfulness measures generation given context.
Context precision/recall measures retrieval independently.
This lets you diagnose WHERE failures occur:
    low faithfulness    → explainer is hallucinating despite good context
    low context_precision → retriever returning irrelevant chunks
    low context_recall  → retriever missing important information

Reference: Es et al., RAGAS (2023) https://arxiv.org/abs/2309.15217
"""

from dataclasses import dataclass


@dataclass
class RAGASScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    composite: float


class RAGASEvaluator:
    """
    Wrapper around the ragas library for standardized RAG evaluation.

    Design decision: ragas over custom RAG eval.
    ragas is the standard library for RAG evaluation — comparable
    results across papers and teams. Custom eval would be harder
    to benchmark against published work.

    Note: ragas makes LLM calls internally — budget accordingly.
    At 50 test cases: ~200-300 LLM calls for full RAGAS eval.
    Run periodically (not after every commit) — use RecSys metrics
    for fast iteration feedback.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._init_ragas()

    def _init_ragas(self):
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            self._evaluate = evaluate
            self._metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            self._available = True
        except ImportError:
            print("ragas not installed. Run: pip install ragas")
            self._available = False

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict:
        """
        Run RAGAS evaluation on a single query-response-context triple.

        Args:
            query: user's original query
            response: agent's final response
            contexts: list of retrieved text chunks
            ground_truth: reference answer (optional — needed for context_recall)
        """
        if not self._available:
            return self._dummy_scores()

        if not contexts or all(not c.strip() for c in contexts):
            # No context retrieved — retrieval failed entirely
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }

        try:
            from datasets import Dataset

            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts],
                "ground_truth": [ground_truth or query],
            }
            dataset = Dataset.from_dict(data)
            result = self._evaluate(dataset, metrics=self._metrics)

            return {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
            }
        except Exception as e:
            print(f"RAGAS eval failed: {e}")
            return self._dummy_scores()

    def _dummy_scores(self) -> dict:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    def batch_evaluate(self, test_cases: list[dict]) -> list[dict]:
        """Run RAGAS on a list of test cases. Each case needs query, response, contexts."""
        return [
            self.evaluate(
                query=tc["query"],
                response=tc["response"],
                contexts=tc["contexts"],
                ground_truth=tc.get("ground_truth", ""),
            )
            for tc in test_cases
        ]