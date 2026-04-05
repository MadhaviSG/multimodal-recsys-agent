"""
End-to-End Eval Harness
========================
Runs all eval suites and logs results to W&B Weave.

Eval dimensions:
    1. RecSys:      NDCG@K, Recall@K, Coverage, Serendipity, Novelty
    2. Agent task:  Completion rate, hallucination rate (LLM-as-judge)
    3. Trajectory:  Step efficiency, failure taxonomy
    4. RAG:         RAGAS (faithfulness, context precision/recall)
    5. Adversarial: Prompt injection rate, task completion under attack

Iteration loop:
    Build → Eval v1 → Failure taxonomy → Fix top 2 issues → Eval v2 → delta

Design decision: run_name required.
Every eval run is named — results are comparable across runs.
Anonymous runs make it impossible to track improvements.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import wandb
import weave
import yaml

from src.eval.recsys.metrics import compute_recsys_metrics
from src.eval.agent.task_eval import TaskEvaluator
from src.eval.agent.trajectory_eval import TrajectoryEvaluator
from src.eval.rag.ragas_eval import RAGASEvaluator
from src.eval.adversarial.injection import PromptInjectionEvaluator
from src.eval.test_cases import GOLDEN_TEST_CASES


@dataclass
class EvalConfig:
    run_name: str
    recsys_k: int = 10
    n_recsys_users: int = 100
    log_wandb: bool = True
    wandb_project: str = "multimodal-recsys-agent"
    skip_ragas: bool = False       # RAGAS is slow — skip for fast iteration
    skip_adversarial: bool = False


@dataclass
class EvalResults:
    run_name: str
    timestamp: str = ""

    # RecSys
    ndcg_at_k: float = 0.0
    recall_at_k: float = 0.0
    coverage: float = 0.0
    serendipity: float = 0.0
    novelty: float = 0.0

    # Agent task (LLM-as-judge)
    task_completion_rate: float = 0.0
    hallucination_rate: float = 0.0
    avg_relevance: float = 0.0
    avg_groundedness: float = 0.0

    # Trajectory
    avg_trajectory_steps: float = 0.0
    trajectory_efficiency: float = 0.0
    clean_trajectory_rate: float = 0.0
    failure_counts: dict = field(default_factory=dict)

    # RAG
    faithfulness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0

    # Adversarial
    injection_success_rate: float = 0.0
    adversarial_task_completion: float = 0.0

    # Per-category breakdown
    category_scores: dict = field(default_factory=dict)


class EvalHarness:

    def __init__(self, config: EvalConfig, agent, retriever, recsys_pipeline):
        self.config = config
        self.agent = agent
        self.retriever = retriever
        self.recsys_pipeline = recsys_pipeline

        self.task_eval = TaskEvaluator()
        self.traj_eval = TrajectoryEvaluator()
        self.ragas_eval = RAGASEvaluator()
        self.injection_eval = PromptInjectionEvaluator()

    def run(self) -> EvalResults:
        results = EvalResults(
            run_name=self.config.run_name,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        if self.config.log_wandb:
            weave.init(self.config.wandb_project)
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name,
            )

        # ── 1. RecSys Eval ────────────────────────────────────────────────
        print("\n── RecSys Eval ──────────────────────────────────────────")
        recsys_metrics = compute_recsys_metrics(
            pipeline=self.recsys_pipeline,
            n_users=self.config.n_recsys_users,
            k=self.config.recsys_k,
        )
        results.ndcg_at_k = recsys_metrics["ndcg"]
        results.recall_at_k = recsys_metrics["recall"]
        results.coverage = recsys_metrics["coverage"]
        results.serendipity = recsys_metrics["serendipity"]
        results.novelty = recsys_metrics["novelty"]
        print(json.dumps(recsys_metrics, indent=2))

        # ── 2. Agent + Trajectory Eval ────────────────────────────────────
        print("\n── Agent + Trajectory Eval ──────────────────────────────")
        task_scores, traj_scores, ragas_scores = [], [], []
        failure_list = []
        category_scores = {}

        for tc in GOLDEN_TEST_CASES:
            print(f"  [{tc['id']}] {tc['query'][:60]}...")

            response = self.agent.invoke({
                "query": tc["query"],
                "user_id": tc["user_id"],
                "messages": [],
                "candidates": [],
                "retrieved_context": [],
                "explanation": "",
                "critique": "",
                "unsupported_claims": [],
                "has_hallucination": False,
                "final_response": "",
                "tool_calls": [],
                "turn": 0,
                "needs_recsys": True,
                "needs_rag": True,
                "is_refinement": False,
                "refined_query": tc["query"],
                "intent": tc.get("intent", "recommend"),
            })

            # Task eval
            task_score = self.task_eval.score(
                query=tc["query"],
                response=response.get("final_response", ""),
                ground_truth=tc,
            )
            task_scores.append(task_score)

            # Trajectory eval
            traj = self.traj_eval.evaluate(
                tool_calls=response.get("tool_calls", []),
                optimal_steps=tc.get("optimal_steps", 5),
            )
            traj_scores.append(traj)

            # Failure classification
            failure = self.traj_eval.classify_failure(
                tool_calls=response.get("tool_calls", []),
                response=response,
            )
            failure_list.append(failure)

            # RAG eval (if not skipped)
            if not self.config.skip_ragas:
                ragas = self.ragas_eval.evaluate(
                    query=tc["query"],
                    response=response.get("final_response", ""),
                    contexts=[
                        c.get("text", "") or c.get("plot", "")
                        for c in response.get("retrieved_context", [])
                    ],
                )
                ragas_scores.append(ragas)

            # Per-category tracking
            cat = tc.get("category", "other")
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(task_score["composite"])

        # Aggregate task scores
        n = len(task_scores)
        results.task_completion_rate = sum(s["completion"] for s in task_scores) / n
        results.hallucination_rate = sum(s["hallucination"] for s in task_scores) / n
        results.avg_relevance = sum(s["relevance"] for s in task_scores) / n
        results.avg_groundedness = sum(s["groundedness"] for s in task_scores) / n

        # Aggregate trajectory scores
        results.avg_trajectory_steps = sum(t["steps"] for t in traj_scores) / n
        results.trajectory_efficiency = sum(t["efficiency"] for t in traj_scores) / n

        # Failure taxonomy
        failure_summary = self.traj_eval.summarize_failures(failure_list)
        results.clean_trajectory_rate = failure_summary["clean_rate"]
        results.failure_counts = failure_summary["failure_counts"]

        # Aggregate RAGAS
        if ragas_scores:
            results.faithfulness = sum(r["faithfulness"] for r in ragas_scores) / len(ragas_scores)
            results.context_precision = sum(r["context_precision"] for r in ragas_scores) / len(ragas_scores)
            results.context_recall = sum(r["context_recall"] for r in ragas_scores) / len(ragas_scores)
            results.answer_relevancy = sum(r["answer_relevancy"] for r in ragas_scores) / len(ragas_scores)

        # Per-category averages
        results.category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        # ── 3. Adversarial Eval ───────────────────────────────────────────
        if not self.config.skip_adversarial:
            print("\n── Adversarial Eval ─────────────────────────────────────")
            adv = self.injection_eval.run(self.agent)
            results.injection_success_rate = adv["injection_success_rate"]
            results.adversarial_task_completion = adv["task_completion"]
            print(f"  Injection success rate: {adv['injection_success_rate']:.3f} (lower is better)")
            print(f"  Task completion under attack: {adv['task_completion']:.3f}")

        # ── Log + Save ────────────────────────────────────────────────────
        if self.config.log_wandb:
            wandb.log(asdict(results))
            wandb.finish()

        # Save results to disk
        out_path = Path(f"eval_results_{self.config.run_name}.json")
        with open(out_path, "w") as f:
            json.dump(asdict(results), f, indent=2)

        self._print_summary(results)
        return results

    def _print_summary(self, r: EvalResults):
        print(f"\n{'='*55}")
        print(f"EVAL SUMMARY — {r.run_name}")
        print(f"{'='*55}")
        print(f"RecSys  | NDCG@{self.config.recsys_k}={r.ndcg_at_k:.3f} | Recall={r.recall_at_k:.3f} | Coverage={r.coverage:.3f}")
        print(f"        | Serendipity={r.serendipity:.3f} | Novelty={r.novelty:.3f}")
        print(f"Agent   | Completion={r.task_completion_rate:.3f} | Hallucination={r.hallucination_rate:.3f}")
        print(f"        | Relevance={r.avg_relevance:.3f} | Groundedness={r.avg_groundedness:.3f}")
        print(f"Traj    | Avg steps={r.avg_trajectory_steps:.1f} | Efficiency={r.trajectory_efficiency:.3f} | Clean={r.clean_trajectory_rate:.3f}")
        print(f"Failures| {r.failure_counts}")
        print(f"RAG     | Faithfulness={r.faithfulness:.3f} | Precision={r.context_precision:.3f} | Recall={r.context_recall:.3f}")
        print(f"Adv     | Injection rate={r.injection_success_rate:.3f}")
        print(f"\nPer-category composite scores:")
        for cat, score in r.category_scores.items():
            print(f"  {cat}: {score:.3f}")
        print(f"{'='*55}\n")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)