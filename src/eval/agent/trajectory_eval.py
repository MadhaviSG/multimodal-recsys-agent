"""
Trajectory Evaluator
=====================
Measures agent efficiency and classifies failure modes from tool_calls log.

Failure taxonomy:
    hallucinated_claim     — critic flagged unsupported claims
    tool_loop              — same node called 3+ times
    premature_termination  — refiner called with empty explanation
    context_loss           — candidates exist but explainer got empty context
    over_delegation        — more steps than optimal * 2

Design decision: failure taxonomy over aggregate error rate.
A single "error rate" metric hides the root cause.
Knowing 60% of failures are tool_loops vs. hallucinations
tells you exactly what to fix next.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass
class TrajectoryScore:
    steps: int
    optimal_steps: int
    efficiency: float          # optimal / actual (1.0 = perfect)
    failure_mode: str | None   # from taxonomy, or None if clean


class TrajectoryEvaluator:

    def evaluate(
        self,
        tool_calls: list[dict],
        optimal_steps: int = 5,
    ) -> dict:
        """
        Compute trajectory efficiency.

        efficiency = optimal_steps / actual_steps
        Perfect agent: efficiency = 1.0
        Agent that took 2x optimal steps: efficiency = 0.5

        Design decision: cap efficiency at 1.0.
        An agent that takes fewer steps than optimal may have skipped
        necessary nodes — don't reward that as "super efficient."
        """
        actual_steps = len(tool_calls)
        efficiency = min(1.0, optimal_steps / actual_steps) if actual_steps > 0 else 0.0

        return {
            "steps": actual_steps,
            "optimal_steps": optimal_steps,
            "efficiency": efficiency,
            "node_sequence": [tc.get("node") for tc in tool_calls],
        }

    def classify_failure(
        self,
        tool_calls: list[dict],
        response: dict,
    ) -> str | None:
        """
        Classify failure mode from tool_calls + final response state.

        Returns failure mode string or None if trajectory is clean.
        Priority order: most severe failure mode wins.
        """
        nodes = [tc.get("node") for tc in tool_calls]
        node_counts = Counter(nodes)

        # Tool loop: same node called 3+ times
        # Indicates the agent got stuck retrying a failing operation
        if any(count >= 3 for count in node_counts.values()):
            return "tool_loop"

        # Hallucinated claim: critic flagged unsupported claims
        if response.get("has_hallucination", False):
            return "hallucinated_claim"

        # Premature termination: refiner ran but explanation was empty
        if "refiner" in nodes and not response.get("explanation", "").strip():
            return "premature_termination"

        # Context loss: explainer ran but retrieved_context was empty
        # despite retriever being called
        if (
            "retriever" in nodes
            and "explainer" in nodes
            and not response.get("retrieved_context", [])
        ):
            return "context_loss"

        # Over-delegation: took more than 2x optimal steps
        optimal = 5  # default
        if len(nodes) > optimal * 2:
            return "over_delegation"

        return None  # clean trajectory

    def summarize_failures(self, failure_list: list[str | None]) -> dict:
        """
        Aggregate failure taxonomy across all test cases.
        Returns counts per failure mode — tells you what to fix first.
        """
        counts = Counter(f for f in failure_list if f is not None)
        total = len(failure_list)
        clean = sum(1 for f in failure_list if f is None)

        return {
            "total_cases": total,
            "clean_trajectories": clean,
            "clean_rate": clean / total if total > 0 else 0.0,
            "failure_counts": dict(counts),
            "failure_rates": {
                k: v / total for k, v in counts.items()
            },
        }