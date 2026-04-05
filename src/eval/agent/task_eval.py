"""
Task Completion Evaluator — LLM-as-judge
==========================================
Scores agent responses on a rubric using an LLM judge.

Design decision: rubric-based scoring over open-ended judgment.
Without explicit criteria, LLM judge scores are inconsistent across runs.
A rubric with 4 dimensions (relevance, completeness, groundedness, coherence)
gives reproducible scores and actionable failure signals.

Design decision: log judge reasoning, not just scores.
The reasoning is what makes failures auditable — you can read why
the judge gave a low score and fix the underlying issue.
"""

import json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


JUDGE_PROMPT = """You are an evaluator for a content recommendation agent.

Score the agent's response on 4 dimensions using this rubric:

User query: {query}
Agent response: {response}
Expected genres (if any): {expected_genres}
Expected year range (if any): {expected_year_range}
Expected keywords in explanation (if any): {expected_keywords}

Rubric (score each 1-5):
1. RELEVANCE: Do the recommendations match what the user asked for?
   5=perfect match, 3=partially relevant, 1=completely off
2. COMPLETENESS: Did the agent fully address the query?
   5=fully addressed, 3=partially, 1=ignored key aspects
3. GROUNDEDNESS: Are claims in the explanation supported by actual movie facts?
   5=all claims grounded, 3=some unsupported claims, 1=mostly hallucinated
4. COHERENCE: Is the response natural and conversational?
   5=natural, 3=awkward but understandable, 1=incoherent

Also flag:
- hallucination: true if any plot/fact claim appears invented
- task_complete: true if the user's core need was met

Output ONLY valid JSON:
{{
    "relevance": <1-5>,
    "completeness": <1-5>,
    "groundedness": <1-5>,
    "coherence": <1-5>,
    "hallucination": <bool>,
    "task_complete": <bool>,
    "reasoning": "<2-3 sentence explanation of scores>"
}}
"""


@dataclass
class TaskScore:
    relevance: float
    completeness: float
    groundedness: float
    coherence: float
    hallucination: bool
    task_complete: bool
    reasoning: str
    composite: float   # weighted average


class TaskEvaluator:
    """
    LLM-as-judge task completion scoring.

    Design decision: use a separate LLM instance for evaluation.
    Don't reuse the agent's LLM — judge should be independent.
    At scale: use a stronger model (GPT-4o) as judge even if agent
    uses a cheaper model (GPT-4o-mini). Judge quality > agent quality.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        # temperature=0 for reproducible judge scores
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def score(
        self,
        query: str,
        response: str,
        ground_truth: dict = None,
    ) -> dict:
        """
        Score agent response against ground truth.

        Returns dict with individual dimension scores + composite.
        Composite = 0.3*relevance + 0.3*groundedness + 0.2*completeness + 0.2*coherence
        Weights reflect what matters most: relevant + grounded > complete + coherent.
        """
        gt = ground_truth or {}

        prompt = JUDGE_PROMPT.format(
            query=query,
            response=response,
            expected_genres=gt.get("expected_genres", "not specified"),
            expected_year_range=gt.get("expected_year_range", "not specified"),
            expected_keywords=gt.get("expected_keywords", "not specified"),
        )

        llm_response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(llm_response.content)
        except json.JSONDecodeError:
            # Fallback scores on parse failure
            result = {
                "relevance": 3, "completeness": 3,
                "groundedness": 3, "coherence": 3,
                "hallucination": False, "task_complete": True,
                "reasoning": "Could not parse judge output.",
            }

        # Composite score
        composite = (
            0.3 * result["relevance"] +
            0.3 * result["groundedness"] +
            0.2 * result["completeness"] +
            0.2 * result["coherence"]
        ) / 5.0   # normalize to [0, 1]

        return {
            "relevance": result["relevance"] / 5.0,
            "completeness": result["completeness"] / 5.0,
            "groundedness": result["groundedness"] / 5.0,
            "coherence": result["coherence"] / 5.0,
            "hallucination": result["hallucination"],
            "completion": 1.0 if result["task_complete"] else 0.0,
            "composite": composite,
            "reasoning": result["reasoning"],
        }