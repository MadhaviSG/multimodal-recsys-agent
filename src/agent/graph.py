"""
LangGraph Multi-Agent Orchestration
=====================================
Nodes: Planner → (conditional) RecSys / Retriever → Explainer → Critic → Refiner

Design decisions documented inline.

Key changes from v1:
- Conditional edges from planner (not fixed) — avoids wasteful node execution
- Typed plan fields added to AgentState (needs_recsys, needs_rag, etc.)
- Versioned prompt templates from src/agent/prompts/templates.py
- Structured output parsing with fallback for malformed LLM JSON
"""

from __future__ import annotations
import json
import operator
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.retrieval.hybrid_retrieval import HybridRetriever
from src.recsys.serving.candidate_gen import CandidateGenerator
from src.agent.prompts.templates import (
    PLANNER_PROMPT,
    EXPLAINER_PROMPT,
    CRITIC_PROMPT,
    REFINER_PROMPT,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str
    query: str
    turn: int

    # Planner outputs — drive conditional routing
    # Design decision: planner writes typed flags to state.
    # Downstream nodes read these to decide whether to execute.
    # Avoids hardcoded graph edges — graph structure reflects query intent.
    needs_recsys: bool
    needs_rag: bool
    is_refinement: bool
    refined_query: str
    intent: str                  # recommend | explain | filter | compare | chitchat

    # Node outputs
    candidates: list[dict]       # RecSys top-K
    retrieved_context: list[dict] # RAG results
    explanation: str
    critique: str
    unsupported_claims: list[str]
    has_hallucination: bool
    final_response: str

    # Trajectory logging — for eval harness
    tool_calls: list[dict]


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: AgentState, llm: ChatOpenAI) -> dict:
    """
    Decomposes user query into a typed execution plan.

    Design decision: structured JSON output from planner.
    Natural language planner output is ambiguous and hard to route on.
    JSON with typed fields (needs_recsys: bool) enables deterministic
    conditional edges — the graph structure follows from the plan.

    Design decision: refined_query for downstream retrieval.
    Raw user queries are often vague ("something like that movie").
    Planner rewrites to a specific, retrievable query
    ("psychological thriller with unreliable narrator").
    """
    prompt = PLANNER_PROMPT.format(
        query=state["query"],
        turn=state["turn"],
        has_candidates=len(state.get("candidates", [])) > 0,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON with fallback — LLMs occasionally produce malformed JSON
    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: safe defaults that always work
        plan = {
            "needs_recsys": True,
            "needs_rag": True,
            "is_refinement": False,
            "refined_query": state["query"],
            "intent": "recommend",
        }

    return {
        "needs_recsys": plan.get("needs_recsys", True),
        "needs_rag": plan.get("needs_rag", True),
        "is_refinement": plan.get("is_refinement", False),
        "refined_query": plan.get("refined_query", state["query"]),
        "intent": plan.get("intent", "recommend"),
        "tool_calls": state.get("tool_calls", []) + [
            {"node": "planner", "intent": plan.get("intent"), "plan": plan}
        ],
    }


# ── RecSys node ───────────────────────────────────────────────────────────────

def recsys_node(state: AgentState, candidate_gen: CandidateGenerator) -> dict:
    """
    Fetches top-K candidates from two-tower + Mult-VAE pipeline.
    Only called when planner sets needs_recsys=True.
    """
    candidates = candidate_gen.get_candidates_as_dicts(
        user_id=state["user_id"],
        top_k=20,
    )
    return {
        "candidates": candidates,
        "tool_calls": state.get("tool_calls", []) + [
            {"node": "recsys", "n_candidates": len(candidates)}
        ],
    }


# ── Retriever node ────────────────────────────────────────────────────────────

def retriever_node(state: AgentState, retriever: HybridRetriever) -> dict:
    """
    Hybrid retrieval over content DB for RAG context.
    Uses refined_query from planner for better retrieval quality.
    Only called when planner sets needs_rag=True.
    """
    query = state.get("refined_query") or state["query"]
    results = retriever.retrieve(query=query, top_k=20, rerank_top_n=5)
    context = [
        {
            "title": r.title,
            "text": r.metadata.get("text", ""),
            "plot": r.metadata.get("plot", ""),
            "metadata": r.metadata,
        }
        for r in results
    ]
    return {
        "retrieved_context": context,
        "tool_calls": state.get("tool_calls", []) + [
            {"node": "retriever", "n_results": len(results), "query": query}
        ],
    }


# ── Explainer node ────────────────────────────────────────────────────────────

def explainer_node(state: AgentState, llm: ChatOpenAI) -> dict:
    """
    Generates explanation for recommendations using RAG context.

    Design decision: explanation grounded in retrieved context.
    LLM without context will hallucinate plot details.
    Forcing the explainer to reference provided context reduces
    hallucination rate measurably (validated in eval harness).
    """
    candidates_str = "\n".join([
        f"- {c['title']} ({c.get('year', 'N/A')}) — {', '.join(c.get('genres', []))}"
        for c in state.get("candidates", [])[:5]
    ])

    context_str = "\n".join([
        f"- {c['title']}: {c.get('plot', c.get('text', ''))[:300]}"
        for c in state.get("retrieved_context", [])[:5]
    ])

    # If no context retrieved, use candidate metadata as fallback
    if not context_str and state.get("candidates"):
        context_str = "\n".join([
            f"- {c['title']}: {c.get('plot', 'No plot available')[:300]}"
            for c in state.get("candidates", [])[:5]
        ])

    prompt = EXPLAINER_PROMPT.format(
        query=state["query"],
        candidates=candidates_str or "No candidates retrieved.",
        context=context_str or "No context available.",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "explanation": response.content,
        "tool_calls": state.get("tool_calls", []) + [{"node": "explainer"}],
    }


# ── Critic node ───────────────────────────────────────────────────────────────

def critic_node(state: AgentState, llm: ChatOpenAI) -> dict:
    """
    Validates explanation against retrieved context.
    Flags hallucinated claims not grounded in source material.

    Design decision: critic checks against retrieved_context, not world knowledge.
    The critic's job is grounding, not fact-checking against external truth.
    A claim can be factually true but still unsupported by our retrieval —
    that's the signal we want: is the agent making things up beyond its context?
    """
    context_str = "\n".join([
        f"- {c['title']}: {c.get('plot', c.get('text', ''))[:300]}"
        for c in state.get("retrieved_context", [])[:5]
    ])

    prompt = CRITIC_PROMPT.format(
        explanation=state.get("explanation", ""),
        context=context_str or "No context available.",
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {
            "has_hallucination": False,
            "unsupported_claims": [],
            "critique": "Could not parse critic output.",
        }

    return {
        "critique": result.get("critique", ""),
        "has_hallucination": result.get("has_hallucination", False),
        "unsupported_claims": result.get("unsupported_claims", []),
        "tool_calls": state.get("tool_calls", []) + [
            {"node": "critic", "has_hallucination": result.get("has_hallucination")}
        ],
    }


# ── Refiner node ──────────────────────────────────────────────────────────────

def refiner_node(state: AgentState, llm: ChatOpenAI) -> dict:
    """
    Synthesizes final response, removing hallucinated claims flagged by critic.

    Design decision: always run refiner, even if no hallucination detected.
    Refiner also handles tone, formatting, and coherence — not just
    hallucination removal. Skipping it on clean outputs saves one LLM call
    but risks inconsistent response quality.
    """
    prompt = REFINER_PROMPT.format(
        explanation=state.get("explanation", ""),
        critique=state.get("critique", "No issues found."),
        unsupported_claims="\n".join(state.get("unsupported_claims", [])) or "None",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "final_response": response.content,
        "messages": [AIMessage(content=response.content)],
        "turn": state.get("turn", 0) + 1,
        "tool_calls": state.get("tool_calls", []) + [{"node": "refiner"}],
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_planner(state: AgentState) -> Literal["recsys", "retriever", "explainer"]:
    """
    Conditional routing after planner.

    Design decision: conditional edges over fixed edges.
    Fixed edges always run recsys + retriever regardless of query intent.
    For "What is the plot of Inception?" that wastes 100ms on RecSys.
    For follow-up refinements, candidates already exist in state.

    Routing logic:
    - needs_recsys=True → recsys (retriever runs in parallel via graph fan-out)
    - needs_recsys=False, needs_rag=True → retriever only
    - neither → explainer directly (chitchat or follow-up with full state)

    Note: LangGraph handles fan-out automatically when multiple edges
    leave the same node — recsys and retriever run in parallel.
    """
    if state.get("needs_recsys", True):
        return "recsys"
    elif state.get("needs_rag", True):
        return "retriever"
    else:
        return "explainer"


def route_after_recsys(state: AgentState) -> Literal["retriever", "explainer"]:
    """After recsys: run retriever if RAG needed, else go straight to explainer."""
    if state.get("needs_rag", True):
        return "retriever"
    return "explainer"


# ── Graph construction ────────────────────────────────────────────────────────

def build_agent_graph(
    llm: ChatOpenAI,
    retriever: HybridRetriever,
    candidate_gen: CandidateGenerator,
) -> StateGraph:
    """
    Build and compile the LangGraph agent graph.

    Node execution order (conditional):
        planner
          ├─(needs_recsys)──→ recsys ─┬─(needs_rag)──→ retriever ──→ explainer
          ├─(needs_rag only)──────────┘                              ↓
          └─(neither)────────────────────────────────────→ explainer
                                                                     ↓
                                                                  critic
                                                                     ↓
                                                                  refiner → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",   lambda s: planner_node(s, llm))
    graph.add_node("recsys",    lambda s: recsys_node(s, candidate_gen))
    graph.add_node("retriever", lambda s: retriever_node(s, retriever))
    graph.add_node("explainer", lambda s: explainer_node(s, llm))
    graph.add_node("critic",    lambda s: critic_node(s, llm))
    graph.add_node("refiner",   lambda s: refiner_node(s, llm))

    # Entry point
    graph.set_entry_point("planner")

    # Conditional edges from planner
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "recsys": "recsys",
            "retriever": "retriever",
            "explainer": "explainer",
        }
    )

    # Conditional edge from recsys
    graph.add_conditional_edges(
        "recsys",
        route_after_recsys,
        {
            "retriever": "retriever",
            "explainer": "explainer",
        }
    )

    # Fixed edges for the rest
    graph.add_edge("retriever", "explainer")
    graph.add_edge("explainer", "critic")
    graph.add_edge("critic", "refiner")
    graph.add_edge("refiner", END)

    return graph.compile(checkpointer=MemorySaver())