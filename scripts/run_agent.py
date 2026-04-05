"""
Run the Multimodal Conversational RecSys Agent
================================================
End-to-end entry point: loads all components and starts interactive session.

Usage:
    # Text mode
    python scripts/run_agent.py

    # Voice mode
    python scripts/run_agent.py --voice

    # Single query
    python scripts/run_agent.py --query "Recommend me psychological thrillers"
"""

import argparse
import yaml
from langchain_openai import ChatOpenAI

from src.agent.graph import build_agent_graph
from src.recsys.serving.candidate_gen import load_candidate_generator
from src.retrieval.hybrid_retrieval import HybridRetriever


def load_components(config: dict):
    """Load all system components."""
    print("Loading components...")

    # LLM backbone
    llm = ChatOpenAI(
        model=config["agent"]["llm_model"],
        temperature=config["agent"]["temperature"],
    )

    # Candidate generator (two-tower + Mult-VAE + Qdrant)
    candidate_gen = load_candidate_generator()

    # Hybrid retriever (dense + BM25 + reranker)
    retriever = HybridRetriever(
        qdrant_url=config["retrieval"]["qdrant_url"],
        collection_name=config["retrieval"]["collection_name"],
    )

    # Build agent graph
    agent = build_agent_graph(llm, retriever, candidate_gen)

    print("✓ All components loaded.")
    return agent


def run_single_query(agent, query: str, user_id: str = "demo_user"):
    """Run a single query through the agent."""
    response = agent.invoke({
        "query": query,
        "user_id": user_id,
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
        "refined_query": query,
        "intent": "recommend",
    })

    print(f"\nAgent: {response['final_response']}")
    print(f"\n[Debug] Nodes executed: {[tc['node'] for tc in response['tool_calls']]}")
    print(f"[Debug] Candidates: {len(response.get('candidates', []))}")
    print(f"[Debug] Hallucination: {response.get('has_hallucination', False)}")
    return response


def run_interactive(agent, user_id: str = "demo_user", voice: bool = False):
    """Run interactive multi-turn session."""
    print("\n" + "="*55)
    print("Multimodal Conversational RecSys Agent")
    print("Type 'quit' to exit, 'voice' to switch to voice mode")
    print("="*55 + "\n")

    state = {
        "query": "",
        "user_id": user_id,
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
        "refined_query": "",
        "intent": "recommend",
    }

    while True:
        if voice:
            print("🎤 Speak now (press Enter when done recording)...")
            # In real implementation: record audio, pass to Whisper
            # For demo: fall back to text
            query = input("You (text fallback): ").strip()
        else:
            query = input("You: ").strip()

        if query.lower() == "quit":
            break

        state["query"] = query
        state["refined_query"] = query
        state["is_refinement"] = state["turn"] > 0

        state = agent.invoke(state)
        print(f"\nAgent: {state['final_response']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--user_id", default="demo_user")
    parser.add_argument("--voice", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    agent = load_components(config)

    if args.query:
        run_single_query(agent, args.query, args.user_id)
    else:
        run_interactive(agent, args.user_id, voice=args.voice)


if __name__ == "__main__":
    main()