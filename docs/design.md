# Project Design Document
**Multimodal Conversational RecSys Agent**
*Version 1.0 — written before any code*

---

## 1. Problem Statement

Users of content platforms face two distinct but related problems:

**Discovery paralysis** — A catalog of 10,000+ titles is overwhelming. Users default to re-watching familiar content or abandoning the platform. The system must surface content the user didn't know they wanted.

**Articulation gap** — Users often can't express what they want in a search query. "Something like that movie I watched last year with the twist ending" is not a query a traditional search system handles. The system must meet users where their intent is vague and help them converge on something concrete through conversation.

These two problems compound for **new users** (no history, no signal) and **business analysts** (need trend insight, not personal recommendations).

---

## 2. Users

| User Type | Primary Need | Success Looks Like |
|---|---|---|
| Content consumer | Find something to watch right now | Watches and finishes the recommendation |
| Business analyst | Understand content trends, catalog gaps | Actionable insight from natural language query |

Both users interact via the same interface — the agent handles intent classification internally.

---

## 3. Input Modalities

| Modality | Example | Why It Matters |
|---|---|---|
| Text query | "Show me psychological thrillers from the last 5 years" | Primary interface, lowest friction |
| Voice query | Spoken version of above | Accessibility, mobile-first use |
| Image input | Upload a poster: "more like this" | Bypasses the articulation gap entirely |
| User history | Implicit from past interactions | Richest signal for personalization |

**Design implication:** All modalities must resolve to the same internal representation — a user intent vector — before hitting the RecSys layer. The multimodal I/O layer is a translation layer, not a separate recommendation path.

---

## 4. Success Metrics

We reject single-metric optimization. Systems optimized purely for clicks produce clickbait. Systems optimized purely for watch time surface addictive but low-quality content. We define a **composite metric** across three dimensions:

### 4.1 Engagement (did they interact?)
- **Click-through rate (CTR)** on recommendations
- **Watch completion rate** — did they finish what we recommended?

### 4.2 Satisfaction (did they like it?)
- **Explicit rating** (thumbs up/down, stars)
- **Implicit signal**: re-watch, share, add to watchlist

### 4.3 Discovery (did they find something new?)
- **Novelty**: average self-information of recommended items (-log(popularity))
- **Serendipity**: relevant AND outside the user's typical taste profile
- **Catalog coverage**: what fraction of the catalog gets recommended across all users

### 4.4 Composite Score (for offline eval)
```
composite = 0.4 * (NDCG@10) + 0.3 * (serendipity) + 0.3 * (novelty)
```
Weights are a starting hypothesis — revisit after first eval run.

**Why this matters in interviews:** Most candidates say "I used NDCG." Being able to say "I rejected single-metric optimization because it misaligns with business goals, so I defined a composite metric and here's how I weighted it and why" is a Staff-level answer.

---

## 5. Constraints

### Latency
- **Hard constraint: <500ms end-to-end** for recommendation response
- Implication: candidate generation must be ANN (not exact search), LLM backbone must be quantized, KV cache must be managed carefully for multi-turn
- Agent explanation (RAG) can be streamed — first token in <500ms, full response in <2s

### Cold Start
Cold start is a **fallback chain**, not a single strategy.

| Step | Strategy | Signal Available | Notes |
|---|---|---|---|
| 1 | Structured onboarding (3-5 questions) | Genre, mood, language, rating preference | Users may answer carelessly — treat as weak signal |
| 2 | Content-based filtering | Item features (genre, cast, plot embeddings) | No user signal needed — degrades gracefully |
| 3 | Diffusion augmentation | Synthetic history from onboarding answers | DiffRec-inspired — generates plausible interaction sequences |
| 4 | Popularity baseline (segmented by region/device) | None | Last resort — same for everyone in segment |

**Why onboarding alone is insufficient:** Users click through quickly, answers are static, and 3-5 questions give sparse signal. The fallback chain ensures we always have *something* useful, and we transition to collaborative filtering as soon as real interactions accumulate.

**Why diffusion augmentation is the research-grade contribution:** Rather than cold-starting with zeros, we generate a synthetic interaction history consistent with the user's onboarding answers — giving the Mult-VAE meaningful input from session one.

Cold start is a first-class problem, not an afterthought.

### Scale (design for, not build for)
- MVP: 1K users, MovieLens 25M dataset
- Design decisions should hold at: 10M users, 1M item catalog, 10K QPS
- Document where the current implementation breaks at scale and what you'd change

---

## 6. What Success Is NOT

Explicitly defining non-goals prevents scope creep and sharpens the system:

- ❌ We are not building a content streaming platform
- ❌ We are not optimizing for ad revenue or sponsored content
- ❌ We are not solving real-time personalization (model updates are batch, not online learning — yet)
- ❌ We are not building a general-purpose chatbot — the agent is scoped to content recommendation and explanation

---

## 7. ML Problem Framing

### 7.1 Recommendation as retrieval + ranking
The core recommendation task decomposes into two stages:

**Stage 1 — Candidate Generation (recall)**
- Goal: retrieve top-K (~100-500) relevant items from a catalog of 10K+
- Method: two-tower model + ANN (Qdrant HNSW)
- Constraint: must run in <50ms (leaves budget for ranking + LLM)
- Metric: Recall@500 (did the relevant item make it through?)

**Stage 2 — Ranking (precision)**
- Goal: rerank top-K candidates to top-N (N=10-20) for display
- Method: LightGBM with rich features (RecSys score, recency, diversity, VAE distance)
- Constraint: must run in <100ms
- Metric: NDCG@10, composite score

**Why two stages?**
A single end-to-end model over the full catalog is computationally infeasible at inference time — scoring every item for every query at <500ms doesn't scale. Two stages let you use a fast approximate model for recall and a more expensive precise model for ranking. This is the architecture used by Netflix, YouTube, Spotify, LinkedIn.

### 7.2 User modeling
- **Mult-VAE** (Liang et al., 2018): learns a *distribution* over preferences, not a point estimate
- Better generalization with sparse interaction data (most users have <50 interactions)
- Latent vector used as user tower input to two-tower model
- At scale: would explore sequential models (SASRec, BERT4Rec) for richer temporal modeling

### 7.3 Conversational layer
- LLM agent handles: intent clarification, explanation, multi-turn refinement
- Not a replacement for RecSys — a layer on top of it
- Agent gets RecSys candidates as context, uses RAG to explain them, uses critic to ground claims

---

## 8. High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│         Text · Voice (Whisper) · Image (VLM)        │
└──────────────────────┬──────────────────────────────┘
                       │ unified intent vector
┌──────────────────────▼──────────────────────────────┐
│                  Agent Layer                         │
│   Planner → Retriever → Explainer → Critic →        │
│   Refiner  (LangGraph, LangSmith tracing)           │
└──────────┬───────────────────────┬──────────────────┘
           │                       │
┌──────────▼──────────┐ ┌──────────▼──────────────────┐
│    RecSys Layer     │ │      RAG / Retrieval Layer   │
│  Mult-VAE + Two-    │ │  Qdrant HNSW + BM25 + RRF   │
│  Tower → LightGBM  │ │  + cross-encoder reranker    │
└──────────┬──────────┘ └──────────┬──────────────────┘
           │                       │
┌──────────▼───────────────────────▼──────────────────┐
│              Inference Layer                         │
│    Quantized LLM (int4) · KV Cache · Triton kernel  │
└─────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────┐
│                   Eval Framework                     │
│  RecSys · Agent · RAG · Adversarial · W&B Weave     │
└─────────────────────────────────────────────────────┘
```

---

## 9. Data

| Dataset | Purpose | Size |
|---|---|---|
| MovieLens 25M | User-item interactions for RecSys training | 25M ratings, 162K users, 62K movies |
| TMDB API | Movie posters (VLM), plot summaries (RAG) | ~500K movies |
| Amazon Reviews (Movies & TV) | Review text for RAG content DB | ~8M reviews |

### 9.1 Train/Val/Test Split — Temporal, Not Random

**This is an ML system design decision.**

Random splitting leaks future interactions into training — the model sees ratings made *after* a user's early interactions, which it wouldn't have at inference time. This inflates offline metrics and causes the model to underperform in production.

**Correct approach: temporal split**
- Train: all interactions before timestamp T1 (e.g. before 2018)
- Val: interactions between T1 and T2 (2018–2019)
- Test: interactions after T2 (2019+)

This simulates real deployment: model trained on the past, evaluated on the future.

---

## 10. Eval Plan

*Defined before building — not retrofitted after.*

| Layer | Metrics | Tool |
|---|---|---|
| RecSys | NDCG@10, Recall@10, Coverage, Serendipity, Novelty | Custom harness |
| Agent task | Completion rate, hallucination rate (LLM-as-judge) | LangSmith + Weave |
| Agent trajectory | Step efficiency, tool call correctness, failure taxonomy | Custom harness |
| RAG | Faithfulness, context precision/recall, answer relevancy | RAGAS |
| Adversarial | Prompt injection rate, GAN user robustness | Custom harness |

**Iteration loop:**
```
Build → Eval v1 → Failure taxonomy → Fix top 2 issues → Eval v2 → measure delta
```
Minimum: two eval runs with a documented before/after delta.

---

## 11. Scale Considerations

*We build for MVP, but design decisions should hold at scale. Document the breakpoints.*

| Component | Breaks at scale because | What you'd do |
|---|---|---|
| Qdrant single node | Memory limit ~100M vectors | Distributed Qdrant cluster or Faiss with IVF-PQ |
| LangGraph in-process | No horizontal scaling | Deploy as microservice, add message queue |
| Batch user embeddings | Stale if user interacts frequently | Introduce feature store (Feast, Tecton) for real-time features |
| Single LLM endpoint | Throughput bottleneck | vLLM with continuous batching, multiple replicas |
| Offline eval only | Doesn't catch distribution shift | Shadow deployment + online A/B test framework |

---

## 12. Open Questions

Things we'll answer as we build — document them now so we don't forget:

- [ ] What's the right latent dimension for Mult-VAE? (hypothesis: 64, validate via ablation)
- [ ] How many candidates should Stage 1 return? (hypothesis: 200, validate via Recall@200)
- [ ] What's the KL annealing schedule? (hypothesis: linear over 10K steps)
- [ ] How do we handle items with no text metadata for RAG? (fallback to title + genre only)
- [ ] What's the right composite metric weighting? (0.4/0.3/0.3 — revisit after eval v1)

---

*This document is a living artifact. Update it when decisions change and record why.*