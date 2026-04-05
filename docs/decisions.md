# Design Decisions Log
**Multimodal Conversational RecSys Agent**

A living document. Every meaningful engineering decision recorded here with alternatives considered and reasoning. Updated as we build.

---

## How to Use This Document

Each decision follows this format:

- **Decision:** What we chose
- **Alternatives considered:** What else we evaluated
- **Why:** The reasoning
- **What breaks at scale:** Where this decision stops holding
- **Interview note:** How to talk about this in a system design round

---

## 1. Success Metric: Composite Score Over Single Metric

**Decision:** Define a composite offline metric: `0.4 * NDCG@10 + 0.3 * serendipity + 0.3 * novelty`

**Alternatives considered:**
- NDCG@10 only (standard academic baseline)
- Click-through rate only (engagement proxy)
- Watch completion rate only (satisfaction proxy)

**Why:**
Single-metric optimization produces misaligned systems. A system optimized purely for NDCG@10 will recommend popular items — high precision, low discovery value. A system optimized for watch time will surface addictive but low-quality content. The composite metric captures three business dimensions simultaneously: ranking quality, discovery, and catalog diversity.

Weights (0.4/0.3/0.3) are a starting hypothesis — revisit after eval v1.

**What breaks at scale:**
Composite weights need to be tuned to actual business outcomes via online A/B testing. Offline composite is a proxy, not ground truth. At scale you'd run holdout experiments to validate that offline composite improvements translate to online engagement gains.

**Interview note:**
*"I rejected single-metric optimization because it misaligns with business goals. Most candidates say they used NDCG — the follow-up question is always 'what about catalog diversity?' Being able to define a composite metric and explain the weighting rationale is a Staff-level answer."*

---

## 2. Cold Start: Fallback Chain Over Single Strategy

**Decision:** Cold start is a four-step fallback chain, not a single strategy:
1. Structured onboarding (3-5 questions) → weak preference vector
2. Content-based filtering from item features → no user signal needed
3. Diffusion augmentation → synthetic interaction history (DiffRec-inspired)
4. Popularity baseline segmented by region/device → last resort

**Alternatives considered:**
- Onboarding questions only
- Popularity baseline only
- Transfer from similar users (meta-learning / user-kNN)
- VLM poster grid (show posters, user picks what looks interesting)

**Why:**
Onboarding questions alone are insufficient — users answer carelessly, signal is sparse and static, and 3-5 questions give very limited coverage of preference space. Each step in the fallback chain has different signal availability and quality. The chain ensures we always have *something* useful and degrades gracefully as signal decreases. We transition to collaborative filtering as soon as real interactions accumulate.

Diffusion augmentation is the research-grade contribution: rather than cold-starting with zeros, we generate a synthetic interaction history consistent with onboarding answers, giving Mult-VAE meaningful input from session one.

**What breaks at scale:**
Onboarding drop-off increases with friction — at scale you'd A/B test onboarding question count vs. recommendation quality to find the optimal friction/signal tradeoff. Diffusion model needs retraining as user population distribution shifts.

**Interview note:**
*"Cold start is a fallback chain, not a single strategy. I can walk you through each step and what signal it requires. The interesting part is the diffusion augmentation — rather than popularity-based cold start, we generate a synthetic interaction history..."*

---

## 3. Train/Val/Test Split: Temporal, Not Random

**Decision:** Sort all interactions by timestamp globally, split 80/10/10 by time.

**Alternatives considered:**
- Random split (standard in academic benchmarks)
- Per-user leave-one-out (last interaction as test)
- Per-user leave-last-N (last N interactions as test)

**Why:**
Random splitting leaks future interactions into training. The model sees ratings made *after* a user's early interactions — information it wouldn't have at inference time. This inflates offline metrics and causes the model to underperform in production relative to offline eval.

Temporal split simulates real deployment: train on the past, evaluate on the future. This is the correct evaluation methodology for any time-series interaction data.

Per-user leave-one-out is more rigorous (standard in academic RecSys papers) but more complex to implement and less representative of production conditions where multiple future interactions exist.

**What breaks at scale:**
If the dataset has strong recency skew (most interactions in the last 6 months), the 10% test window may be very short. At scale, pin val/test to specific calendar date ranges rather than percentile splits.

**Interview note:**
*"Random splitting is a common mistake in RecSys evaluation — it leaks future data into training and inflates offline metrics. Temporal split is the correct approach because it simulates real deployment conditions."*

---

## 4. User Modeling: Mult-VAE Over Standard Matrix Factorization

**Decision:** Use Mult-VAE (Liang et al., 2018) for user modeling — learns a distribution over preferences rather than a point estimate.

**Alternatives considered:**
- Standard matrix factorization (SVD, ALS)
- Neural collaborative filtering (NCF)
- Sequential models (SASRec, BERT4Rec)
- Standard two-tower with user embedding lookup

**Why:**
Standard MF learns a fixed vector per user — a point estimate of preferences. Mult-VAE learns a distribution (mean + variance), which has two advantages:
1. **Sparse data generalization:** Most users have <50 interactions. A distribution over preferences generalizes better than a point estimate when data is scarce.
2. **Uncertainty-aware recommendations:** The variance captures preference uncertainty — useful for exploration/exploitation decisions downstream.

The multinomial likelihood (vs. Gaussian) is well-suited for implicit feedback — we're modeling which items a user interacts with, not predicting exact rating values.

Sequential models (SASRec) would be the production upgrade — they capture temporal preference drift. Left as a future extension; noted in open questions.

**What breaks at scale:**
Mult-VAE requires the full item vocabulary as input dimension — doesn't scale beyond ~500K items without architectural changes (sampled softmax, hierarchical decoding). At Netflix scale (millions of titles), you'd move to a two-stage approach with item sampling during training.

**Interview note:**
*"I implemented Mult-VAE from the Netflix research paper. The key insight is learning a distribution over preferences rather than a point estimate — this gives better generalization for the majority of users who have sparse interaction histories."*

---

## 5. Candidate Generation: Two-Stage Retrieval + Ranking

**Decision:** Two-stage pipeline — Stage 1: ANN retrieval (two-tower + Qdrant HNSW), Stage 2: LightGBM reranker.

**Alternatives considered:**
- Single end-to-end model scoring all items
- Pure collaborative filtering with no reranking
- Learning-to-rank with neural reranker

**Why:**
A single model scoring every item at inference time is computationally infeasible at scale. Scoring 62K items per query at <500ms is borderline; at 1M items it's impossible without approximation.

Two stages let you use a fast approximate model for recall (ANN in <50ms) and a more expensive precise model for ranking (LightGBM on top-K candidates in <100ms). This is the architecture used by Netflix, YouTube, Spotify, and LinkedIn.

LightGBM reranker (vs. neural reranker) is a deliberate choice: interpretable features, fast inference, no GPU required for ranking, easier to debug. Neural reranker would give better accuracy at the cost of latency and interpretability.

**What breaks at scale:**
At 10M+ items, HNSW index may exceed single-node memory. Move to IVF-PQ (compressed vectors, lower recall) or distributed HNSW. Reranker features need a feature store at high QPS — precomputing and caching item features becomes critical.

**Interview note:**
*"This is the standard two-stage architecture. Stage 1 optimizes for recall — did the relevant item make it through? Stage 2 optimizes for precision — is it ranked at the top? The key insight is that you can't afford to run your expensive ranking model over the full catalog."*

---

## 6. Vector DB: Qdrant Over Pinecone

**Decision:** Self-hosted Qdrant with HNSW indexing.

**Alternatives considered:**
- Pinecone (managed, fully hosted)
- Weaviate (managed + self-hosted)
- FAISS (library, not a DB)
- pgvector (Postgres extension)

**Why:**
- **HNSW parameter control:** Qdrant exposes `m` and `ef_construction` — tunable recall/latency/memory tradeoffs. Pinecone abstracts this away.
- **No egress cost:** At high retrieval volume, Pinecone egress costs become significant. Self-hosted Qdrant has no per-query cost.
- **Native hybrid search:** Qdrant supports dense + sparse retrieval natively, which we need for RRF fusion.
- **Research control:** For a portfolio project, understanding the internals matters more than operational simplicity.

Pinecone is the right choice for managed production scale where operational burden matters more than cost/control.

FAISS is a library, not a database — no persistence, no filtering, no payload storage. Right for prototyping, not for a system with metadata filtering needs.

**What breaks at scale:**
Single-node Qdrant has a memory limit (~100M vectors at 768-dim float32 ≈ 288GB). At scale: distributed Qdrant cluster, or switch to IVF-PQ compressed index in FAISS for memory efficiency at the cost of recall.

**Interview note:**
*"I chose Qdrant over Pinecone for three reasons: HNSW parameter control, no egress cost at high retrieval volume, and native hybrid search support. Pinecone is the right answer for managed production — Qdrant is right when you want systems-level control."*

---

## 7. Retrieval: Hybrid (Dense + BM25 + RRF) Over Dense-Only

**Decision:** Reciprocal Rank Fusion over dense (bge-large) + sparse (BM25) retrieval, with cross-encoder reranker on top.

**Alternatives considered:**
- Dense retrieval only
- BM25 only
- Learned sparse retrieval (SPLADE)
- ColBERT (late interaction)

**Why:**
Dense retrieval alone fails on exact match queries — "Inception 2010" or "Christopher Nolan films" where the exact token matters. BM25 alone misses semantic similarity — "movies about dreams" won't match "Inception" on keyword overlap alone.

RRF fusion recovers the best of both with no additional training — it's a parameter-free combination that consistently outperforms either method alone. Validated with RAGAS context precision metrics.

Cross-encoder reranker on top-K candidates adds a final precision boost — it's expensive (O(n) LLM forward passes) so we run it only on the top 10-20 candidates from RRF.

**What breaks at scale:**
BM25 index is in-memory — doesn't scale beyond ~10M documents without sharding. At scale: Elasticsearch for distributed BM25, or learned sparse (SPLADE) which fits in the same vector index as dense embeddings.

**Interview note:**
*"Dense retrieval has a well-known failure mode on exact match queries. Hybrid retrieval with RRF fusion is the standard fix — it's parameter-free and consistently outperforms either method in isolation. I validated this with RAGAS context precision metrics."*

---

## 8. Sparse Data Handling: Sparse CPU → Dense GPU Per Batch

**Decision:** Keep interaction matrix as scipy CSR sparse on CPU. Convert only the current batch to dense and move to GPU.

**Alternatives considered:**
- Load full dense matrix into GPU memory
- Use PyTorch sparse tensors end-to-end
- Pre-shard matrix into dense chunks on disk

**Why:**
Dense float32 matrix at 162K users × 62K items = ~38GB. Doesn't fit in CPU RAM, let alone GPU VRAM. Sparse CSR format stores only non-zero values — ~50MB for MovieLens 25M.

Per-batch dense conversion costs ~120MB GPU memory at batch_size=512 — fits comfortably. The conversion overhead is negligible compared to forward/backward pass time.

PyTorch sparse tensors are not yet well-supported in all operations (especially autograd through custom ops) — scipy CSR + dense conversion is more reliable.

**What breaks at scale:**
At 10M users, even the sparse matrix may exceed RAM. At scale: shard by user ID, load shards on demand, or use a feature store that serves user interaction vectors pre-computed.

**Interview note:**
*"The interaction matrix doesn't fit in memory as dense. Standard approach: keep it sparse on CPU, convert each batch to dense on-the-fly. The key insight is that sparsity in RecSys data is extreme — typically <0.1% density — so sparse format gives 1000x memory reduction."*

---

## 9. KL Annealing: Linear Schedule to Prevent Posterior Collapse

**Decision:** Linear KL annealing from beta=0 → beta=1 over 10K training steps.

**Alternatives considered:**
- No annealing (beta=1 from start)
- Cyclical annealing (beta oscillates)
- Fixed beta < 1 (beta-VAE)

**Why:**
Without annealing, the KL term dominates early training and forces the encoder to map every user to the prior N(0,1). The decoder then learns to reconstruct from the prior alone — producing identical latent codes and identical recommendations for all users. This is posterior collapse.

Annealing lets the model first learn reconstruction (beta=0 → pure reconstruction loss), then gradually introduce regularization. By the time KL weight is significant, the decoder is already producing good reconstructions and resists collapsing.

Linear over 10K steps is the recommendation from the original paper — treat as a hyperparameter and validate via ablation.

**What breaks at scale:**
Optimal annealing schedule depends on dataset size and model capacity. At scale, monitor KL term magnitude during training — if it collapses to zero, annealing is too fast.

**Interview note:**
*"Posterior collapse is the main failure mode in VAE training. The symptom is that the KL term goes to zero — the encoder ignores the input and maps everything to the prior. Linear KL annealing is the standard fix from the original Mult-VAE paper."*

---

## 10. Validation: Fixed User Panel Over Full Evaluation

**Decision:** Sample 1K users once with fixed seed at training start. Evaluate NDCG@10 every 5 epochs on this panel only.

**Alternatives considered:**
- Full evaluation on all val users every epoch
- Random resample of val users each epoch
- Evaluate on train loss only

**Why:**
Full evaluation on 162K users every epoch is slow (~8 min) and wasteful. 1K users gives statistically stable NDCG estimates with ~5s eval time.

Fixed seed ensures the panel is identical across all runs — making NDCG curves directly comparable between experiments. Random resampling per epoch introduces noise that makes it hard to distinguish real improvements from sampling variance.

Evaluating on train loss only doesn't catch overfitting — train loss can decrease while val NDCG degrades.

**What breaks at scale:**
1K users may not represent tail user behavior (very sparse or very dense interaction histories). At scale: stratified sampling across user activity buckets — ensure the panel covers cold, warm, and power users proportionally.

**Interview note:**
*"We fix the validation panel at training start with a fixed seed. This makes val NDCG curves directly comparable across experiments — you can't do that if you resample every epoch. It's a small detail that matters a lot for reproducibility."*

---

## 11. Codebase Structure: Separation of Concerns

**Decision:** Model definitions in `src/recsys/models/`, training scripts in `scripts/`, evaluation in `src/eval/`.

**Alternatives considered:**
- Monolithic training script (model + training loop in one file)
- Jupyter notebooks for all development

**Why:**
Model files answer: *"what is the model?"* Training scripts answer: *"how do we train it?"* Evaluation files answer: *"how do we measure it?"* Keeping these separate means:
- The same model file is imported by training, evaluation, and serving — single source of truth
- Training strategy can change without touching model architecture
- Evaluation logic is reusable across different models
- Notebooks are for exploration only — production code lives in `.py` files

**What breaks at scale:**
At scale you'd introduce a model registry (MLflow, W&B Artifacts) to version model artifacts alongside code. Configuration management moves to a dedicated system (Hydra, ConfigParser) rather than a single YAML file.

**Interview note:**
*"I separated model definitions from training logic deliberately — the same model file is used by training, evaluation, and serving. This is standard practice and it matters when you're iterating: you can change your training strategy without touching the model, and vice versa."*

---

*Last updated: Session 2 — Data pipeline + Mult-VAE training*
*Next update: Two-tower training decisions*

---

## 12. Why Two-Tower for ANN and Mult-VAE for Re-scoring (Not Vice Versa)

**Decision:** Two-tower handles ANN retrieval. Mult-VAE re-scores the candidate set.

**Alternatives considered:**
- Mult-VAE for retrieval, two-tower for re-scoring
- Mult-VAE only (full catalog scoring at query time)
- Two-tower only (no Mult-VAE re-scoring)

**Why:**
Two-tower produces a single dense vector per user — directly compatible with ANN indexing. You hand Qdrant a 64-dim vector, it finds nearest neighbors in <50ms.

Mult-VAE produces a score distribution over all 62K items simultaneously. It cannot produce a single query vector — it requires the full interaction history as input and outputs scores for every item. To use it for retrieval you'd score all 62K items at query time — incompatible with ANN and too slow for <500ms SLA.

On the candidate set (200 items), Mult-VAE re-scoring is cheap (<10ms) and adds personalization signal that the two-tower embedding misses.

**What breaks at scale:**
At 10M items, even Mult-VAE re-scoring on 200 candidates is fine. The bottleneck shifts to feature serving — loading the user interaction vector from the sparse matrix doesn't scale. Replace with a feature store (Feast, Redis) serving pre-computed user vectors.

**Interview note:**
*"Two-tower produces a vector — ANN-compatible. Mult-VAE produces a score distribution — not ANN-compatible. That's the core reason. We use each model where it naturally fits."*

---

## 13. Score Fusion: Weighted Combination Over Learned Fusion

**Decision:** `combined = alpha * retrieval_score + (1-alpha) * sigmoid(vae_score)`, alpha=0.5.

**Alternatives considered:**
- Learned fusion (small MLP trained on score pairs)
- Rank-based fusion (like RRF — 1/(k + rank))
- VAE score only for re-ranking

**Why:**
Weighted combination is simple, interpretable, and requires no additional training data or model. Alpha=0.5 is a starting hypothesis — tune via offline NDCG ablation.

Learned fusion would give better results but requires labeled data (user feedback on candidate pairs) and adds a training dependency to the serving pipeline.

Sigmoid normalization on VAE scores puts them on the same [0,1] scale as retrieval scores — necessary for fair weighting.

**What breaks at scale:**
Alpha=0.5 is not optimal. At scale: grid search alpha on val set, or train a shallow fusion model on logged user feedback signals (clicks, completions).

**Interview note:**
*"We use a simple weighted combination with a tunable alpha. The important detail is normalization — VAE outputs logits, retrieval scores are cosine similarities. You need them on the same scale before combining."*

---

## 14. Agent Routing: Conditional Edges Over Fixed Edges

**Decision:** Planner writes typed flags (needs_recsys, needs_rag) to state. Conditional edges route based on these flags.

**Alternatives considered:**
- Fixed edges (always run all nodes)
- Router LLM (separate model decides routing)
- Rule-based routing without LLM planner

**Why:**
Fixed edges always run RecSys + retriever regardless of query intent. For "What is the plot of Inception?" that wastes ~100ms on RecSys and returns irrelevant candidates to the explainer. For follow-up queries where candidates already exist in state, re-running RecSys gives identical results.

Typed flags (bool) in state enable deterministic conditional edges — no ambiguity in routing. The planner LLM decides intent once; routing logic is pure Python.

**What breaks at scale:**
Planner LLM adds ~200ms latency. At scale: cache planner decisions for common query patterns, or use a lightweight classifier instead of a full LLM for routing.

**Interview note:**
*"Fixed edges in LangGraph are simple but wasteful — you run every node regardless of whether it's needed. Conditional edges let the graph structure reflect the query intent. The planner writes typed flags to state, routing is deterministic Python."*

---

## 15. Prompts: Versioned Templates Over Inline Strings

**Decision:** All prompts live in src/agent/prompts/templates.py, not inline in node functions.

**Alternatives considered:**
- Inline f-strings in each node function
- External prompt files (.txt or .yaml)
- LangChain PromptTemplate objects

**Why:**
Inline prompts make A/B testing impossible — you'd have to change node logic to test a prompt variant. Versioned templates are independently changeable, inspectable, and testable. A prompt engineer can iterate on templates without touching orchestration logic.

External files (.txt) add file I/O and make the codebase harder to navigate. Python constants in a module are the right balance — version controlled, importable, easy to read.

**What breaks at scale:**
At scale: move prompts to a prompt management system (LangSmith Hub, Humanloop) for online A/B testing and rollback without code deploys.

**Interview note:**
*"Prompts are a first-class artifact, not an implementation detail. Versioning them separately from node logic enables prompt A/B testing and independent iteration by prompt engineers."*

---

## 16. Eval Ground Truth: Hand-Curated Golden Test Set

**Decision:** 12 fixed, hand-curated test cases covering 5 categories. Fixed across all eval runs.

**Alternatives considered:**
- Training data as ground truth
- Dynamically sampled test cases
- Real user feedback (not available)

**Why:**
Training data tells you what users historically interacted with — not whether the agent's explanation was coherent or grounded. Golden test cases let you specify exactly what a correct response looks like for each query type.

Fixed test set means every eval run is directly comparable — you can measure deltas between versions. Dynamic sampling introduces variance that makes it hard to tell if improvements are real.

**Interview note:**
*"We don't have real user feedback, so we hand-curated a golden test set covering simple recommendations, follow-ups, factual queries, ambiguous inputs, and cold start users. Fixed across runs — the only way to measure real improvement is a stable baseline."*

---

## 17. Failure Taxonomy Over Aggregate Error Rate

**Decision:** Classify failures into 5 categories: hallucinated_claim, tool_loop, premature_termination, context_loss, over_delegation.

**Alternatives considered:**
- Single "error rate" metric
- Binary pass/fail per test case
- Free-form LLM judge error description

**Why:**
A single error rate hides root causes. Knowing 60% of failures are tool_loops tells you the agent is getting stuck retrying — fix the retry logic. Knowing 40% are hallucinated_claims tells you the critic isn't catching enough — improve the critic prompt. Without taxonomy you're guessing what to fix.

**Interview note:**
*"Aggregate error rate is useless for debugging. The failure taxonomy tells you exactly what's broken. After eval v1 we found X% tool loops — that told us exactly where to look."*

---

## 18. Speech: Resample to 16kHz Mono Before Whisper

**Decision:** Resample all audio to 16kHz mono before feeding to Whisper.

**Why:**
Whisper was trained on 16kHz mono audio. Feeding 44.1kHz stereo means ~2.75x more samples than expected and two channels instead of one. Speech frequencies are all below 8kHz — anything above is wasted computation. Resampling matches inference-time input to model training distribution — same principle as normalizing item features before the two-tower.

**Interview note:**
*"Always match inference input to training distribution. Whisper was trained on 16kHz mono — feeding it 44.1kHz stereo doesn't crash it, but you're processing 3x more data than necessary and potentially degrading quality."*

---

## 19. VLM: Text Description Over Direct Embedding for Retrieval

**Decision:** PaliGemma → structured text description → hybrid retrieval (Option 1). Not VLM embedding → ANN search (Option 2).

**Alternatives considered:**
- Option 2: train projection layer, map visual embedding into item tower space

**Why:**
Option 1 plugs into existing pipeline with zero new infrastructure. Text description is auditable and debuggable. No projection layer training required. Option 2 requires poster-to-interaction training signal and an additional training pipeline.

**At scale (Option 2):** Once sufficient poster data exists, train a projection layer mapping PaliGemma visual embeddings into item tower space. Enables direct ANN retrieval from visual input — captures style cues text can miss.

**Interview note:**
*"We went with text-mediated retrieval because it's auditable and requires no additional training. The production upgrade is a learned projection from visual embedding space into item tower space — same idea as how VLMs align vision and language encoders."*

---

## 20. Quantization: int4 NF4 Over int8

**Decision:** int4 NF4 quantization via bitsandbytes for the LLM backbone.

**Alternatives considered:**
- fp16 (no quantization)
- int8
- GPTQ (post-training quantization with calibration dataset)

**Why:**
int4 reduces GPU memory by ~75% vs fp16 with <3% task completion degradation — validated in benchmark. This fits a 7B model on a single 16GB GPU, enabling local development without cloud GPU cost.

NF4 (normalized float 4) over standard int4: NF4 is information-theoretically optimal for normally distributed weights, which neural network weights approximately are. Recovers ~0.5-1% quality vs standard int4 at no cost.

GPTQ gives better quality than bitsandbytes int4 but requires a calibration dataset and is slower to load. For development iteration speed, bitsandbytes is preferred.

**What breaks at scale:**
At production scale: vLLM with continuous batching + AWQ quantization. AWQ (Activation-aware Weight Quantization) outperforms NF4 by accounting for activation magnitudes during quantization.

**Interview note:**
*"int4 NF4 gave us ~75% memory reduction with <3% quality loss on task completion. The key insight is NF4 vs standard int4 — NF4 uses a non-uniform quantization grid optimized for normally distributed weights, which is what neural network weights approximately are."*

---

## 21. KV Cache: Session-Level With Sliding Window

**Decision:** Session-level KV cache with 2048-token sliding window, time-based eviction.

**Alternatives considered:**
- No caching (recompute every turn)
- Request-level cache (clear between nodes)
- Full context cache (no window limit)
- PagedAttention (vLLM)

**Why:**
Session-level cache persists across all 4-5 LLM calls within an agent turn. Without it, each node (planner, explainer, critic, refiner) recomputes K, V for the full conversation history. At turn 5 that's ~25x more attention computation than necessary.

Sliding window bounds memory: full cache grows linearly with conversation length and eventually OOMs. 2048 tokens covers ~10 turns of conversation — sufficient for our use case.

**What breaks at scale:**
PagedAttention (vLLM): manages KV cache as paged memory, eliminating fragmentation and enabling much higher concurrency. This is the production serving solution.

**Interview note:**
*"KV cache benefit compounds with conversation length — turn 1 gets no benefit, turn 5 gets ~4x speedup because you're reusing 4 turns of cached K, V. This is critical for multi-turn agent sessions where each turn makes 4-5 LLM calls."*

---

## 22. Diffusion: MLP Denoising Network (Not Transformer)

**Decision:** MLP for the denoising network in the diffusion model.

**Why:**
Interaction vectors are permutation-invariant — item order doesn't matter for collaborative filtering. Transformer imposes sequence structure that doesn't exist in interaction data. MLP is simpler, faster, and captures the right inductive bias.

**Interview note:**
*"Architecture choice should match data structure. Interaction vectors have no ordering — an MLP is the right choice, not a Transformer."*

---

## 23. GAN: WGAN-GP Over Vanilla GAN

**Decision:** Wasserstein GAN with gradient penalty for synthetic user generation.

**Why:**
Vanilla GAN training is unstable — mode collapse causes the generator to produce limited diversity. WGAN-GP uses Wasserstein distance + gradient penalty instead of binary cross-entropy, providing stable training and diverse outputs. Critical for generating the 4 distinct profile types needed for adversarial eval.

**Interview note:**
*"Mode collapse is the main failure mode in vanilla GAN training. WGAN-GP fixes this via Wasserstein distance, which provides meaningful gradients even when discriminator is near-optimal."*

---

## 24. Reranker: LightGBM Over Neural Reranker

**Decision:** LightGBM with handcrafted features for Stage 2 ranking.

**Why:**
LightGBM: <10ms inference, interpretable feature importances, no GPU required. Neural reranker: higher accuracy, 50-200ms inference, GPU required, harder to debug. For our <500ms SLA, LightGBM keeps reranking within budget. Feature importance from LightGBM also directly tells you what signals matter — useful for debugging and for interviews.

**What breaks at scale:**
At scale: two-stage reranker — LightGBM for fast initial reranking, then a small cross-encoder (BERT-based) for the final top-10. This is the architecture used at LinkedIn.

**Interview note:**
*"LightGBM gives interpretable feature importances — I can tell you exactly which signals drive the ranking. For a 200-candidate reranking problem, it's faster and more debuggable than a neural reranker."*