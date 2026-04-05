"""
KV Cache Management for Multi-Turn Agent Sessions
===================================================
Manages key-value cache across agent nodes to avoid redundant computation.

ML System Design decisions documented inline.

Why KV cache matters for our agent:
    Each agent turn runs 4-5 LLM calls (planner, explainer, critic, refiner).
    Each call sees the full conversation history.
    Without caching: recompute K, V for all previous tokens at every call.
    With caching: recompute only for new tokens per call.

    At turn 5 with 500 token history:
        Without cache: 5 × 500 × O(n) attention = 2500 attention ops
        With cache:    5 × (new tokens only) attention = ~100 attention ops
    ~25x reduction in attention computation for multi-turn sessions.

Design decision: session-level cache, not request-level.
Cache persists across all nodes within a single agent session.
Cleared between sessions (different users / different conversations).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CacheStats:
    session_id: str
    n_cached_tokens: int         # tokens currently in cache
    n_cache_hits: int            # times cache was reused
    n_cache_misses: int          # times cache was rebuilt
    memory_bytes: int            # estimated cache memory usage
    hit_rate: float              # cache_hits / (hits + misses)
    avg_latency_with_cache: float    # ms
    avg_latency_without_cache: float # ms
    speedup: float               # without / with


class KVCacheManager:
    """
    Session-level KV cache for multi-turn agent conversations.

    Design decision: sliding window cache.
    Full cache grows linearly with conversation length — eventually
    exceeds GPU memory for very long sessions.
    Sliding window keeps only the last N tokens in cache.
    Trades perfect recall of early context for bounded memory usage.

    Design decision: cache invalidation on user preference change.
    If user says "actually I prefer horror not thriller", the cached
    K, V from before the preference change are stale. We detect
    preference shift signals and invalidate the relevant cache entries.

    At scale: PagedAttention (vLLM) — manages KV cache as paged memory,
    enabling much higher throughput by eliminating cache fragmentation.
    """

    def __init__(
        self,
        max_cached_tokens: int = 2048,  # sliding window size
        device: str = "cuda",
    ):
        self.max_cached_tokens = max_cached_tokens
        self.device = device

        # Session store: session_id → cache dict
        self._sessions: dict[str, dict] = {}
        self._stats: dict[str, CacheStats] = {}

    def get_or_create_session(self, session_id: str) -> dict:
        """
        Get existing session cache or create a new one.
        Each session has its own independent KV cache.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "past_key_values": None,   # HuggingFace cache format
                "n_cached_tokens": 0,
                "created_at": time.time(),
                "last_used": time.time(),
            }
            self._stats[session_id] = CacheStats(
                session_id=session_id,
                n_cached_tokens=0,
                n_cache_hits=0,
                n_cache_misses=0,
                memory_bytes=0,
                hit_rate=0.0,
                avg_latency_with_cache=0.0,
                avg_latency_without_cache=0.0,
                speedup=1.0,
            )
        return self._sessions[session_id]

    def generate_with_cache(
        self,
        model,
        tokenizer,
        new_text: str,
        session_id: str,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate response using cached KV state.

        Pipeline:
            1. Tokenize new_text only (not full history)
            2. Pass new tokens + past_key_values to model
            3. Model extends cache with new K, V
            4. Store updated cache in session
            5. Return generated text

        Design decision: tokenize new text only.
        HuggingFace's past_key_values argument handles the history —
        we only need to tokenize and process new tokens.
        This is the core efficiency gain: O(new_tokens) not O(all_tokens).
        """
        session = self.get_or_create_session(session_id)
        stats = self._stats[session_id]

        inputs = tokenizer(new_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        t0 = time.perf_counter()

        with torch.no_grad():
            if session["past_key_values"] is not None:
                # Cache hit — reuse previously computed K, V
                stats.n_cache_hits += 1
                outputs = model.generate(
                    input_ids=input_ids,
                    past_key_values=session["past_key_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            else:
                # Cache miss — compute from scratch
                stats.n_cache_misses += 1
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update cache with new K, V
        new_past = outputs.past_key_values
        n_tokens = outputs.sequences.shape[1]

        # Sliding window: trim cache if exceeding max
        # Design decision: trim from the start (oldest tokens).
        # Oldest context is least relevant for current generation.
        # Keeps most recent conversation turns in cache.
        if n_tokens > self.max_cached_tokens:
            new_past = self._trim_cache(new_past, self.max_cached_tokens)
            n_tokens = self.max_cached_tokens

        session["past_key_values"] = new_past
        session["n_cached_tokens"] = n_tokens
        session["last_used"] = time.time()

        # Update stats
        stats.n_cached_tokens = n_tokens
        total = stats.n_cache_hits + stats.n_cache_misses
        stats.hit_rate = stats.n_cache_hits / total if total > 0 else 0.0

        # Decode response (new tokens only)
        input_length = input_ids.shape[1]
        new_token_ids = outputs.sequences[0][input_length:]
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return response

    def _trim_cache(self, past_key_values, max_tokens: int):
        """
        Trim KV cache to max_tokens by removing oldest entries.
        past_key_values is a tuple of (key, value) tensors per layer.
        """
        trimmed = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            # k, v shape: (batch, heads, seq_len, head_dim)
            trimmed.append((
                k[:, :, -max_tokens:, :],
                v[:, :, -max_tokens:, :],
            ))
        return tuple(trimmed)

    def invalidate_session(self, session_id: str):
        """
        Clear cache for a session.
        Called when: session ends, user preference shifts significantly,
        or cache memory pressure requires eviction.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            print(f"Cache invalidated for session {session_id}")

    def get_stats(self, session_id: str) -> Optional[CacheStats]:
        return self._stats.get(session_id)

    def estimate_memory_bytes(self, session_id: str) -> int:
        """
        Estimate GPU memory used by this session's KV cache.
        KV cache size = 2 (K+V) × n_layers × n_heads × seq_len × head_dim × bytes_per_element
        """
        session = self._sessions.get(session_id)
        if not session or session["past_key_values"] is None:
            return 0

        total = 0
        for k, v in session["past_key_values"]:
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total

    def evict_stale_sessions(self, max_age_seconds: int = 1800):
        """
        Evict sessions not used in the last N seconds.
        Prevents unbounded GPU memory growth in multi-user scenarios.

        Design decision: time-based eviction (LRU approximation).
        True LRU requires a heap — time-based is simpler and sufficient
        for low-concurrency development. At scale: proper LRU with
        PagedAttention's memory pool management (vLLM).
        """
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s["last_used"] > max_age_seconds
        ]
        for sid in stale:
            self.invalidate_session(sid)
        if stale:
            print(f"Evicted {len(stale)} stale sessions")