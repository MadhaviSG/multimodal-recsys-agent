"""
LLM Quantization
=================
int4 quantization via bitsandbytes + benchmark harness.

ML System Design decisions documented inline.

Design decision: int4 (NF4) over int8.
int8: ~50% memory reduction, minimal quality loss.
int4: ~75% memory reduction, small quality loss (<3% task completion).
For our agent backbone (GPT-4o-mini via API, or local Mistral/Phi),
int4 fits a 7B model on a single 16GB GPU — enabling local development.

Design decision: quantize linear layers only, not embeddings or norms.
Embedding tables and LayerNorm weights are small — quantizing them
saves little memory but risks measurable quality degradation.
Linear layers hold >95% of parameters — that's where the savings are.

Usage:
    model = load_quantized_model("mistralai/Mistral-7B-v0.1")
    # Drop-in replacement for full precision model
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import json


@dataclass
class QuantizationBenchmark:
    precision: str              # fp16, int8, int4
    gpu_memory_gb: float        # peak GPU memory usage
    tokens_per_second: float    # generation throughput
    latency_ms_per_token: float # per-token latency
    task_completion_score: float # from eval harness (0-1)
    perplexity: float           # language model quality proxy


def load_quantized_model(
    model_id: str,
    quantization: str = "int4",
    device: str = "cuda",
):
    """
    Load an LLM with quantization via bitsandbytes.

    Design decision: NF4 (normalized float 4) over standard int4.
    NF4 is information-theoretically optimal for normally distributed
    weights — which neural network weights approximately are.
    Standard int4 uses uniform quantization grid, wastes bits on
    underrepresented values. NF4 recovers ~0.5-1% quality vs int4.

    Design decision: double quantization enabled.
    Quantizes the quantization constants themselves — saves ~0.4 bits
    per parameter with negligible quality impact. Free lunch.

    Args:
        model_id: HuggingFace model ID or local path
        quantization: "fp16", "int8", or "int4"
        device: "cuda" or "cpu"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {model_id} with {quantization} quantization...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 over standard int4
            bnb_4bit_compute_dtype=torch.float16, # compute in fp16, store in int4
            bnb_4bit_use_double_quant=True,       # quantize quantization constants
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

    elif quantization == "int8":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
        )

    else:  # fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model.eval()
    print(f"  Loaded. Memory: {_get_gpu_memory():.2f} GB")
    return model, tokenizer


def _get_gpu_memory() -> float:
    """Peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def benchmark_quantization(
    model_id: str = "microsoft/phi-2",
    prompt: str = "Recommend a psychological thriller movie and explain why:",
    n_tokens: int = 100,
    n_runs: int = 5,
) -> list[QuantizationBenchmark]:
    """
    Benchmark fp16 vs int8 vs int4 on latency, memory, and throughput.

    Design decision: use a small model (Phi-2, 2.7B) for benchmarking.
    Phi-2 fits on a single GPU at all precision levels — enables
    apples-to-apples comparison without OOM errors.
    Results extrapolate to larger models with similar relative gains.

    Results shape the inference serving decision:
        int4: ~70% memory reduction, <3% quality loss → deploy this
        int8: ~50% memory reduction, <1% quality loss → if quality critical
        fp16: baseline, best quality, most memory
    """
    results = []

    for precision in ["fp16", "int8", "int4"]:
        torch.cuda.reset_peak_memory_stats()

        try:
            model, tokenizer = load_quantized_model(model_id, precision)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            # Warmup
            with torch.no_grad():
                for _ in range(2):
                    model.generate(**inputs, max_new_tokens=20, do_sample=False)

            # Benchmark
            latencies = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=n_tokens,
                        do_sample=False,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)

            avg_latency = sum(latencies) / len(latencies)
            tps = n_tokens / avg_latency

            results.append(QuantizationBenchmark(
                precision=precision,
                gpu_memory_gb=_get_gpu_memory(),
                tokens_per_second=tps,
                latency_ms_per_token=(avg_latency / n_tokens) * 1000,
                task_completion_score=0.0,  # filled by eval harness
                perplexity=0.0,             # filled separately
            ))

            print(f"  {precision}: {_get_gpu_memory():.2f}GB | "
                  f"{tps:.1f} tok/s | {avg_latency/n_tokens*1000:.1f}ms/tok")

            # Free memory between runs
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  {precision} failed: {e}")

    return results


def save_benchmark_results(
    results: list[QuantizationBenchmark],
    out_path: str = "checkpoints/quantization_benchmark.json",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "precision": r.precision,
            "gpu_memory_gb": r.gpu_memory_gb,
            "tokens_per_second": r.tokens_per_second,
            "latency_ms_per_token": r.latency_ms_per_token,
            "task_completion_score": r.task_completion_score,
            "perplexity": r.perplexity,
        }
        for r in results
    ]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Benchmark results saved to {out_path}")


if __name__ == "__main__":
    results = benchmark_quantization()
    save_benchmark_results(results)