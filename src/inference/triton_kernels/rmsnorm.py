"""
Fused RMSNorm Triton Kernel

Design decision: Separate LayerNorm kernels require two passes over data
(compute mean, then normalize). Fusing into a single Triton kernel reduces
memory bandwidth usage and kernel launch overhead — important for LLM
inference latency in multi-turn agent sessions.

Benchmark results: see notebooks/inference_benchmarks.ipynb
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr,          # input pointer
    W_ptr,          # weight pointer
    Y_ptr,          # output pointer
    stride,         # row stride
    N: tl.constexpr,       # hidden dim
    eps: tl.constexpr,     # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS in a single pass
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)

    # Normalize and scale
    w = tl.load(W_ptr + cols, mask=mask)
    y = (x / rms) * w

    tl.store(Y_ptr + cols, y, mask=mask)


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.hidden_size)
        y = torch.empty_like(x)

        N = self.hidden_size
        BLOCK_SIZE = triton.next_power_of_2(N)
        n_rows = x.shape[0]

        _rmsnorm_kernel[(n_rows,)](
            x, self.weight, y,
            stride=x.stride(0),
            N=N,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y.view(shape)


def benchmark_rmsnorm(hidden_size: int = 4096, n_rows: int = 2048):
    """Compare fused Triton kernel vs. PyTorch naive RMSNorm."""
    import time

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    fused = FusedRMSNorm(hidden_size).cuda().half()
    naive = torch.nn.RMSNorm(hidden_size).cuda().half()

    # Warmup
    for _ in range(10):
        fused(x)
        naive(x)
    torch.cuda.synchronize()

    # Benchmark fused
    t0 = time.perf_counter()
    for _ in range(100):
        fused(x)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 10  # ms per call

    # Benchmark naive
    t0 = time.perf_counter()
    for _ in range(100):
        naive(x)
    torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - t0) * 10

    print(f"Fused RMSNorm: {fused_ms:.3f} ms")
    print(f"Naive RMSNorm: {naive_ms:.3f} ms")
    print(f"Speedup: {naive_ms / fused_ms:.2f}x")
