"""Triton backends for the Qwen MLP projections."""

# ruff: noqa: ANN001, ANN202, N803

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

TRITON_AVAILABLE = False
try:
    import triton as _triton
    import triton.language as _tl
except ImportError:
    triton = cast("Any", None)
    tl = cast("Any", None)
else:
    TRITON_AVAILABLE = True
    triton = cast("Any", _triton)
    tl = cast("Any", _tl)

SUPPORTED_TRITON_DTYPES = {torch.float16}
MIN_LINEAR_NDIM = 2

if TRITON_AVAILABLE:
    _MATMUL_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=8,
        ),
    ]

    @triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _triton_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Compute one output tile of a row-major `A @ B` matmul."""
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(tl.cdiv(K, BLOCK_K)):
            k_offsets = k_start * BLOCK_K + offs_k
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

else:
    _triton_matmul_kernel = cast("Any", None)


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten leading dims and return the original prefix shape."""
    if x.dim() < MIN_LINEAR_NDIM:
        msg = "x must have at least 2 dimensions"
        raise ValueError(msg)
    prefix_shape = tuple(x.shape[:-1])
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    return x_2d, prefix_shape


def _packed_weight_cache_key(
    weight: torch.Tensor,
) -> tuple[int, torch.device, torch.dtype]:
    return weight.data_ptr(), weight.device, weight.dtype


def get_packed_linear_weight(linear: nn.Linear) -> torch.Tensor:
    """Return and cache a contiguous transposed view of a linear weight."""
    cache_key = _packed_weight_cache_key(linear.weight)
    packed_weight = getattr(linear, "_triton_packed_weight", None)
    cached_key = getattr(linear, "_triton_packed_weight_key", None)
    if packed_weight is None or cached_key != cache_key:
        packed_weight = linear.weight.t().contiguous()
        linear.__dict__["_triton_packed_weight"] = packed_weight
        linear.__dict__["_triton_packed_weight_key"] = cache_key
    return cast("torch.Tensor", packed_weight)


def triton_linear_supported(x: torch.Tensor, packed_weight: torch.Tensor) -> bool:
    """Return whether the Triton linear backend can handle these inputs."""
    x_2d = x.reshape(-1, x.shape[-1])
    return (
        TRITON_AVAILABLE
        and not torch.is_grad_enabled()
        and x.is_cuda
        and packed_weight.is_cuda
        and x.device == packed_weight.device
        and x.dtype == packed_weight.dtype
        and x.dtype in SUPPORTED_TRITON_DTYPES
        and x_2d.shape[1] == packed_weight.shape[0]
        and x_2d.stride(-1) == 1
        and packed_weight.stride(-1) == 1
        and x_2d.is_contiguous()
        and packed_weight.is_contiguous()
    )

def _matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run row-major `a @ b` with a Triton matmul kernel."""
    if not triton_linear_supported(a, b):
        msg = "Triton linear backend is not available for these inputs"
        raise RuntimeError(msg)
    if a.dim() != MIN_LINEAR_NDIM or b.dim() != MIN_LINEAR_NDIM:
        msg = "Triton matmul expects 2D tensors"
        raise ValueError(msg)
    if a.shape[1] != b.shape[0]:
        msg = "Matmul inner dimensions must agree"
        raise ValueError(msg)

    m, k = a.shape
    _, n = b.shape
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )

    kernel: Any = _triton_matmul_kernel
    kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output


def gate_up_proj_triton(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    """Apply the merged gate/up projection with a Triton matmul kernel."""
    x_2d, prefix_shape = _flatten_to_2d(x)
    gate_up = _matmul_triton(x_2d, packed_weight)
    return gate_up.view(*prefix_shape, packed_weight.shape[1])


def down_proj_triton(x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
    """Apply the down projection with a Triton matmul kernel."""
    x_2d, prefix_shape = _flatten_to_2d(x)
    down = _matmul_triton(x_2d, packed_weight)
    return down.view(*prefix_shape, packed_weight.shape[1])
