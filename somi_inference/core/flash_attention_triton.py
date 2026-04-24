"""Triton kernel for prefill causal attention."""

# ruff: noqa: ANN001, ANN202, N803, PLR0915

from __future__ import annotations

from typing import Any, cast

import torch

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

MAX_TRITON_HEAD_DIM = 256
SMALL_HEAD_DIM = 64
MEDIUM_HEAD_DIM = 128
LOG2E = 1.44269504
SUPPORTED_TRITON_DTYPES = {torch.float16, torch.bfloat16}

if TRITON_AVAILABLE:

    @triton.jit
    def _triton_causal_attention_kernel(
        output_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        output_stride_0,
        output_stride_1,
        output_stride_2,
        output_stride_3,
        q_stride_0,
        q_stride_1,
        q_stride_2,
        q_stride_3,
        k_stride_0,
        k_stride_1,
        k_stride_2,
        k_stride_3,
        v_stride_0,
        v_stride_1,
        v_stride_2,
        v_stride_3,
        seq_len,
        num_queries_per_kv,
        scale_log2,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute one causal attention output tile for one batch/head pair."""
        tl.static_assert(BLOCK_M % BLOCK_N == 0)
        start_m = tl.program_id(axis=0)
        batch_idx = tl.program_id(axis=1)
        q_head_idx = tl.program_id(axis=2)
        kv_head_idx = q_head_idx // num_queries_per_kv

        # Split causal traversal into two stages:
        # 1. off-band full tiles strictly before the current query block
        # 2. on-band tiles inside the current query block that still need masking
        start_m_offset = start_m * BLOCK_M
        q_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, HEAD_SIZE_PADDED)

        q_mask = q_offsets < seq_len
        dim_mask = dim_offsets < HEAD_SIZE

        q_ptrs = (
            q_ptr
            + batch_idx * q_stride_0
            + q_head_idx * q_stride_1
            + q_offsets[:, None] * q_stride_2
            + dim_offsets[None, :] * q_stride_3
        )
        q = tl.load(
            q_ptrs,
            mask=q_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_SIZE_PADDED), dtype=tl.float32)

        for start_n in tl.range(0, start_m_offset, BLOCK_N):
            k_positions = start_n + kv_offsets
            k_mask = k_positions < seq_len
            score_mask = q_mask[:, None] & k_mask[None, :]

            k_ptrs = (
                k_ptr
                + batch_idx * k_stride_0
                + kv_head_idx * k_stride_1
                + k_positions[None, :] * k_stride_2
                + dim_offsets[:, None] * k_stride_3
            )
            k = tl.load(
                k_ptrs,
                mask=dim_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, k) * scale_log2
            scores = tl.where(score_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_m_i = tl.maximum(m_i, block_max)
            new_m_i = tl.where(q_mask, new_m_i, 0.0)
            alpha = tl.where(q_mask, tl.math.exp2(m_i - new_m_i), 0.0)
            p = tl.where(score_mask, tl.math.exp2(scores - new_m_i[:, None]), 0.0)

            v_ptrs = (
                v_ptr
                + batch_idx * v_stride_0
                + kv_head_idx * v_stride_1
                + k_positions[:, None] * v_stride_2
                + dim_offsets[None, :] * v_stride_3
            )
            v = tl.load(
                v_ptrs,
                mask=k_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_m_i

        block_end = tl.minimum(start_m_offset + BLOCK_M, seq_len)
        for start_n in tl.range(start_m_offset, block_end, BLOCK_N):
            k_positions = start_n + kv_offsets
            k_mask = k_positions < seq_len
            causal_mask = q_offsets[:, None] >= k_positions[None, :]
            score_mask = q_mask[:, None] & k_mask[None, :] & causal_mask

            k_ptrs = (
                k_ptr
                + batch_idx * k_stride_0
                + kv_head_idx * k_stride_1
                + k_positions[None, :] * k_stride_2
                + dim_offsets[:, None] * k_stride_3
            )
            k = tl.load(
                k_ptrs,
                mask=dim_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q, k) * scale_log2
            scores = tl.where(score_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_m_i = tl.maximum(m_i, block_max)
            new_m_i = tl.where(q_mask, new_m_i, 0.0)
            alpha = tl.where(q_mask, tl.math.exp2(m_i - new_m_i), 0.0)
            p = tl.where(score_mask, tl.math.exp2(scores - new_m_i[:, None]), 0.0)

            v_ptrs = (
                v_ptr
                + batch_idx * v_stride_0
                + kv_head_idx * v_stride_1
                + k_positions[:, None] * v_stride_2
                + dim_offsets[None, :] * v_stride_3
            )
            v = tl.load(
                v_ptrs,
                mask=k_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_m_i

        acc = acc / tl.where(q_mask, l_i, 1.0)[:, None]
        output_ptrs = (
            output_ptr
            + batch_idx * output_stride_0
            + q_head_idx * output_stride_1
            + q_offsets[:, None] * output_stride_2
            + dim_offsets[None, :] * output_stride_3
        )
        tl.store(output_ptrs, acc, mask=q_mask[:, None] & dim_mask[None, :])

else:
    _triton_causal_attention_kernel = cast("Any", None)


def _supports_bfloat16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.cuda.is_bf16_supported()


def triton_causal_attention_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Return whether the Triton prefill backend can handle these inputs."""
    if not TRITON_AVAILABLE:
        return False
    head_dim = q.shape[-1]
    return (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype
        and q.dtype in SUPPORTED_TRITON_DTYPES
        and (q.dtype != torch.bfloat16 or _supports_bfloat16(q.device))
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and k.shape[1] == v.shape[1]
        and 0 < head_dim <= MAX_TRITON_HEAD_DIM
    )


def _launch_config(head_dim: int) -> tuple[int, int, int]:
    """Return `(BLOCK_M, BLOCK_N, num_warps)` for the given head dim."""
    if head_dim <= SMALL_HEAD_DIM:
        return MEDIUM_HEAD_DIM, SMALL_HEAD_DIM, 4
    if head_dim <= MEDIUM_HEAD_DIM:
        return SMALL_HEAD_DIM, SMALL_HEAD_DIM, 4
    return 64, 32, 8


def causal_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Run prefill causal attention with the Triton backend."""
    if not triton_causal_attention_supported(q, k, v):
        msg = "Triton causal attention backend is not available for these inputs"
        raise RuntimeError(msg)

    output = torch.empty_like(q)
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        msg = "num_q_heads must be divisible by num_kv_heads for GQA"
        raise AssertionError(msg)

    block_m, block_n, num_warps = _launch_config(head_dim)
    head_dim_padded = triton.next_power_of_2(head_dim)

    kernel: Any = _triton_causal_attention_kernel
    kernel[(triton.cdiv(seq_len, block_m), batch_size, num_q_heads)](
        output,
        q,
        k,
        v,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        seq_len,
        num_q_heads // num_kv_heads,
        (head_dim**-0.5) * LOG2E,
        HEAD_SIZE=head_dim,
        HEAD_SIZE_PADDED=head_dim_padded,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return output
