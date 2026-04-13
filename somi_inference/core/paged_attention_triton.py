"""Triton kernels for paged attention decode."""

# ruff: noqa: ANN001, ANN202, N803

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
HEAD_DIM_WARP_SWITCH_THRESHOLD = 128
SUPPORTED_TRITON_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

if TRITON_AVAILABLE:

    @triton.jit
    def _triton_paged_attention_decode_kernel(
        output_ptr,
        query_ptr,
        kv_cache_ptr,
        block_tables_ptr,
        seq_lens_ptr,
        query_stride_0,
        query_stride_1,
        query_stride_2,
        kv_stride_0,
        kv_stride_1,
        kv_stride_2,
        kv_stride_3,
        kv_stride_4,
        block_table_stride_0,
        output_stride_0,
        output_stride_1,
        output_stride_2,
        scale,
        num_queries_per_kv,
        BLOCK_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        MAX_BLOCKS_PER_SEQ: tl.constexpr,
    ):
        """Compute one decode output row for one sequence and one query head."""
        # V1 uses the simplest launch mapping: one program owns one
        # `(sequence, query_head)` output row and scans that sequence's pages.
        seq_idx = tl.program_id(axis=0)
        q_head_idx = tl.program_id(axis=1)
        # GQA is handled as an integer mapping instead of duplicating KV heads.
        kv_head_idx = q_head_idx // num_queries_per_kv

        dim_offsets = tl.arange(0, HEAD_SIZE_PADDED)
        dim_mask = dim_offsets < HEAD_SIZE
        q_block_ptr = tl.make_block_ptr(
            base=query_ptr + seq_idx * query_stride_0 + q_head_idx * query_stride_1,
            shape=(HEAD_SIZE,),
            strides=(query_stride_2,),
            offsets=(0,),
            block_shape=(HEAD_SIZE_PADDED,),
            order=(0,),
        )
        # Keep the score path in fp32: q, k, online-softmax state, and value
        # accumulation all use the same higher-precision domain for stability.
        q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )

        # `seq_len` is only used for logical comparisons and masks, so int32 is
        # sufficient and lighter than carrying int64 through the whole loop.
        seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
        block_offsets = tl.arange(0, BLOCK_SIZE)

        # Online softmax state:
        # - `running_max`: current max logit
        # - `running_sum`: exp-sum after max correction
        # - `acc`: value accumulation in the same normalized frame
        acc = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
        running_max = tl.full((1,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((1,), dtype=tl.float32)

        # `MAX_BLOCKS_PER_SEQ` is fixed for this launch, so `static_range`
        # lets Triton generate a tighter loop than a Python-side gather path.
        for block_idx in tl.static_range(0, MAX_BLOCKS_PER_SEQ):
            token_positions = block_idx * BLOCK_SIZE + block_offsets
            # The last logical block is often only partially filled.
            token_mask = token_positions < seq_len
            block_id = tl.load(
                block_tables_ptr + seq_idx * block_table_stride_0 + block_idx
            ).to(tl.int64)
            # `block_id` participates in pointer arithmetic with tensor strides, so
            # we keep it in int64 to match address-offset computation.

            key_offsets = (
                block_id * kv_stride_0
                + block_offsets[:, None] * kv_stride_2
                + kv_head_idx * kv_stride_3
                + dim_offsets[None, :] * kv_stride_4
            )
            key = tl.load(
                kv_cache_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(key * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max)

            value_offsets = (
                block_id * kv_stride_0
                + kv_stride_1
                + block_offsets[:, None] * kv_stride_2
                + kv_head_idx * kv_stride_3
                + dim_offsets[None, :] * kv_stride_4
            )
            value = tl.load(
                kv_cache_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc = acc * correction + tl.sum(weights[:, None] * value, axis=0)
            running_sum = running_sum * correction + tl.sum(weights, axis=0)
            running_max = new_max

        output = acc / running_sum
        # `output_ptr` points to a tensor allocated with `torch.empty_like(q)`, so
        # tl.store implicitly converts the fp32 accumulator back to the output dtype.
        tl.store(
            output_ptr
            + seq_idx * output_stride_0
            + q_head_idx * output_stride_1
            + dim_offsets * output_stride_2,
            output,
            mask=dim_mask,
        )

else:
    _triton_paged_attention_decode_kernel = cast("Any", None)


def triton_paged_attention_supported(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    """Return whether the Triton decode backend can handle these inputs."""
    if not TRITON_AVAILABLE:
        return False
    head_dim = q.shape[-1]
    # These checks describe the scope of our V1 kernel rather than a fundamental
    # Triton limitation. Unsupported cases deliberately fall back to reference.
    return (
        q.is_cuda
        and kv_cache.is_cuda
        and block_tables.device == q.device
        and seq_lens.device == q.device
        and kv_cache.device == q.device
        and q.dtype in SUPPORTED_TRITON_DTYPES
        and kv_cache.dtype == q.dtype
        and block_tables.dtype == torch.long
        and seq_lens.dtype == torch.long
        and block_tables.stride(1) == 1
        and 0 < head_dim <= MAX_TRITON_HEAD_DIM
        and kv_cache.shape[2] > 0
        and block_tables.shape[1] > 0
    )


def paged_attention_decode_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Run decode-only paged attention with the Triton backend."""
    if not triton_paged_attention_supported(q, kv_cache, block_tables, seq_lens):
        msg = "Triton paged attention backend is not available for these inputs"
        raise RuntimeError(msg)

    output = torch.empty_like(q)
    num_seqs, num_q_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[3]
    if num_q_heads % num_kv_heads != 0:
        msg = "num_q_heads must be divisible by num_kv_heads for GQA"
        raise AssertionError(msg)
    num_queries_per_kv = num_q_heads // num_kv_heads
    block_size = kv_cache.shape[2]
    # We pad the reduction width to a power of two for simpler vectorized loads.
    head_dim_padded = triton.next_power_of_2(head_dim)
    num_warps = 4 if head_dim_padded <= HEAD_DIM_WARP_SWITCH_THRESHOLD else 8

    kernel: Any = _triton_paged_attention_decode_kernel
    kernel[(num_seqs, num_q_heads)](
        output,
        q,
        kv_cache,
        block_tables,
        seq_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        block_tables.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim**-0.5,
        num_queries_per_kv,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_dim,
        HEAD_SIZE_PADDED=head_dim_padded,
        MAX_BLOCKS_PER_SEQ=block_tables.shape[1],
        num_warps=num_warps,
    )
    return output
