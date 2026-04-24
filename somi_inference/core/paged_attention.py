"""Paged attention core data structures and decode backends."""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch

from somi_inference.core.paged_attention_triton import (
    paged_attention_decode_triton,
    triton_paged_attention_supported,
)

PagedAttentionBackend = Literal["torch_ref", "triton"]
KV_CACHE_NUM_PLANES = 2
BLOCK_TABLE_NDIM = 2
KV_CACHE_NDIM = 5
QUERY_NDIM = 3
SINGLE_TOKEN_KV_NDIM = 2
PREFILL_TOKEN_BATCH_NDIM = 3


class BlockAllocator:
    """A simple block allocator that manages free blocks and reference counts."""

    def __init__(
        self,
        num_blocks: int,
    ) -> None:
        """Initialize the block allocator."""
        self.free_blocks = list(range(num_blocks))
        self.ref_cnt = {}

    def allocate(self) -> int:
        """Allocate a block and return its ID."""
        block_id = self.free_blocks.pop()
        assert block_id not in self.ref_cnt, "Block ID already allocated"
        self.ref_cnt[block_id] = 1
        return block_id

    def free(self, block_id: int) -> None:
        """Free a block by decrementing its reference count."""
        self.ref_cnt[block_id] -= 1
        if self.ref_cnt[block_id] == 0:
            self.free_blocks.append(block_id)
            del self.ref_cnt[block_id]

    def increase_ref(self, block_id: int) -> None:
        """Increase the reference count of a block."""
        self.ref_cnt[block_id] += 1

    def need_cow(self, block_id: int) -> bool:
        """Check if a block needs to be copied on write."""
        return self.ref_cnt[block_id] > 1

    def num_free_blocks(self) -> int:
        """Return the number of free blocks."""
        return len(self.free_blocks)


class KVCache:
    """Hold the actual KV tensors in a fused vLLM-style layout."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the KV cache with pre-allocated fused K/V storage."""
        # Keep K/V in one fused tensor so the decode backend can pass a single
        # cache pointer around, which is closer to vLLM's runtime contract.
        self.kv_cache = torch.zeros(
            (num_blocks, 2, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @property
    def key_cache(self) -> torch.Tensor:
        """Return a view of the key cache."""
        return self.kv_cache[:, 0]

    @property
    def value_cache(self) -> torch.Tensor:
        """Return a view of the value cache."""
        return self.kv_cache[:, 1]

    def write(
        self, block_id: int, slot: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Write key and value tensors of a single token to the cache."""
        assert key.shape == value.shape == (self.num_kv_heads, self.head_dim), (
            f"Key and value tensors must have shape "
            f"({self.num_kv_heads}, {self.head_dim})"
        )
        self.kv_cache[block_id, 0, slot] = key
        self.kv_cache[block_id, 1, slot] = value

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy the entire fused KV block from source to dest."""
        self.kv_cache[dst_block_id] = self.kv_cache[src_block_id]


class KVCacheManager:
    """Coordinate BlockAllocator and KVCache, maintain block table per sequence."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        n_layers: int = 1,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the KV cache manager."""
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
        )
        self.n_layers = n_layers
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self.kv_caches = [
            KVCache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.n_layers)
        ]
        self.block_size = block_size
        self.seq_to_block: dict[int, list[int]] = {}
        self.seq_to_num_tokens: dict[int, int] = {}

    def register_sequence(self, seq_id: int) -> None:
        """Register a new sequence ID for tracking."""
        if seq_id in self.seq_to_block:
            msg = f"Sequence ID {seq_id} is already registered."
            raise ValueError(msg)
        self.seq_to_block[seq_id] = []
        self.seq_to_num_tokens[seq_id] = 0

    def get_block_ids(self, seq_id: int) -> list[int]:
        """Get the list of block IDs associated with the given sequence ID."""
        return self.seq_to_block[seq_id]

    def get_num_tokens(self, seq_id: int) -> int:
        """Get the number of tokens currently stored for the given sequence ID."""
        return self.seq_to_num_tokens[seq_id]

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> None:
        """Fork a sequence by copying the block references and token count."""
        for block_id in self.seq_to_block[src_seq_id]:
            self.allocator.increase_ref(block_id)
        self.seq_to_block[dst_seq_id] = self.seq_to_block[src_seq_id].copy()
        self.seq_to_num_tokens[dst_seq_id] = self.seq_to_num_tokens[src_seq_id]

    def build_block_tables(
        self, seq_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble block_tables and seq_lens tensors for paged attention decode."""
        block_ids_list = [self.get_block_ids(sid) for sid in seq_ids]
        max_blocks = max(len(b) for b in block_ids_list)
        block_tables = torch.zeros(
            len(seq_ids), max_blocks, dtype=torch.long, device=self.device
        )
        # `seq_lens` is the source of truth at decode time, so zero-padding the
        # tail of each row is fine as long as callers also pass the true length.
        for i, bids in enumerate(block_ids_list):
            block_tables[i, : len(bids)] = torch.tensor(
                bids, dtype=torch.long, device=self.device
            )
        seq_lens = torch.tensor(
            [self.get_num_tokens(sid) for sid in seq_ids],
            dtype=torch.long,
            device=self.device,
        )
        return block_tables, seq_lens

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks associated with the given sequence ID."""
        for block_id in self.seq_to_block[seq_id]:
            self.allocator.free(block_id)
        del self.seq_to_block[seq_id]
        del self.seq_to_num_tokens[seq_id]

    def allocate_slots(self, seq_id: int, new_num_tokens: int) -> None:
        """Allocate blocks and slots for upcoming tokens, handling COW if needed."""
        current = self.seq_to_num_tokens[seq_id]
        for i in range(new_num_tokens):
            pos = current + i
            slot = pos % self.block_size
            if slot == 0:
                # Starting a new logical page always needs a fresh physical block.
                block_id = self.allocator.allocate()
                self.seq_to_block[seq_id].append(block_id)
            else:
                block_id = self.seq_to_block[seq_id][pos // self.block_size]
                if self.allocator.need_cow(block_id):
                    # Forked sequences can still share a partially-filled block.
                    # The first in-place write must break sharing to preserve COW.
                    new_block_id = self.allocator.allocate()
                    for cache in self.kv_caches:
                        cache.copy_block(block_id, new_block_id)
                    self.allocator.free(block_id)
                    self.seq_to_block[seq_id][pos // self.block_size] = new_block_id

    def write_kv(
        self,
        seq_id: int,
        layer_idx: int,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
    ) -> None:
        """Write key/value tensors to paged cache for a single layer."""
        cache = self.kv_caches[layer_idx]
        if layer_key.shape != layer_value.shape:
            msg = "layer_key and layer_value must have the same shape"
            raise ValueError(msg)
        if layer_key.dim() == SINGLE_TOKEN_KV_NDIM:
            self._write_kv_single(seq_id, cache, layer_key, layer_value)
            return
        if layer_key.dim() != PREFILL_TOKEN_BATCH_NDIM:
            msg = (
                "layer_key and layer_value must have shape "
                "(num_kv_heads, head_dim) or (num_tokens, num_kv_heads, head_dim)"
            )
            raise ValueError(msg)
        self._write_kv_prefill(seq_id, cache, layer_key, layer_value)

    def _write_kv_single(
        self,
        seq_id: int,
        cache: KVCache,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
    ) -> None:
        """Write one token of KV data for decode."""
        pos = self.seq_to_num_tokens[seq_id]
        block_id = self.seq_to_block[seq_id][pos // self.block_size]
        slot = pos % self.block_size
        cache.write(block_id, slot, layer_key, layer_value)

    def _write_kv_prefill(
        self,
        seq_id: int,
        cache: KVCache,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
    ) -> None:
        """Write a prefill token batch with block-wise slice assignments."""
        base = self.seq_to_num_tokens[seq_id]
        token_offset = 0
        pos = base
        num_tokens = layer_key.shape[0]
        block_ids = self.seq_to_block[seq_id]

        while token_offset < num_tokens:
            block_index = pos // self.block_size
            block_id = block_ids[block_index]
            start_slot = pos % self.block_size
            tokens_in_block = min(
                self.block_size - start_slot,
                num_tokens - token_offset,
            )
            end_slot = start_slot + tokens_in_block
            token_end = token_offset + tokens_in_block

            cache.key_cache[block_id, start_slot:end_slot] = layer_key[
                token_offset:token_end
            ]
            cache.value_cache[block_id, start_slot:end_slot] = layer_value[
                token_offset:token_end
            ]

            token_offset = token_end
            pos += tokens_in_block

    def advance_tokens(self, seq_id: int, num_tokens: int) -> None:
        """Advance token count after writing KV for multiple layers."""
        self.seq_to_num_tokens[seq_id] += num_tokens


def pack_kv_cache(key_cache: torch.Tensor, value_cache: torch.Tensor) -> torch.Tensor:
    """Pack separate key/value tensors into fused ``kv_cache`` layout."""
    # This helper is mainly for tests and microbenchmarks. The real runtime
    # keeps KV fused from the moment it is allocated.
    if key_cache.shape != value_cache.shape:
        msg = "key_cache and value_cache must have the same shape"
        raise ValueError(msg)
    return torch.stack((key_cache, value_cache), dim=1)


def _num_queries_per_kv(num_q_heads: int, num_kv_heads: int) -> int:
    if num_q_heads % num_kv_heads != 0:
        msg = "num_q_heads must be divisible by num_kv_heads for GQA"
        raise AssertionError(msg)
    return num_q_heads // num_kv_heads


def _validate_paged_attention_inputs(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    if q.dim() != QUERY_NDIM:
        msg = "q must have shape (num_seqs, num_q_heads, head_dim)"
        raise ValueError(msg)
    if kv_cache.dim() != KV_CACHE_NDIM or kv_cache.shape[1] != KV_CACHE_NUM_PLANES:
        msg = (
            "kv_cache must have shape "
            "(num_blocks, 2, block_size, num_kv_heads, head_dim)"
        )
        raise ValueError(msg)
    if block_tables.dim() != BLOCK_TABLE_NDIM:
        msg = "block_tables must have shape (num_seqs, max_blocks_per_seq)"
        raise ValueError(msg)
    if seq_lens.dim() != 1:
        msg = "seq_lens must have shape (num_seqs,)"
        raise ValueError(msg)
    if q.shape[0] != block_tables.shape[0] or q.shape[0] != seq_lens.shape[0]:
        msg = "q, block_tables, and seq_lens must agree on num_seqs"
        raise ValueError(msg)
    if q.shape[-1] != kv_cache.shape[-1]:
        msg = "q head_dim must match kv_cache head_dim"
        raise ValueError(msg)
    if kv_cache.device != q.device or block_tables.device != q.device:
        msg = "q, kv_cache, and block_tables must be on the same device"
        raise ValueError(msg)
    if seq_lens.device != q.device:
        msg = "seq_lens must be on the same device as q"
        raise ValueError(msg)
    _num_queries_per_kv(q.shape[1], kv_cache.shape[3])


def paged_attention_decode_torch_ref(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute reference paged decode attention over fused KV cache."""
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    num_q_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    softmax_dtype = (
        torch.float32
        if q.dtype in {torch.float16, torch.bfloat16}
        else q.dtype
    )
    # Keep GQA as an index mapping instead of expanding KV heads up front.
    q_to_kv_head = torch.arange(num_q_heads, device=q.device) // _num_queries_per_kv(
        num_q_heads, num_kv_heads
    )
    scale_factor = 1 / sqrt(q.shape[-1])
    seq_score_max = torch.full(
        (q.shape[0], q.shape[1]),
        -torch.inf,
        device=q.device,
        dtype=softmax_dtype,
    )
    output = torch.zeros_like(q)
    running_sum = torch.zeros_like(seq_score_max)
    max_blocks_per_seq = block_tables.shape[1]
    block_size = key_cache.shape[1]
    block_offsets = torch.arange(block_size, device=q.device)

    for block_n in range(max_blocks_per_seq):
        token_position = block_n * block_size + block_offsets
        invalid_position_msk = token_position.unsqueeze(0) >= seq_lens.unsqueeze(1)
        block_ids = block_tables[:, block_n]
        # The reference path still gathers dense per-block tensors. This keeps
        # the implementation easy to read, and gives us a correctness oracle
        # for the Triton kernel, but it is exactly the extra materialization
        # that the custom kernel is meant to avoid on the hot path.
        key_block = key_cache[block_ids].index_select(dim=2, index=q_to_kv_head)
        value_block = value_cache[block_ids].index_select(dim=2, index=q_to_kv_head)
        scores = (
            torch.einsum("s h d, s b h d -> s h b", q, key_block) * scale_factor
        ).to(softmax_dtype)
        scores = scores.masked_fill(invalid_position_msk.unsqueeze(1), -torch.inf)

        block_max = scores.max(dim=-1)
        running_max = torch.maximum(seq_score_max, block_max.values)
        correction = torch.exp(seq_score_max - running_max)
        output *= correction.unsqueeze(-1).to(output.dtype)
        running_sum *= correction
        weights = torch.exp(scores - running_max.unsqueeze(-1))
        output += torch.einsum(
            "s h b, s b h d -> s h d", weights.to(value_block.dtype), value_block
        )
        running_sum += weights.sum(dim=-1)
        seq_score_max = running_max

    output /= running_sum.unsqueeze(-1).to(output.dtype)
    return output


def paged_attention_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    backend: PagedAttentionBackend = "torch_ref",
) -> torch.Tensor:
    """Compute decode attention over paged KV cache."""
    # Validate once at the public entry point; the backend helpers assume the
    # basic shape/device contract already holds.
    _validate_paged_attention_inputs(q, kv_cache, block_tables, seq_lens)
    if backend not in {"torch_ref", "triton"}:
        msg = f"Unsupported paged attention backend: {backend}"
        raise ValueError(msg)
    if backend == "torch_ref":
        return paged_attention_decode_torch_ref(q, kv_cache, block_tables, seq_lens)
    if not triton_paged_attention_supported(q, kv_cache, block_tables, seq_lens):
        msg = "Triton paged attention backend is not available for these inputs"
        raise RuntimeError(msg)
    return paged_attention_decode_triton(q, kv_cache, block_tables, seq_lens)
