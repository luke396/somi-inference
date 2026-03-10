"""Paged attention demo."""

from math import sqrt

import torch


class BlockAllocator:
    """A simple block allocator that manages free blocks and reference counts.

    Just ID management, not holding any data.
    """

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
    """Hold the actual KV tensors and provide read/write interface."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize the KV cache with pre-allocated space for keys and values."""
        self.key_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim))
        self.value_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim))

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def write(
        self, block_id: int, slot: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Write key and value tensors of single token to the cache.

        Writes to the specified block and slot.
        The layout of key and value should be (num_heads, head_dim) for a single token.
        """
        assert key.shape == (value.shape) == (self.num_kv_heads, self.head_dim), (
            f"Key and value tensors must have shape "
            f"({self.num_kv_heads}, {self.head_dim})"
        )
        self.key_cache[block_id, slot] = key
        self.value_cache[block_id, slot] = value

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy the entire block of keys and values from source to dest."""
        self.key_cache[dst_block_id] = self.key_cache[src_block_id]
        self.value_cache[dst_block_id] = self.value_cache[src_block_id]


class KVCacheManager:
    """Coordinate BlockAllocator and KVCache, maintain block table per sequence."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        n_layers: int = 1,
    ) -> None:
        """Initialize the KV cache manager."""
        self.allocator = BlockAllocator(
            num_blocks=num_blocks,
        )
        self.n_layers = n_layers
        self.kv_caches = [
            KVCache(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            for _ in range(self.n_layers)
        ]  # every layer has its own KVCache, block IDs are shared across layers
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
        """Fork a sequence by copying the block references and token count.

        Increase the reference count for each block to ensure proper memory management.
        """
        for block_id in self.seq_to_block[src_seq_id]:
            self.allocator.increase_ref(block_id)
        self.seq_to_block[dst_seq_id] = self.seq_to_block[src_seq_id].copy()
        self.seq_to_num_tokens[dst_seq_id] = self.seq_to_num_tokens[src_seq_id]

    def build_block_tables(
        self, seq_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble block_tables and seq_lens tensors for paged_attention_decode.

        This returns block_tables and seq_lens for given sequence IDs,
        which can be directly used in paged_attention_decode.
        """
        block_ids_list = [self.get_block_ids(sid) for sid in seq_ids]
        max_blocks = max(len(b) for b in block_ids_list)
        block_tables = torch.zeros(len(seq_ids), max_blocks, dtype=torch.long)
        for i, bids in enumerate(block_ids_list):
            block_tables[i, : len(bids)] = torch.tensor(bids, dtype=torch.long)
        seq_lens = torch.tensor(
            [self.get_num_tokens(sid) for sid in seq_ids], dtype=torch.long
        )
        return block_tables, seq_lens

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks associated with the given sequence ID."""
        for block_id in self.seq_to_block[seq_id]:
            self.allocator.free(block_id)
        del self.seq_to_block[seq_id]
        del self.seq_to_num_tokens[seq_id]

    def allocate_slots(self, seq_id: int, new_num_tokens: int) -> None:
        """Allocate blocks and slots for upcoming tokens, handling COW if needed.

        Not writing kv cahce, not advancing seq_to_num_tokens;
        just ensure the necessary blocks are allocated and COW-ed if needed.
        """
        current = self.seq_to_num_tokens[seq_id]
        slot = self.seq_to_num_tokens[seq_id] % self.block_size
        for i in range(new_num_tokens):
            pos = current + i
            slot = pos % self.block_size
            if slot == 0:
                block_id = self.allocator.allocate()
                self.seq_to_block[seq_id].append(block_id)
            else:  # slot > 0, need to check COW
                block_id = self.seq_to_block[seq_id][pos // self.block_size]
                if self.allocator.need_cow(block_id):
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
        """Write key/value tensors to paged cache for a single layer.

        The layout of k v should be (num_tokens, num_heads, head_dim) or
        (num_heads, head_dim) for a single token.

        Allocate blocks must have been done by allocate_slots()
        before calling this function.

        Does NOT advance seq_to_num_tokens; call advance_tokens() after all
        layers are written.
        """
        # (num_heads, head_dim) -> (num_tokens, num_heads, head_dim)
        dim_single_token = 2
        if layer_key.dim() == dim_single_token:
            layer_key = layer_key.unsqueeze(0)
            layer_value = layer_value.unsqueeze(0)
        cache = self.kv_caches[layer_idx]
        base = self.seq_to_num_tokens[seq_id]
        for t, (token_key, token_value) in enumerate(
            zip(layer_key, layer_value, strict=True)
        ):
            # k v : (num_heads, head_dim)
            pos = base + t
            block_id = self.seq_to_block[seq_id][pos // self.block_size]
            slot = pos % self.block_size
            cache.write(block_id, slot, token_key, token_value)

    def advance_tokens(self, seq_id: int, num_tokens: int) -> None:
        """Advance token count after writing KV for multiple layers."""
        self.seq_to_num_tokens[seq_id] += num_tokens


def paged_attention_decode(
    q: torch.Tensor,  # (num_seqs, num_q_heads, head_dim)
    key_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    value_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    block_tables: torch.Tensor,  # (num_seqs, max_blocks_per_seq)
    seq_lens: torch.Tensor,  # (num_seqs,)
) -> torch.Tensor:
    """Compute paged attention decode with online softmax over KV cache blocks."""
    num_q_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    # GQA:repeat KV heads to match Q heads
    # Simply copy q_heads to match kv_heads, maybe some memory waste
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, (
            "num_q_heads must be divisible by num_kv_heads for GQA"
        )
        repeat_factor = num_q_heads // num_kv_heads
        # from (num_blocks, block_size, num_kv_heads, head_dim)
        # to (..., num_q_heads, ...)
        key_cache = key_cache.repeat_interleave(repeat_factor, dim=2)
        value_cache = value_cache.repeat_interleave(repeat_factor, dim=2)

    scale_factor = 1 / sqrt(q.shape[-1])
    seq_score_max = torch.full(
        (q.shape[0], q.shape[1]), -torch.inf, device=q.device
    )  # (num_seqs, num_heads)
    output = torch.zeros_like(q)  # (num_seqs, num_heads, head_dim)
    running_sum = torch.zeros_like(seq_score_max)  # (num_seqs, num_heads)
    max_blocks_per_seq = block_tables.shape[1]
    block_size = key_cache.shape[1]

    block_offsets = torch.arange(block_size, device=q.device)  # (block_size,)

    for block_n in range(max_blocks_per_seq):
        token_position = block_n * block_size + block_offsets  # (block_size,)
        invalid_position_msk = token_position.unsqueeze(0) >= seq_lens.unsqueeze(
            1
        )  # (num_seqs, block_size)
        block_ids = block_tables[:, block_n]  # (num_seqs,)
        key_block = key_cache[block_ids]  # (num_seqs, block_size, num_heads, head_dim)
        value_block = value_cache[
            block_ids
        ]  # (num_seqs, block_size, num_heads, head_dim)
        scores = (
            torch.einsum("s h d, s b h d -> s h b", q, key_block) * scale_factor
        )  # (num_seqs, num_heads, block_size)
        scores = scores.masked_fill(invalid_position_msk.unsqueeze(1), -torch.inf)

        block_max = scores.max(dim=-1)  # (num_seqs, num_heads)
        running_max = torch.maximum(
            seq_score_max, block_max.values
        )  # (num_seqs, num_heads)
        correction = torch.exp(seq_score_max - running_max)  # (num_seqs, num_heads)
        output *= correction.unsqueeze(-1)  # (num_seqs, num_heads, head_dim)
        running_sum *= correction  # (num_seqs, num_heads)
        weights = torch.exp(
            scores - running_max.unsqueeze(-1)
        )  # (num_seqs, num_heads, block_size)
        output += torch.einsum(
            "s h b, s b h d -> s h d", weights, value_block
        )  # (num_seqs, num_heads, head_dim)
        running_sum += weights.sum(dim=-1)  # (num_seqs, num_heads)
        seq_score_max = running_max

    output /= running_sum.unsqueeze(-1)  # (num_seqs, num_heads, head_dim)

    return output
