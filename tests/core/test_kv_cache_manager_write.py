"""Focused tests for KVCacheManager write_kv dispatch paths."""

from __future__ import annotations

import torch

from somi_inference.core.paged_attention import KVCacheManager

BLOCK_SIZE = 4
NUM_BLOCKS = 10
NUM_KV_HEADS = 2
PRIMARY_HEAD_DIM = 3
SECONDARY_HEAD_DIM = 2
LAYER_IDX = 0
PREFILL_TOKENS = 6
FIRST_BLOCK_TOKENS = 4
PREFIX_TOKENS = 3
FORK_APPEND_TOKENS = 2
PREFIX_VALUE_OFFSET = 100
FORK_VALUE_OFFSET = 100
FORK_KEY_OFFSET = 200
SEQ_ID = 0
FORKED_SEQ_ID = 1


def _make_manager(*, head_dim: int) -> KVCacheManager:
    return KVCacheManager(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=head_dim,
        n_layers=1,
    )


def test_write_kv_prefill_writes_block_slices() -> None:
    """Prefill writes should honor the current base offset across blocks."""
    manager = _make_manager(head_dim=PRIMARY_HEAD_DIM)
    manager.register_sequence(seq_id=SEQ_ID)
    manager.allocate_slots(seq_id=SEQ_ID, new_num_tokens=PREFILL_TOKENS)

    key = torch.arange(
        PREFILL_TOKENS * NUM_KV_HEADS * PRIMARY_HEAD_DIM,
        dtype=torch.float32,
    ).view(PREFILL_TOKENS, NUM_KV_HEADS, PRIMARY_HEAD_DIM)
    value = key + 1000
    manager.write_kv(
        seq_id=SEQ_ID,
        layer_idx=LAYER_IDX,
        layer_key=key,
        layer_value=value,
    )

    first_block_id, second_block_id = manager.get_block_ids(SEQ_ID)
    cache = manager.kv_caches[LAYER_IDX]
    torch.testing.assert_close(
        cache.key_cache[first_block_id, :FIRST_BLOCK_TOKENS],
        key[:FIRST_BLOCK_TOKENS],
    )
    torch.testing.assert_close(
        cache.value_cache[first_block_id, :FIRST_BLOCK_TOKENS],
        value[:FIRST_BLOCK_TOKENS],
    )
    torch.testing.assert_close(
        cache.key_cache[second_block_id, : PREFILL_TOKENS - FIRST_BLOCK_TOKENS],
        key[FIRST_BLOCK_TOKENS:],
    )
    torch.testing.assert_close(
        cache.value_cache[second_block_id, : PREFILL_TOKENS - FIRST_BLOCK_TOKENS],
        value[FIRST_BLOCK_TOKENS:],
    )
    assert manager.get_num_tokens(SEQ_ID) == 0


def test_write_kv_prefill_preserves_copy_on_write() -> None:
    """Prefill writes should use the blocks prepared by allocate_slots()."""
    manager = _make_manager(head_dim=SECONDARY_HEAD_DIM)
    manager.register_sequence(seq_id=SEQ_ID)
    manager.allocate_slots(seq_id=SEQ_ID, new_num_tokens=PREFIX_TOKENS)

    prefix_key = torch.arange(
        PREFIX_TOKENS * NUM_KV_HEADS * SECONDARY_HEAD_DIM,
        dtype=torch.float32,
    ).view(PREFIX_TOKENS, NUM_KV_HEADS, SECONDARY_HEAD_DIM)
    prefix_value = prefix_key + PREFIX_VALUE_OFFSET
    manager.write_kv(
        seq_id=SEQ_ID,
        layer_idx=LAYER_IDX,
        layer_key=prefix_key,
        layer_value=prefix_value,
    )
    manager.advance_tokens(seq_id=SEQ_ID, num_tokens=PREFIX_TOKENS)

    manager.fork_sequence(src_seq_id=SEQ_ID, dst_seq_id=FORKED_SEQ_ID)
    original_block_id = manager.get_block_ids(SEQ_ID)[0]

    manager.allocate_slots(seq_id=FORKED_SEQ_ID, new_num_tokens=FORK_APPEND_TOKENS)
    new_first_block_id, new_second_block_id = manager.get_block_ids(FORKED_SEQ_ID)
    assert new_first_block_id != original_block_id

    fork_key = torch.arange(
        FORK_APPEND_TOKENS * NUM_KV_HEADS * SECONDARY_HEAD_DIM,
        dtype=torch.float32,
    ).view(FORK_APPEND_TOKENS, NUM_KV_HEADS, SECONDARY_HEAD_DIM) + FORK_KEY_OFFSET
    fork_value = fork_key + FORK_VALUE_OFFSET
    manager.write_kv(
        seq_id=FORKED_SEQ_ID,
        layer_idx=LAYER_IDX,
        layer_key=fork_key,
        layer_value=fork_value,
    )

    cache = manager.kv_caches[LAYER_IDX]
    torch.testing.assert_close(
        cache.key_cache[original_block_id, :PREFIX_TOKENS],
        prefix_key,
    )
    torch.testing.assert_close(
        cache.value_cache[original_block_id, :PREFIX_TOKENS],
        prefix_value,
    )
    torch.testing.assert_close(
        cache.key_cache[new_first_block_id, PREFIX_TOKENS],
        fork_key[0],
    )
    torch.testing.assert_close(
        cache.value_cache[new_first_block_id, PREFIX_TOKENS],
        fork_value[0],
    )
    torch.testing.assert_close(cache.key_cache[new_second_block_id, 0], fork_key[1])
    torch.testing.assert_close(cache.value_cache[new_second_block_id, 0], fork_value[1])
    assert manager.get_num_tokens(FORKED_SEQ_ID) == PREFIX_TOKENS
