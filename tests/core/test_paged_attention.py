"""Tests for paged attention components."""
import pytest
import torch
from somi_inference.core.paged_attention import (
    BlockAllocator,
    KVCache,
    KVCacheManager,
    paged_attention_decode,
)


class TestBlockAllocator:
    """Test BlockAllocator functionality."""

    def test_allocate_and_free(self):
        """Test basic allocation and freeing."""
        allocator = BlockAllocator(num_blocks=10)
        assert allocator.num_free_blocks() == 10

        # Allocate a block
        block_id = allocator.allocate()
        assert allocator.num_free_blocks() == 9
        assert block_id in allocator.ref_cnt
        assert allocator.ref_cnt[block_id] == 1

        # Free the block
        allocator.free(block_id)
        assert allocator.num_free_blocks() == 10
        assert block_id not in allocator.ref_cnt

    def test_reference_counting(self):
        """Test reference counting mechanism."""
        allocator = BlockAllocator(num_blocks=5)
        block_id = allocator.allocate()

        # Increase reference count
        allocator.increase_ref(block_id)
        assert allocator.ref_cnt[block_id] == 2

        # First free should not release the block
        allocator.free(block_id)
        assert allocator.num_free_blocks() == 4
        assert block_id in allocator.ref_cnt

        # Second free should release the block
        allocator.free(block_id)
        assert allocator.num_free_blocks() == 5
        assert block_id not in allocator.ref_cnt

    def test_copy_on_write_detection(self):
        """Test COW detection."""
        allocator = BlockAllocator(num_blocks=5)
        block_id = allocator.allocate()

        # Single reference - no COW needed
        assert not allocator.need_cow(block_id)

        # Multiple references - COW needed
        allocator.increase_ref(block_id)
        assert allocator.need_cow(block_id)


class TestKVCache:
    """Test KVCache functionality."""

    def test_write_and_read(self, device):
        """Test writing and reading KV cache."""
        num_blocks = 4
        block_size = 8
        num_heads = 4
        head_dim = 16

        cache = KVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Write to cache
        key = torch.randn(num_heads, head_dim)
        value = torch.randn(num_heads, head_dim)
        cache.write(block_id=0, slot=3, key=key, value=value)

        # Verify write
        assert torch.allclose(cache.key_cache[0, 3], key)
        assert torch.allclose(cache.value_cache[0, 3], value)

    def test_copy_block(self, device):
        """Test block copying."""
        cache = KVCache(num_blocks=4, block_size=8, num_heads=4, head_dim=16)

        # Write to source block
        for slot in range(8):
            key = torch.randn(4, 16)
            value = torch.randn(4, 16)
            cache.write(block_id=0, slot=slot, key=key, value=value)

        # Copy block
        cache.copy_block(src_block_id=0, dst_block_id=1)

        # Verify copy
        assert torch.allclose(cache.key_cache[0], cache.key_cache[1])
        assert torch.allclose(cache.value_cache[0], cache.value_cache[1])


class TestKVCacheManager:
    """Test KVCacheManager functionality."""

    def test_register_and_free_sequence(self):
        """Test sequence registration and freeing."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=8,
            num_heads=4,
            head_dim=16,
            n_layers=2,
        )

        # Register sequence
        manager.register_sequence(seq_id=0)
        assert 0 in manager.seq_to_block
        assert manager.get_num_tokens(0) == 0

        # Free sequence
        manager.free_sequence(seq_id=0)
        assert 0 not in manager.seq_to_block

    def test_allocate_slots(self):
        """Test slot allocation."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,  # Small block size for testing
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )

        manager.register_sequence(seq_id=0)

        # Allocate 5 tokens (should use 2 blocks: 4 + 1)
        manager.allocate_slots(seq_id=0, new_num_tokens=5)
        block_ids = manager.get_block_ids(0)
        assert len(block_ids) == 2

    def test_fork_sequence(self):
        """Test sequence forking with COW."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )

        # Setup source sequence
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=4)
        manager.advance_tokens(seq_id=0, num_tokens=4)

        # Fork sequence
        manager.fork_sequence(src_seq_id=0, dst_seq_id=1)

        # Verify fork
        assert manager.get_block_ids(0) == manager.get_block_ids(1)
        assert manager.get_num_tokens(0) == manager.get_num_tokens(1)

        # Verify reference count increased
        block_id = manager.get_block_ids(0)[0]
        assert manager.allocator.ref_cnt[block_id] == 2

    def test_copy_on_write(self):
        """Test COW when writing to shared block."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )

        # Setup and fork with partial block
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=3)  # Partial block
        manager.advance_tokens(seq_id=0, num_tokens=3)
        manager.fork_sequence(src_seq_id=0, dst_seq_id=1)

        original_block_id = manager.get_block_ids(1)[0]

        # Allocate more slots for seq 1 (should trigger COW since slot > 0)
        manager.allocate_slots(seq_id=1, new_num_tokens=1)

        # Verify COW happened
        new_block_id = manager.get_block_ids(1)[0]
        assert new_block_id != original_block_id

    def test_build_block_tables(self):
        """Test block table construction."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )

        # Setup two sequences with different lengths
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=8)
        manager.advance_tokens(seq_id=0, num_tokens=8)

        manager.register_sequence(seq_id=1)
        manager.allocate_slots(seq_id=1, new_num_tokens=4)
        manager.advance_tokens(seq_id=1, num_tokens=4)

        # Build block tables
        block_tables, seq_lens = manager.build_block_tables([0, 1])

        # Verify shapes
        assert block_tables.shape == (2, 2)  # 2 seqs, max 2 blocks
        assert seq_lens.shape == (2,)
        assert seq_lens[0] == 8
        assert seq_lens[1] == 4


class TestPagedAttentionDecode:
    """Test paged_attention_decode functionality."""

    def test_output_shape(self, device, seed):
        """Test output shape is correct."""
        num_seqs = 2
        num_heads = 4
        head_dim = 16
        num_blocks = 8
        block_size = 4
        max_blocks_per_seq = 3

        # Create mock inputs
        q = torch.randn(num_seqs, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks_per_seq))
        seq_lens = torch.tensor([10, 6])

        # Run paged attention
        output = paged_attention_decode(q, key_cache, value_cache, block_tables, seq_lens)

        # Verify shape
        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_causal_masking(self, device, seed):
        """Test that causal masking works (future tokens don't affect output)."""
        num_heads = 2
        head_dim = 8
        num_blocks = 4
        block_size = 4

        # Single sequence
        q = torch.randn(1, num_heads, head_dim)
        key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        value_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

        # Sequence length 5 (only first 5 tokens should be attended)
        block_tables = torch.tensor([[0, 1, 2, 3]])
        seq_lens = torch.tensor([5])

        output1 = paged_attention_decode(q, key_cache, value_cache, block_tables, seq_lens)

        # Modify tokens beyond seq_len (should not affect output)
        key_cache_modified = key_cache.clone()
        value_cache_modified = value_cache.clone()
        key_cache_modified[1, 2:] = torch.randn_like(key_cache_modified[1, 2:])
        value_cache_modified[1, 2:] = torch.randn_like(value_cache_modified[1, 2:])

        output2 = paged_attention_decode(
            q, key_cache_modified, value_cache_modified, block_tables, seq_lens
        )

        # Outputs should be identical (future tokens masked)
        assert torch.allclose(output1, output2, atol=1e-5)
