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
