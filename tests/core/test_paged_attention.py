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
