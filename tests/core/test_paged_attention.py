"""Tests for paged attention components."""

from __future__ import annotations

import pytest
import torch

from somi_inference.core.paged_attention import (
    BlockAllocator,
    KVCache,
    KVCacheManager,
    pack_kv_cache,
    paged_attention_decode,
    paged_attention_decode_torch_ref,
)


def _make_random_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    return pack_kv_cache(key_cache, value_cache)


class TestBlockAllocator:
    """Test BlockAllocator functionality."""

    def test_allocate_and_free(self):
        """Test basic allocation and freeing."""
        allocator = BlockAllocator(num_blocks=10)
        assert allocator.num_free_blocks() == 10

        block_id = allocator.allocate()
        assert allocator.num_free_blocks() == 9
        assert block_id in allocator.ref_cnt
        assert allocator.ref_cnt[block_id] == 1

        allocator.free(block_id)
        assert allocator.num_free_blocks() == 10
        assert block_id not in allocator.ref_cnt

    def test_reference_counting(self):
        """Test reference counting mechanism."""
        allocator = BlockAllocator(num_blocks=5)
        block_id = allocator.allocate()

        allocator.increase_ref(block_id)
        assert allocator.ref_cnt[block_id] == 2

        allocator.free(block_id)
        assert allocator.num_free_blocks() == 4
        assert block_id in allocator.ref_cnt

        allocator.free(block_id)
        assert allocator.num_free_blocks() == 5
        assert block_id not in allocator.ref_cnt

    def test_copy_on_write_detection(self):
        """Test COW detection."""
        allocator = BlockAllocator(num_blocks=5)
        block_id = allocator.allocate()

        assert not allocator.need_cow(block_id)

        allocator.increase_ref(block_id)
        assert allocator.need_cow(block_id)


class TestKVCache:
    """Test KVCache functionality."""

    def test_write_and_read(self, device):
        """Test writing and reading fused KV cache."""
        cache = KVCache(
            num_blocks=4,
            block_size=8,
            num_kv_heads=4,
            head_dim=16,
        )
        key = torch.randn(4, 16)
        value = torch.randn(4, 16)
        cache.write(block_id=0, slot=3, key=key, value=value)

        assert cache.kv_cache.shape == (4, 2, 8, 4, 16)
        assert torch.allclose(cache.key_cache[0, 3], key)
        assert torch.allclose(cache.value_cache[0, 3], value)

    def test_copy_block(self, device):
        """Test fused block copying."""
        cache = KVCache(num_blocks=4, block_size=8, num_kv_heads=4, head_dim=16)

        for slot in range(8):
            key = torch.randn(4, 16)
            value = torch.randn(4, 16)
            cache.write(block_id=0, slot=slot, key=key, value=value)

        cache.copy_block(src_block_id=0, dst_block_id=1)

        assert torch.allclose(cache.kv_cache[0], cache.kv_cache[1])
        assert torch.allclose(cache.key_cache[0], cache.key_cache[1])
        assert torch.allclose(cache.value_cache[0], cache.value_cache[1])


class TestKVCacheManager:
    """Test KVCacheManager functionality."""

    def test_register_and_free_sequence(self):
        """Test sequence registration and freeing."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=8,
            num_kv_heads=4,
            head_dim=16,
            n_layers=2,
        )
        manager.register_sequence(seq_id=0)
        assert 0 in manager.seq_to_block
        assert manager.get_num_tokens(0) == 0
        manager.allocate_slots(seq_id=0, new_num_tokens=1)
        block_id = manager.get_block_ids(0)[0]
        assert block_id in manager.allocator.ref_cnt

        manager.free_sequence(seq_id=0)
        assert 0 not in manager.seq_to_block
        assert block_id not in manager.allocator.ref_cnt

    def test_register_sequence_rejects_duplicate_ids(self):
        """Registering the same sequence twice should raise an error."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=8,
            num_kv_heads=4,
            head_dim=16,
            n_layers=2,
        )
        manager.register_sequence(seq_id=0)

        with pytest.raises(ValueError, match="already registered"):
            manager.register_sequence(seq_id=0)

    def test_allocate_slots(self):
        """Test slot allocation."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            n_layers=1,
        )
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=5)
        assert len(manager.get_block_ids(0)) == 2

    def test_fork_sequence(self):
        """Test sequence forking with COW."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            n_layers=1,
        )
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=4)
        manager.advance_tokens(seq_id=0, num_tokens=4)
        manager.fork_sequence(src_seq_id=0, dst_seq_id=1)

        assert manager.get_block_ids(0) == manager.get_block_ids(1)
        assert manager.get_num_tokens(0) == manager.get_num_tokens(1)
        block_id = manager.get_block_ids(0)[0]
        assert manager.allocator.ref_cnt[block_id] == 2

    def test_copy_on_write(self):
        """Test COW when writing to a shared partial block."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            n_layers=1,
        )
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=3)
        manager.advance_tokens(seq_id=0, num_tokens=3)
        manager.fork_sequence(src_seq_id=0, dst_seq_id=1)

        original_block_id = manager.get_block_ids(1)[0]
        manager.allocate_slots(seq_id=1, new_num_tokens=1)
        assert manager.get_block_ids(1)[0] != original_block_id

    def test_build_block_tables(self):
        """Test block table construction."""
        manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            n_layers=1,
        )
        manager.register_sequence(seq_id=0)
        manager.allocate_slots(seq_id=0, new_num_tokens=8)
        manager.advance_tokens(seq_id=0, num_tokens=8)

        manager.register_sequence(seq_id=1)
        manager.allocate_slots(seq_id=1, new_num_tokens=4)
        manager.advance_tokens(seq_id=1, num_tokens=4)

        block_tables, seq_lens = manager.build_block_tables([0, 1])
        assert block_tables.shape == (2, 2)
        assert seq_lens.shape == (2,)
        assert seq_lens[0] == 8
        assert seq_lens[1] == 4


class TestPagedAttentionDecode:
    """Test paged_attention_decode functionality."""

    def test_output_shape(self, device, seed):
        """Test output shape is correct."""
        q = torch.randn(2, 4, 16)
        kv_cache = _make_random_kv_cache(8, 4, 4, 16)
        block_tables = torch.randint(0, 8, (2, 3))
        seq_lens = torch.tensor([10, 6])

        output = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        assert output.shape == (2, 4, 16)

    def test_causal_masking(self, device, seed):
        """Future tokens past ``seq_len`` should not affect output."""
        q = torch.randn(1, 2, 8)
        kv_cache = _make_random_kv_cache(4, 4, 2, 8)
        block_tables = torch.tensor([[0, 1, 2, 3]])
        seq_lens = torch.tensor([5])

        output1 = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        kv_cache_modified = kv_cache.clone()
        kv_cache_modified[1, :, 2:] = torch.randn_like(kv_cache_modified[1, :, 2:])
        output2 = paged_attention_decode(q, kv_cache_modified, block_tables, seq_lens)

        assert torch.allclose(output1, output2, atol=1e-5)

    def test_gqa_support(self, device, seed):
        """GQA with ``num_q_heads > num_kv_heads`` should still work."""
        q = torch.randn(2, 12, 128)
        kv_cache = _make_random_kv_cache(8, 16, 2, 128)
        block_tables = torch.randint(0, 8, (2, 3))
        seq_lens = torch.tensor([20, 25])

        output = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        assert output.shape == (2, 12, 128)

    def test_mha_backward_compatibility(self, device, seed):
        """MHA should still work when Q and KV head counts match."""
        q = torch.randn(2, 8, 64)
        kv_cache = _make_random_kv_cache(4, 16, 8, 64)
        block_tables = torch.randint(0, 4, (2, 2))
        seq_lens = torch.tensor([15, 20])

        output = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        assert output.shape == (2, 8, 64)

    def test_bfloat16_cpu_support(self, device, seed):
        """CPU bfloat16 inputs should use the reference backend cleanly."""
        q = torch.randn(1, 4, 16, dtype=torch.bfloat16)
        kv_cache = _make_random_kv_cache(2, 4, 4, 16, dtype=torch.bfloat16)
        block_tables = torch.tensor([[0, 1]])
        seq_lens = torch.tensor([6])

        output = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        assert output.dtype == torch.bfloat16
        assert output.shape == (1, 4, 16)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_reduced_precision_matches_dense_reference(
        self, dtype: torch.dtype
    ) -> None:
        """Reduced precision should still match dense attention reasonably well."""
        torch.manual_seed(42)
        num_seqs = 1
        num_heads = 4
        head_dim = 16
        num_blocks = 2
        block_size = 4
        seq_len = 6

        q = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype)
        kv_cache = _make_random_kv_cache(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=dtype,
        )
        block_tables = torch.tensor([[1, 0]])
        seq_lens = torch.tensor([seq_len])

        output = paged_attention_decode(q, kv_cache, block_tables, seq_lens)

        dense_keys = torch.stack(
            [
                kv_cache[block_tables[0, pos // block_size], 0, pos % block_size]
                for pos in range(seq_len)
            ],
            dim=0,
        ).unsqueeze(0)
        dense_values = torch.stack(
            [
                kv_cache[block_tables[0, pos // block_size], 1, pos % block_size]
                for pos in range(seq_len)
            ],
            dim=0,
        ).unsqueeze(0)
        scores = torch.einsum("s h d, s t h d -> s h t", q, dense_keys)
        scores = (scores * (head_dim**-0.5)).to(torch.float32)
        attn = torch.softmax(scores, dim=-1).to(dense_values.dtype)
        expected = torch.einsum("s h t, s t h d -> s h d", attn, dense_values)

        assert output.dtype == dtype
        torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)

    def test_gqa_invalid_ratio(self, device, seed):
        """Invalid GQA ratios should still raise an assertion."""
        q = torch.randn(1, 7, 64)
        kv_cache = _make_random_kv_cache(4, 8, 2, 64)
        block_tables = torch.tensor([[0, 1]])
        seq_lens = torch.tensor([10])

        with pytest.raises(
            AssertionError, match="num_q_heads must be divisible by num_kv_heads"
        ):
            paged_attention_decode(q, kv_cache, block_tables, seq_lens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton")
class TestPagedAttentionDecodeTriton:
    """Test the Triton decode backend against the PyTorch reference."""

    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_triton_matches_torch_reference(self, dtype: torch.dtype) -> None:
        """Triton decode should numerically match the reference backend."""
        from somi_inference.core.paged_attention_triton import TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton is not installed")

        device = torch.device("cuda")
        torch.manual_seed(42)
        q = torch.randn(2, 8, 64, device=device, dtype=dtype)
        kv_cache = _make_random_kv_cache(
            16,
            8,
            2,
            64,
            dtype=dtype,
            device=device,
        )
        block_tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
        seq_lens = torch.tensor([25, 17], device=device)

        expected = paged_attention_decode_torch_ref(q, kv_cache, block_tables, seq_lens)
        actual = paged_attention_decode(
            q, kv_cache, block_tables, seq_lens, backend="triton"
        )

        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
