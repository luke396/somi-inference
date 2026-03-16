"""Tests for continuous batching components."""

import torch
from collections import deque

import pytest

from somi_inference.core.continuous_batching import (
    SequenceStatus,
    Sequence,
    Scheduler,
    ContinuousBatchingEngine,
)
from somi_inference.core.paged_attention import KVCacheManager


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(self, vocab_size=100, return_token=None):
        self.vocab_size = vocab_size
        self.return_token = return_token  # If set, always return this token
        self.prefill_calls = []
        self.decode_calls = []

    def prefill(self, input_ids, kv_manager, seq_id):
        """Mock prefill - return logits that produce a specific token."""
        self.prefill_calls.append((input_ids, seq_id))
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)

        # If return_token is set, make sure argmax returns that token
        if self.return_token is not None:
            logits[:, :, self.return_token] = 100.0

        return logits

    def decode(self, input_ids, kv_manager, seq_ids):
        """Mock decode - return logits that produce a specific token."""
        self.decode_calls.append((input_ids, seq_ids))
        batch_size = input_ids.shape[0]
        logits = torch.randn(batch_size, 1, self.vocab_size)

        # If return_token is set, make sure argmax returns that token
        if self.return_token is not None:
            logits[:, :, self.return_token] = 100.0

        return logits


@pytest.fixture
def kv_manager():
    """Create a KVCacheManager for testing."""
    return KVCacheManager(
        num_blocks=10,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        n_layers=1,
    )


@pytest.fixture
def scheduler(kv_manager):
    """Create a Scheduler for testing."""
    return Scheduler(
        max_concurrent=2,
        block_size=4,
        free_block_num_fn=kv_manager.allocator.num_free_blocks,
    )


class TestScheduler:
    """Test Scheduler functionality."""

    def test_add_request(self):
        """Test adding requests to waiting queue."""
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=lambda: 10,
        )

        seq = Sequence(
            seq_id=0,
            status=SequenceStatus.WAITING,
            prompt_tokens=[1, 2, 3],
            output_tokens=[],
            max_new_tokens=10,
        )

        scheduler.add_request(seq)
        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0] == seq

    def test_schedule_admit_new(self):
        """Test admitting new sequences from waiting queue."""
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=lambda: 10,
        )

        # Add two sequences
        for i in range(2):
            seq = Sequence(
                seq_id=i,
                status=SequenceStatus.WAITING,
                prompt_tokens=[1, 2, 3],
                output_tokens=[],
                max_new_tokens=10,
            )
            scheduler.add_request(seq)

        # Schedule
        output = scheduler.schedule()

        # Both should be admitted (prefill)
        assert len(output.prefill_seq) == 2
        assert len(output.decode_seq) == 0
        assert len(scheduler.running) == 2
        assert len(scheduler.waiting) == 0

    def test_schedule_max_concurrent(self):
        """Test max concurrent limit."""
        scheduler = Scheduler(
            max_concurrent=1,
            block_size=4,
            free_block_num_fn=lambda: 10,
        )

        # Add two sequences
        for i in range(2):
            seq = Sequence(
                seq_id=i,
                status=SequenceStatus.WAITING,
                prompt_tokens=[1, 2, 3],
                output_tokens=[],
                max_new_tokens=10,
            )
            scheduler.add_request(seq)

        # First schedule - admit one
        output = scheduler.schedule()
        assert len(output.prefill_seq) == 1
        assert len(scheduler.running) == 1
        assert len(scheduler.waiting) == 1

        # Second schedule - decode existing, admit next
        output = scheduler.schedule()
        assert len(output.decode_seq) == 1
        assert len(output.prefill_seq) == 0

    def test_schedule_insufficient_blocks(self):
        """Test blocking when insufficient free blocks."""
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=lambda: 0,  # No free blocks
        )

        seq = Sequence(
            seq_id=0,
            status=SequenceStatus.WAITING,
            prompt_tokens=[1, 2, 3, 4, 5],  # 5 tokens, needs 2 blocks
            output_tokens=[],
            max_new_tokens=10,
        )
        scheduler.add_request(seq)

        # Schedule should not admit (insufficient blocks)
        output = scheduler.schedule()
        assert len(output.prefill_seq) == 0
        assert len(scheduler.waiting) == 1

    def test_schedule_retire_finished(self):
        """Test retiring finished sequences."""
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=lambda: 10,
        )

        # Add and admit a sequence
        seq = Sequence(
            seq_id=0,
            status=SequenceStatus.WAITING,
            prompt_tokens=[1, 2, 3],
            output_tokens=[],
            max_new_tokens=10,
        )
        scheduler.add_request(seq)
        scheduler.schedule()

        # Mark as finished
        scheduler.running[0].status = SequenceStatus.FINISHED

        # Schedule should retire it
        output = scheduler.schedule()
        assert len(output.freed_seq_ids) == 1
        assert output.freed_seq_ids[0] == 0
        assert len(scheduler.running) == 0
        assert len(scheduler.finished) == 1


class TestContinuousBatchingEngine:
    """Test ContinuousBatchingEngine functionality."""

    def test_single_sequence_prefill_decode(self, kv_manager, scheduler):
        """Test single sequence prefill and decode."""
        # Use return_token=5 to avoid EOS token (2)
        model = MockModelAdapter(vocab_size=100, return_token=5)
        engine = ContinuousBatchingEngine(
            model=model,
            kv_manager=kv_manager,
            scheduler=scheduler,
            eos_token_id=2,
        )

        # Create request
        seq = Sequence(
            seq_id=0,
            status=SequenceStatus.WAITING,
            prompt_tokens=[1, 2, 3],
            output_tokens=[],
            max_new_tokens=3,
        )
        requests = deque([(0, seq)])

        # Run engine
        finished = engine.run(requests)

        # Verify
        assert len(finished) == 1
        assert finished[0].seq_id == 0
        assert len(finished[0].output_tokens) == 3
        assert all(token == 5 for token in finished[0].output_tokens)
        assert len(model.prefill_calls) == 1
        assert len(model.decode_calls) >= 2  # At least 2 decode steps

    def test_eos_token_handling(self, kv_manager, scheduler):
        """Test early stopping on EOS token."""
        # Use return_token=2 to always return EOS token
        model = MockModelAdapter(vocab_size=100, return_token=2)
        engine = ContinuousBatchingEngine(
            model=model,
            kv_manager=kv_manager,
            scheduler=scheduler,
            eos_token_id=2,
        )

        seq = Sequence(
            seq_id=0,
            status=SequenceStatus.WAITING,
            prompt_tokens=[1, 2, 3],
            output_tokens=[],
            max_new_tokens=10,
        )
        requests = deque([(0, seq)])

        finished = engine.run(requests)

        # Should stop after first token (EOS)
        assert len(finished[0].output_tokens) == 1
        assert finished[0].output_tokens[0] == 2
