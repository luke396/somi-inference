"""Tests for continuous batching components."""
import pytest
import torch
from collections import deque
from somi_inference.core.continuous_batching import (
    SequenceStatus,
    Sequence,
    Scheduler,
    ContinuousBatchingEngine,
)
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.base import ModelAdapter


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.prefill_calls = []
        self.decode_calls = []

    def prefill(self, prompt_tokens, kv_manager, seq_id):
        """Mock prefill - return random logits."""
        self.prefill_calls.append((prompt_tokens, seq_id))
        batch_size, seq_len = prompt_tokens.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)

    def decode(self, input_ids, kv_manager, seq_ids, positions):
        """Mock decode - return random logits."""
        self.decode_calls.append((input_ids, seq_ids))
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, 1, self.vocab_size)


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

    def test_single_sequence_prefill_decode(self):
        """Test single sequence prefill and decode."""
        kv_manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=kv_manager.allocator.num_free_blocks,
        )
        model = MockModelAdapter(vocab_size=100)
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
        assert len(model.prefill_calls) == 1
        assert len(model.decode_calls) >= 2  # At least 2 decode steps

    def test_eos_token_handling(self):
        """Test early stopping on EOS token."""
        kv_manager = KVCacheManager(
            num_blocks=10,
            block_size=4,
            num_heads=2,
            head_dim=8,
            n_layers=1,
        )
        scheduler = Scheduler(
            max_concurrent=2,
            block_size=4,
            free_block_num_fn=kv_manager.allocator.num_free_blocks,
        )

        # Mock adapter that always returns EOS token
        class EOSAdapter(MockModelAdapter):
            def prefill(self, prompt_tokens, kv_manager, seq_id):
                batch_size, seq_len = prompt_tokens.shape
                logits = torch.zeros(batch_size, seq_len, self.vocab_size)
                logits[:, :, 2] = 100.0  # EOS token has highest logit
                return logits

            def decode(self, input_ids, kv_manager, seq_ids, positions):
                batch_size = input_ids.shape[0]
                logits = torch.zeros(batch_size, 1, self.vocab_size)
                logits[:, :, 2] = 100.0
                return logits

        model = EOSAdapter(vocab_size=100)
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
