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
