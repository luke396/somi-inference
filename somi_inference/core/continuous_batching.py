"""Continuous batching with iteration-level scheduling."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.sampler import SamplingParams


class SequenceStatus(Enum):
    """Status of a sequence in the scheduling pipeline."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """A single request tracked through the scheduling pipeline."""

    seq_id: int
    status: SequenceStatus
    prompt_tokens: list[int]
    output_tokens: list[int]
    max_new_tokens: int
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    def __post_init__(self) -> None:
        """Validate sequence-level generation constraints."""
        assert self.max_new_tokens > 0, "max_new_tokens must be > 0"


@dataclass
class SchedulerOutput:
    """Result of a single scheduling step."""

    prefill_seq: list[Sequence]
    decode_seq: list[Sequence]
    freed_seq_ids: list[int]


class Scheduler:
    """FCFS scheduler with three queues: waiting, running, finished."""

    def __init__(
        self,
        max_concurrent: int,
        block_size: int,
        total_blocks: int,
    ) -> None:
        """Initialize the scheduler."""
        self.waiting: deque[Sequence] = deque()  # FIFO
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

        self.max_concurrent = max_concurrent
        self.block_size = block_size
        self.total_blocks = total_blocks

    def _required_blocks(self, seq: Sequence) -> int:
        """Return the worst-case number of blocks reserved for one sequence."""
        max_kv_tokens = len(seq.prompt_tokens) + seq.max_new_tokens - 1
        return (max_kv_tokens + self.block_size - 1) // self.block_size

    def add_request(self, seq: Sequence) -> None:
        """Add a sequence to the waiting queue."""
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling step: retire finished, admit new, build batch."""
        prefill_seq = []
        finished_seq_ids = []
        reserved_blocks = 0

        still_running = []
        for seq in self.running:
            if seq.status == SequenceStatus.FINISHED:
                self.finished.append(seq)
                finished_seq_ids.append(seq.seq_id)
                continue

            still_running.append(seq)
            reserved_blocks += self._required_blocks(seq)

        self.running = still_running

        decode_seq = list(self.running)  # all exiting running seq must be decoded
        available_blocks = self.total_blocks - reserved_blocks

        while self.waiting and len(self.running) < self.max_concurrent:
            next_seq = self.waiting[0]
            needed_blocks = self._required_blocks(next_seq)
            if needed_blocks > available_blocks:
                break

            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill_seq.append(seq)  # new added seq must be prefilled
            available_blocks -= needed_blocks

        return SchedulerOutput(
            prefill_seq=prefill_seq,
            decode_seq=decode_seq,
            freed_seq_ids=finished_seq_ids,
        )

    def has_unfinished(self) -> bool:
        """Check if there are unfinished sequences."""
        return bool(self.waiting) or bool(self.running)


class ContinuousBatchingEngine:
    """Engine that coordinates model, KV cache, and scheduler."""

    def __init__(
        self,
        model_runner: ModelRunner,
        scheduler: Scheduler,
        eos_token_id: int,
    ) -> None:
        """Initialize the engine."""
        self.model_runner = model_runner
        self.scheduler = scheduler
        self.eos_token_id = eos_token_id

    def _prefill(self, seq: Sequence) -> None:
        self.model_runner.kv_manager.register_sequence(seq.seq_id)
        token = self.model_runner.prefill(
            torch.tensor([seq.prompt_tokens]),  seq.seq_id, seq.sampling_params
        )  # (1, prompt_len, vocab_size)
        seq.output_tokens.append(token)
        self._check_finished(seq, token)

    def _check_finished(self, seq: Sequence, token: int) -> None:
        """Check if the sequence is finished after generating a new token.

        Even in the prefill stage, it's possible that the model generates
        an EOS token, which means the sequence is finished and doesn't
        need to be decoded anymore.
        """
        if token == self.eos_token_id or len(seq.output_tokens) >= seq.max_new_tokens:
            seq.status = SequenceStatus.FINISHED

    def _decode_batch(self, seqs: list[Sequence]) -> None:
        input_ids = torch.tensor([seq.output_tokens[-1] for seq in seqs]).unsqueeze(
            1
        )  # (batch_size, 1)
        seq_ids = [seq.seq_id for seq in seqs]
        token_histories = [seq.prompt_tokens +  seq.output_tokens for seq in seqs]
        tokens = self.model_runner.decode(
            input_ids,  seq_ids, [seq.sampling_params for seq in seqs], token_histories
        )  # (batch_size, 1, vocab_size)
        for i, seq in enumerate(seqs):
            token = int(tokens[i].item())
            seq.output_tokens.append(token)
            self._check_finished(seq, token)

    def run(
        self,
        requests: deque[tuple[int, Sequence]],  # (arrival_step, seq)
    ) -> list[Sequence]:
        """Run the engine loop until all requests are finished."""
        step = 0
        with torch.inference_mode():
            while requests or self.scheduler.has_unfinished():
                # inject all arrivals for this step
                while requests and requests[0][0] == step:
                    _, seq = requests.popleft()
                    self.scheduler.add_request(seq)
                # schedule
                output = self.scheduler.schedule()
                # free
                for seq_id in output.freed_seq_ids:
                    self.model_runner.kv_manager.free_sequence(seq_id)
                # prefill and decode
                for seq in output.prefill_seq:
                    self._prefill(seq)
                if output.decode_seq:
                    self._decode_batch(output.decode_seq)
                step += 1
        return self.scheduler.finished
