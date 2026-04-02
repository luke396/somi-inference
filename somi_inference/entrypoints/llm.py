"""High-level text-in/text-out inference API."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from somi_inference.core.continuous_batching import (
    ContinuousBatchingEngine,
    Scheduler,
    Sequence,
    SequenceStatus,
)
from somi_inference.core.model_runner import ModelRunner
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import Sampler, SamplingParams
from somi_inference.models.loader import load_model
from somi_inference.tokenizer import Tokenizer

DEFAULT_BLOCK_SIZE = 16
DEFAULT_MAX_CONCURRENT = 16


def _require_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    assert isinstance(value, int), f"Expected integer config value for {key}"
    return value


class LLM:
    """Minimal high-level inference wrapper for one model."""

    def __init__(
        self,
        model_name: str,
        *,
        num_blocks: int = 256,
        block_size: int = DEFAULT_BLOCK_SIZE,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Initialize tokenizer, model, KV cache, scheduler, and engine."""
        self.tokenizer = Tokenizer(model_name)
        adapter, config = load_model(model_name)

        hidden_size = _require_int(config, "hidden_size")
        num_attention_heads = _require_int(config, "num_attention_heads")
        num_key_value_heads = _require_int(config, "num_key_value_heads")
        n_layers = _require_int(config, "num_hidden_layers")

        assert hidden_size % num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )
        head_dim = hidden_size // num_attention_heads

        self.kv_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            n_layers=n_layers,
        )
        self.sampler = Sampler()
        self.model_runner = ModelRunner(adapter, self.sampler, self.kv_manager)
        self.scheduler = Scheduler(
            max_concurrent=max_concurrent,
            block_size=block_size,
            total_blocks=num_blocks,
        )
        self.engine = ContinuousBatchingEngine(
            self.model_runner,
            self.scheduler,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self._next_seq_id = 0

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate text for one prompt and return only newly generated text."""
        prompt_tokens = self.tokenizer.encode(prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        seq = Sequence(
            seq_id=self._next_seq_id,
            status=SequenceStatus.WAITING,
            prompt_tokens=prompt_tokens,
            output_tokens=[],
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
        )
        self._next_seq_id += 1

        finished = self.engine.run(deque([(0, seq)]))
        assert len(finished) == 1, "generate expects exactly one finished sequence"
        return self.tokenizer.decode(finished[0].output_tokens)

    def generate_stream(self, prompt: str, *, max_new_tokens: int) -> Iterator[str]:
        """Streaming generation is intentionally deferred."""
        del prompt, max_new_tokens
        message = "Streaming not yet implemented"
        raise NotImplementedError(message)
