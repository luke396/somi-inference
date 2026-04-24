"""Model execution layer combining ModelAdapter and Sampler."""

from torch import Tensor

from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.core.sampler import Sampler, SamplingParams
from somi_inference.models.base import ModelAdapter


class ModelRunner:
    """Execute model forward passes and delegate token sampling."""

    def __init__(
        self,
        adapter: ModelAdapter,
        sampler: Sampler,
        kv_manager: KVCacheManager,
    ) -> None:
        """Initialize the execution layer."""
        self.adapter = adapter
        self.sampler = sampler
        self.kv_manager = kv_manager

    def prefill(
        self,
        input_ids: Tensor,
        seq_id: int,
        sampling_params: SamplingParams,
    ) -> int:
        """Run prefill for one sequence and sample the first output token."""
        assert input_ids.size(0) == 1, "prefill expects a single sequence"
        logits = self.adapter.prefill(input_ids, self.kv_manager, seq_id)
        sampled_token = self.sampler.sample(
            logits[:, 0, :],
            sampling_params,
            token_histories=[input_ids[0].tolist()],
        )
        assert sampled_token.shape[0] == 1, (
            "Expected a single token output for prefill sampling."
        )
        return int(sampled_token.item())

    def decode(
        self,
        input_ids: Tensor,
        seq_ids: list[int],
        sampling_params: list[SamplingParams],
        token_histories: list[list[int]],
    ) -> Tensor:
        """Run batch decode and sample one token per sequence."""
        logits = self.adapter.decode(input_ids, self.kv_manager, seq_ids)
        sampled_tokens = self.sampler.sample(
            logits[:, 0, :],
            sampling_params,
            token_histories=token_histories,
        )
        assert sampled_tokens.shape[0] == len(seq_ids), (
            "Expected one sampled token per sequence in decode."
        )
        return sampled_tokens
