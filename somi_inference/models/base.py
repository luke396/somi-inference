"""Model adapter base class for inference engine."""
from typing import Protocol
import torch
from somi_inference.core.paged_attention import KVCacheManager


class ModelAdapter(Protocol):
    """Abstract interface for model adapters.

    Phase 1.2 will implement QwenAdapter for Qwen2.5-1.5B.
    """

    def prefill(
        self,
        prompt_tokens: torch.Tensor,  # (1, prompt_len)
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> torch.Tensor:
        """Prefill KV cache with prompt tokens.

        Args:
            prompt_tokens: Input token IDs, shape (1, prompt_len)
            kv_manager: KV cache manager
            seq_id: Sequence ID for this request

        Returns:
            logits: Output logits, shape (1, prompt_len, vocab_size)
        """
        raise NotImplementedError("Phase 1.2: Implement in QwenAdapter")

    def decode(
        self,
        input_ids: torch.Tensor,  # (batch_size, 1)
        kv_manager: KVCacheManager,
        seq_ids: list[int],
        positions: torch.Tensor,  # (batch_size, 1)
    ) -> torch.Tensor:
        """Decode one token per sequence using paged KV cache.

        Args:
            input_ids: Input token IDs, shape (batch_size, 1)
            kv_manager: KV cache manager
            seq_ids: List of sequence IDs in this batch
            positions: Token positions, shape (batch_size, 1)

        Returns:
            logits: Output logits, shape (batch_size, 1, vocab_size)
        """
        raise NotImplementedError("Phase 1.2: Implement in QwenAdapter")
