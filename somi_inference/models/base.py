"""Model adapter base class for inference engine."""

from typing import Protocol

import torch

from somi_inference.core.paged_attention import KVCacheManager


class ModelAdapter(Protocol):
    """Abstract interface for model adapters.

    Adapters bridge the gap between model implementations and the inference engine,
    handling prefill/decode logic and KV cache management.
    """

    def prefill(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> torch.Tensor:
        """Prefill KV cache with prompt tokens.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            kv_manager: KV cache manager
            seq_id: Sequence ID for this request

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)

        """
        ...

    def decode(
        self,
        input_ids: torch.Tensor,  # (batch_size, 1)
        kv_manager: KVCacheManager,
        seq_ids: list[int],
    ) -> torch.Tensor:
        """Decode one token per sequence using paged KV cache.

        Args:
            input_ids: Input token IDs, shape (batch_size, 1)
            kv_manager: KV cache manager
            seq_ids: List of sequence IDs in this batch

        Returns:
            logits: Output logits, shape (batch_size, 1, vocab_size)

        Note:
            Token positions are derived from kv_manager.get_num_tokens(seq_id).

        """
        ...
