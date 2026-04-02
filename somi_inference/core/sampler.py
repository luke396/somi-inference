"""Sampling strategies for token generation."""

from dataclasses import dataclass

import torch
from torch import Tensor

LOGITS_BATCH_NDIM = 2


@dataclass
class SamplingParams:
    """Sampling configuration for a single sequence."""

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        """Validate sampling parameter ranges."""
        assert self.temperature >= 0.0, "temperature must be >= 0.0"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or > 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0.0, 1.0]"
        assert self.repetition_penalty > 0.0, "repetition_penalty must be > 0.0"


class Sampler:
    """Convert model logits into next-token ids."""

    def sample(
        self,
        logits: Tensor,
        params: SamplingParams | list[SamplingParams],
        token_histories: list[list[int]] | None = None,
    ) -> Tensor:
        """Sample one token for each row in a batch of logits."""
        assert logits.ndim == LOGITS_BATCH_NDIM, (
            f"logits must be 2D, got shape {tuple(logits.shape)}"
        )

        batch_size = logits.size(0)
        params_list = self._normalize_params(params, batch_size)
        token_histories_list = self._normalize_token_histories(
            token_histories, batch_size
        )

        logits = logits.clone()
        logits = self._apply_repetition_penalty(
            logits, params_list, token_histories_list
        )
        logits = self._apply_temperature(logits, params_list)
        logits = self._apply_top_k(logits, params_list)
        logits = self._apply_top_p(logits, params_list)
        return self._sample_tokens(logits, params_list)

    def _normalize_params(
        self,
        params: SamplingParams | list[SamplingParams],
        batch_size: int,
    ) -> list[SamplingParams]:
        if isinstance(params, SamplingParams):
            return [params] * batch_size

        assert len(params) == batch_size, "Length of params must match batch size"
        return params

    def _normalize_token_histories(
        self,
        token_histories: list[list[int]] | None,
        batch_size: int,
    ) -> list[list[int]]:
        if token_histories is None:
            return [[] for _ in range(batch_size)]

        assert len(token_histories) == batch_size, (
            "Length of token_histories must match batch size"
        )
        return token_histories

    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        params: list[SamplingParams],
        token_histories: list[list[int]],
    ) -> Tensor:
        for batch_index, (param, history) in enumerate(
            zip(params, token_histories, strict=True)
        ):
            if param.repetition_penalty == 1.0 or not history:
                continue
            token_ids = torch.tensor(
                list(set(history)),
                device=logits.device,
                dtype=torch.long,
            )
            token_logits = logits[batch_index, token_ids]
            logits[batch_index, token_ids] = torch.where(
                token_logits > 0,
                token_logits / param.repetition_penalty,
                token_logits * param.repetition_penalty,
            )

        return logits

    def _apply_temperature(
        self,
        logits: Tensor,
        params: list[SamplingParams],
    ) -> Tensor:
        for batch_index, param in enumerate(params):
            if param.temperature > 0.0:
                logits[batch_index] = logits[batch_index] / param.temperature

        return logits

    def _apply_top_k(
        self,
        logits: Tensor,
        params: list[SamplingParams],
    ) -> Tensor:
        for batch_index, param in enumerate(params):
            if param.temperature == 0.0 or param.top_k == -1:
                continue

            k = min(param.top_k, logits.size(-1))
            top_k_indices = torch.topk(logits[batch_index], k=k).indices
            keep_mask = torch.zeros_like(logits[batch_index], dtype=torch.bool)
            keep_mask[top_k_indices] = True
            logits[batch_index] = logits[batch_index].masked_fill(
                ~keep_mask, -torch.inf
            )

        return logits

    def _apply_top_p(
        self,
        logits: Tensor,
        params: list[SamplingParams],
    ) -> Tensor:
        for batch_index, param in enumerate(params):
            if param.temperature == 0.0 or param.top_p == 1.0:
                continue

            sorted_logits, sorted_indices = torch.sort(
                logits[batch_index], descending=True
            )
            sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > param.top_p
            # prevent removing the first token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[batch_index, indices_to_remove] = -torch.inf

        return logits

    def _sample_tokens(
        self,
        logits: Tensor,
        params: list[SamplingParams],
    ) -> Tensor:
        tokens: list[int] = []

        for batch_index, param in enumerate(params):
            if param.temperature == 0.0:
                token = int(torch.argmax(logits[batch_index]).item())
            else:
                probs = torch.softmax(logits[batch_index], dim=-1, dtype=torch.float32)
                token = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(token)

        return torch.tensor(tokens, device=logits.device, dtype=torch.long)
