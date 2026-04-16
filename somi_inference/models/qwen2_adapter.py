"""Qwen2.5 model adapter: bridges QwenModel with the inference engine."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812

from somi_inference.core.paged_attention import KVCacheManager, paged_attention_decode
from somi_inference.models.qwen2 import (
    ForwardContext,
    ForwardMode,
    MLPBackend,
    PrefillAttentionBackend,
    QwenMLP,
    QwenModel,
    causal_attention,
)

# Type alias for attention function signature
AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]


class QwenAdapter:
    """Adapter bridging QwenModel with KV cache management for inference."""

    def __init__(
        self,
        model: QwenModel,
        *,
        prefill_attention_backend: PrefillAttentionBackend = "auto",
        mlp_backend: MLPBackend = "auto",
    ) -> None:
        """Initialize adapter with a QwenModel instance."""
        self.model = model
        self.prefill_attention_backend = prefill_attention_backend
        self._mlp_backend: MLPBackend = "auto"
        self.mlp_backend = mlp_backend

    @property
    def mlp_backend(self) -> MLPBackend:
        """Return the configured MLP backend for all decoder layers."""
        return self._mlp_backend

    @mlp_backend.setter
    def mlp_backend(self, backend: MLPBackend) -> None:
        """Set the MLP backend for every decoder-layer MLP in the model."""
        for module in self.model.modules():
            if isinstance(module, QwenMLP):
                module.backend = backend
        self._mlp_backend = backend

    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocab logits using tied embedding weights."""
        return F.linear(hidden_states, self.model.token_embedding.weight)

    def prefill(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> torch.Tensor:
        """Prefill a single sequence: write KV cache and return last-token logits."""

        def _make_prefill_attn(seq_id: int, kv_manager: KVCacheManager) -> AttnFn:
            def attn_fn(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                layer_idx: int,
            ) -> torch.Tensor:
                k_write = k.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
                v_write = v.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
                kv_manager.write_kv(seq_id, layer_idx, k_write, v_write)
                return causal_attention(q, k, v, backend=self.prefill_attention_backend)

            return attn_fn

        seq_len = input_ids.size(1)
        kv_manager.allocate_slots(seq_id, seq_len)
        ctx = ForwardContext(
            mode=ForwardMode.PREFILL,
            attn_fn=_make_prefill_attn(seq_id, kv_manager),
            posi_idx=torch.arange(seq_len, device=input_ids.device).unsqueeze(0),
        )
        hidden_states = self.model(input_ids, ctx)
        kv_manager.advance_tokens(seq_id, seq_len)
        return self._lm_head(hidden_states[:, -1:, :])

    def decode(
        self,
        input_ids: torch.Tensor,  # (batch_size, 1)
        kv_manager: KVCacheManager,
        seq_ids: list[int],
    ) -> torch.Tensor:
        """Decode one token per sequence using paged KV cache attention."""

        def _make_decode_attn(seq_ids: list[int], kv_manager: KVCacheManager) -> AttnFn:
            def attn_fn(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                layer_idx: int,
            ) -> torch.Tensor:
                # q, k, v: (batch_size, num_heads, 1, head_dim)
                for i, seq_id in enumerate(seq_ids):
                    k_write = k[i].squeeze(1)  # (num_heads, head_dim)
                    v_write = v[i].squeeze(1)
                    kv_manager.write_kv(seq_id, layer_idx, k_write, v_write)
                block_tables, seq_lens = kv_manager.build_block_tables(seq_ids)
                seq_lens = seq_lens + 1  # include the just-written token
                cache = kv_manager.kv_caches[layer_idx]
                q_decode = q.squeeze(2)  # (batch_size, num_heads, head_dim)
                attn_output = paged_attention_decode(
                    q_decode, cache.kv_cache, block_tables, seq_lens
                )  # (batch_size, num_heads, head_dim)
                return attn_output.unsqueeze(2)  # (batch_size, num_heads, 1, head_dim)

            return attn_fn

        for seq_id in seq_ids:
            kv_manager.allocate_slots(seq_id, 1)
        posi_idx = torch.tensor(
            [kv_manager.get_num_tokens(seq_id) for seq_id in seq_ids],
            device=input_ids.device,
        ).unsqueeze(1)  # (batch_size,1)
        ctx = ForwardContext(
            mode=ForwardMode.DECODE,
            attn_fn=_make_decode_attn(seq_ids, kv_manager),
            posi_idx=posi_idx,
        )
        hidden_states = self.model(input_ids, ctx)
        for seq_id in seq_ids:
            kv_manager.advance_tokens(seq_id, 1)
        return self._lm_head(hidden_states)


def _map_hf_key(hf_key: str) -> str | None:
    """Map HF state_dict key to somi key, or None to skip.

    Mapping rules:
    - Strip 'model.' prefix
    - 'embed_tokens' -> 'token_embedding'
    - 'norm' -> 'final_layernorm'
    - `mlp.gate_proj.weight` / `mlp.up_proj.weight` are merged separately
    - Skip 'lm_head.weight' (tied weights)
    - Skip 'rotary_emb.*' (computed buffers)
    """
    # Skip lm_head (tied weights)
    if hf_key == "lm_head.weight":
        return None

    # Skip rotary_emb (computed buffers)
    if "rotary_emb" in hf_key:
        return None

    # Strip 'model.' prefix
    key = hf_key.removeprefix("model.")

    # Rename embed_tokens -> token_embedding
    if key == "embed_tokens.weight":
        return "token_embedding.weight"

    # Rename norm -> final_layernorm
    if key == "norm.weight":
        return "final_layernorm.weight"

    return key


def _map_hf_gate_up_proj_key(hf_key: str) -> tuple[str, int] | None:
    """Map HF MLP gate/up shards onto the merged somi `gate_up_proj` weight."""
    key = hf_key.removeprefix("model.")
    if key.endswith(".mlp.gate_proj.weight"):
        return key.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight"), 0
    if key.endswith(".mlp.up_proj.weight"):
        return key.replace(".mlp.up_proj.weight", ".mlp.gate_up_proj.weight"), 1
    return None


def load_from_hf(model_name: str) -> QwenAdapter:
    """Load Qwen model from Hugging Face and create somi adapter.

    Args:
        model_name: HF model name (e.g., "Qwen/Qwen2.5-1.5B")

    Returns:
        QwenAdapter with loaded weights

    """
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    # Load HF config
    hf_config = AutoConfig.from_pretrained(model_name)

    # Extract rope_theta from rope_parameters
    rope_theta = 1_000_000.0  # Default value
    if hf_config.rope_parameters is not None:
        rope_theta = hf_config.rope_parameters.get("rope_theta", rope_theta)

    # Create somi model
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    somi_model = QwenModel(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_size=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta,
    )

    # Load HF weights
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    hf_state_dict = hf_model.state_dict()

    # Map and load weights
    somi_state_dict = {}
    merged_gate_up_proj_shards: dict[str, dict[int, torch.Tensor]] = {}
    for hf_key, hf_tensor in hf_state_dict.items():
        merged_gate_up_proj = _map_hf_gate_up_proj_key(hf_key)
        if merged_gate_up_proj is not None:
            somi_key, shard_idx = merged_gate_up_proj
            shard_map = merged_gate_up_proj_shards.setdefault(somi_key, {})
            shard_map[shard_idx] = hf_tensor
            continue
        somi_key = _map_hf_key(hf_key)
        if somi_key is not None:
            somi_state_dict[somi_key] = hf_tensor
    for somi_key, shard_map in merged_gate_up_proj_shards.items():
        if set(shard_map) != {0, 1}:
            msg = f"Missing gate/up shard for merged MLP weight: {somi_key}"
            raise KeyError(msg)
        somi_state_dict[somi_key] = torch.cat((shard_map[0], shard_map[1]), dim=0)

    # Load into somi model
    somi_model.load_state_dict(somi_state_dict, strict=True)
    somi_model.requires_grad_(requires_grad=False)

    return QwenAdapter(somi_model)
