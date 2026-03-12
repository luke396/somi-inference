"""Qwen2.5 model adapter: bridges QwenModel with the inference engine."""

import torch
import torch.nn.functional as F

from somi_inference.core.paged_attention import KVCacheManager, paged_attention_decode
from somi_inference.models.qwen2 import (
    ForwardContext,
    ForwardMode,
    QwenModel,
    causal_attention,
)


class QwenAdapter:
    def __init__(self, model: QwenModel):
        self.model = model

    def _lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocab logits using tied embedding weights."""
        return F.linear(hidden_states, self.model.token_embedding.weight)

    def prefill(
        self,
        input_ids: torch.Tensor,  # (batch_size, seq_len)
        kv_manager: KVCacheManager,
        seq_id: int,
    ) -> torch.Tensor:

        def _make_prefill_attn(seq_id: int, kv_manager: KVCacheManager):
            def attn_fn(
                q,  # [batch_size=1, num_heads, seq_len, head_dim]
                k,
                v,
                layer_idx,
            ):
                k_write = k.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
                v_write = v.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
                kv_manager.write_kv(seq_id, layer_idx, k_write, v_write)
                return causal_attention(q, k, v)

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
        return self._lm_head(hidden_states)

    def decode(
        self,
        input_ids: torch.Tensor,  # (batch_size, 1)
        kv_manager: KVCacheManager,
        seq_ids: list[int],
    ) -> torch.Tensor:
        def _make_decode_attn(seq_ids: list[int], kv_manager: KVCacheManager):
            def attn_fn(q, k, v, layer_idx):
                # q, k, v: (batch_size, num_heads, 1, head_dim)
                for i, seq_id in enumerate(seq_ids):
                    k_write = k[i].squeeze(1)  # (num_heads, head_dim)
                    v_write = v[i].squeeze(1)
                    kv_manager.write_kv(seq_id, layer_idx, k_write, v_write)
                block_tables, seq_lens = kv_manager.build_block_tables(seq_ids)
                cache = kv_manager.kv_caches[layer_idx]
                q_decode = q.squeeze(2)  # (batch_size, num_heads, head_dim)
                attn_output = paged_attention_decode(
                    q_decode, cache.key_cache, cache.value_cache, block_tables, seq_lens
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
