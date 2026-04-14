"""Qwen 2.5 base components: RMSNorm, RotaryEmbedding, causal_attention."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import torch
from torch import nn

from somi_inference.core.flash_attention_triton import (
    causal_attention_triton,
    triton_causal_attention_supported,
)

PrefillAttentionBackend = Literal["auto", "torch_ref", "triton"]
CAUSAL_ATTENTION_NDIM = 4


class ForwardMode(Enum):
    """Distinguish between prefill and decode modes for attention behavior."""

    PREFILL = auto()  # SGLang's EXTEND
    DECODE = auto()  # SGLang's DECODE


@dataclass
class ForwardContext:
    """Per-forward-pass context, constructed by Adapter."""

    mode: ForwardMode
    # layer_idx is needed for kv_cache manager
    attn_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor
    ]  # (q, k, v, layer_idx) -> attn_output
    posi_idx: torch.Tensor


class RMSNorm(nn.Module):
    """RMS normalization (no mean subtraction, no bias)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm with learnable weight."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x by its RMS and scale by weight."""
        # x: [batch_size, seq_len, hidden_size]
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings (cos/sin lookup table)."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self, hid_dim: int, theta_base: float = 1_000_000.0, max_seq_len: int = 8192
    ) -> None:
        """Precompute cos/sin tables for all positions up to max_seq_len."""
        super().__init__()
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, hid_dim, 2) / hid_dim))
        t = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freq = torch.outer(t, inv_freq)  # [max_seq_len, hid_dim // 2]
        emb = torch.cat((freq, freq), dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        posi_idx: torch.Tensor,  # [batch, seq_len]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up cos/sin for given position indices."""
        cos = self.cos_cached[posi_idx]  # [batch, seq_len, head_dim]
        sin = self.sin_cached[posi_idx]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [batch_size, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _num_queries_per_kv(num_q_heads: int, num_kv_heads: int) -> int:
    if num_q_heads % num_kv_heads != 0:
        msg = (
            f"num_q_heads ({num_q_heads}) must be "
            f"divisible by num_kv_heads ({num_kv_heads})"
        )
        raise ValueError(msg)
    return num_q_heads // num_kv_heads


def _validate_causal_attention_inputs(
    q: torch.Tensor,  # (batch_size, num_q_heads, seq_len, head_dim)
    k: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
    v: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
) -> None:
    if q.dim() != CAUSAL_ATTENTION_NDIM:
        msg = "q must have shape (batch_size, num_q_heads, seq_len, head_dim)"
        raise ValueError(msg)
    if k.dim() != CAUSAL_ATTENTION_NDIM:
        msg = "k must have shape (batch_size, num_kv_heads, seq_len, head_dim)"
        raise ValueError(msg)
    if v.dim() != CAUSAL_ATTENTION_NDIM:
        msg = "v must have shape (batch_size, num_kv_heads, seq_len, head_dim)"
        raise ValueError(msg)
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        msg = "q, k, and v must agree on batch_size"
        raise ValueError(msg)
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        msg = "q, k, and v must agree on seq_len"
        raise ValueError(msg)
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        msg = "q, k, and v must agree on head_dim"
        raise ValueError(msg)
    if k.shape[1] != v.shape[1]:
        msg = "k and v must agree on num_kv_heads"
        raise ValueError(msg)
    if q.device != k.device or q.device != v.device:
        msg = "q, k, and v must be on the same device"
        raise ValueError(msg)
    _num_queries_per_kv(q.shape[1], k.shape[1])


def causal_attention_torch_ref(
    q: torch.Tensor,  # (batch_size, num_q_heads, seq_len, head_dim)
    k: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
    v: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
) -> torch.Tensor:
    """Compute causal self-attention with GQA support for prefill.

    Precision policy matches ``paged_attention_decode()``:
    keep ``q @ k`` and ``attn @ v`` in the incoming activation dtype, but run
    the softmax logits/probabilities in ``float32`` for numerical stability.
    """
    _validate_causal_attention_inputs(q, k, v)
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads != num_kv_heads:
        repeat_factor = _num_queries_per_kv(num_q_heads, num_kv_heads)
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    seq_len = q.shape[2]
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
    )
    scores = scores + mask
    # Softmax stays in float32, then cast weights back before the value matmul.
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    return torch.einsum("bhij,bhjd->bhid", attn, v)


def causal_attention(
    q: torch.Tensor,  # (batch_size, num_q_heads, seq_len, head_dim)
    k: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
    v: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
    *,
    backend: PrefillAttentionBackend = "auto",
) -> torch.Tensor:
    """Compute prefill causal attention with automatic backend dispatch."""
    _validate_causal_attention_inputs(q, k, v)
    if backend not in {"auto", "torch_ref", "triton"}:
        msg = f"Unsupported causal attention backend: {backend}"
        raise ValueError(msg)
    if backend == "torch_ref":
        return causal_attention_torch_ref(q, k, v)
    if backend == "triton":
        if not triton_causal_attention_supported(q, k, v):
            msg = "Triton causal attention backend is not available for these inputs"
            raise RuntimeError(msg)
        return causal_attention_triton(q, k, v)
    if triton_causal_attention_supported(q, k, v):
        return causal_attention_triton(q, k, v)
    return causal_attention_torch_ref(q, k, v)


class QwenMLP(nn.Module):
    """SwiGLU-based MLP: gate + up projection, SiLU activation, down projection."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Initialize projections for SwiGLU MLP."""
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: down(act(gate(x)) * up(x))."""
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class QwenAttention(nn.Module):
    """Multi-head attention with GQA and rotary position embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        sliding_window_size: int = 0,
    ) -> None:
        """Initialize Q/K/V/O projections for grouped-query attention."""
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.sliding_window_size = sliding_window_size or None

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Project, apply RoPE, delegate to attn_fn, and project output."""
        batch, seq_len = hidden_states.shape[:2]
        q, k, v = self._project_qkv(hidden_states, cos, sin)
        attn_output = ctx.attn_fn(q, k, v, self.layer_idx)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len = hidden_states.shape[:2]
        reshape = (batch, seq_len, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(reshape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(reshape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(reshape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v  # [batch, num_heads, seq_len, head_dim]


class QwenDecoderLayer(nn.Module):
    """Single transformer decoder layer: attention + MLP with residual connections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """Initialize attention, MLP, and layer norms."""
        super().__init__()
        self.self_attn = QwenAttention(
            hidden_size, num_attention_heads, num_key_value_heads, head_dim, layer_idx
        )
        self.mlp = QwenMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Apply pre-norm attention and pre-norm MLP with residual connections."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cos, sin, ctx)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class QwenModel(nn.Module):
    """Full Qwen2.5 transformer: embedding + N decoder layers + final norm."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_seq_size: int,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        """Initialize embeddings, decoder layers, and final norm."""
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = RotaryEmbedding(
            hid_dim=head_dim, max_seq_len=max_seq_size, theta_base=rope_theta
        )
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    layer_idx=idx,
                    rms_norm_eps=rms_norm_eps,
                )
                for idx in range(num_hidden_layers)
            ]
        )
        self.final_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        """Run input_ids through embedding, all decoder layers, and final norm."""
        hidden_states = self.token_embedding(input_ids)
        cos, sin = self.positional_embedding(ctx.posi_idx)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, ctx)
        return self.final_layernorm(hidden_states)
