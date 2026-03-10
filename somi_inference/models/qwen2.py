"""Qwen 2.5 base components: RMSNorm, RotaryEmbedding, causal_attention."""

import torch
from torch import nn


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
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

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


def causal_attention(
    q: torch.Tensor,  # (batch_size, num_q_heads, seq_len, head_dim)
    k: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
    v: torch.Tensor,  # (batch_size, num_kv_heads, seq_len, head_dim)
) -> torch.Tensor:
    """Compute causal self-attention with GQA support for prefill."""
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        msg = (
            f"num_q_heads ({num_q_heads}) must be "
            f"divisible by num_kv_heads ({num_kv_heads})"
        )
        raise ValueError(msg)
    if num_q_heads != num_kv_heads:
        repeat_fractor = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeat_fractor, dim=1)
        v = v.repeat_interleave(repeat_fractor, dim=1)

    seq_len = q.shape[2]
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
    )
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum("bhij,bhjd->bhid", attn, v)
