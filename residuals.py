import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_sinkhorn(logits: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Project positive logits toward a doubly stochastic matrix."""
    z = logits.float()
    z = z - z.amax(dim=(-2, -1), keepdim=True)
    w = torch.exp(z) + eps
    for _ in range(iters):
        w = w / (w.sum(dim=-1, keepdim=True) + eps)
        w = w / (w.sum(dim=-2, keepdim=True) + eps)
    return w.to(logits.dtype)


def matrix_errors(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    row_err = (w.sum(dim=-1) - 1.0).abs().amax()
    col_err = (w.sum(dim=-2) - 1.0).abs().amax()
    return row_err, col_err


class LayerNorm(nn.Module):
    """LayerNorm with optional bias, matching nanoGPT."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias_mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, channels // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class VanillaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.last_aux = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if self.training:
            self.last_aux = {"activation_norm": x.detach().float().norm(dim=-1).mean()}
        return x


@dataclass
class HCAux:
    activation_norm: torch.Tensor
    row_error: float
    col_error: float
    stream_std: torch.Tensor


class DynamicHCUnit(nn.Module):
    """DeepSeek-V4-style HC pre/post mixer adapted to GPT blocks.

    Input streams are [B, T, H, D]. A lightweight learned function emits:
    pre weights [H], post weights [H], and a combination matrix [H, H].
    mHC constrains the combination matrix with Sinkhorn; naive HC uses row-softmax only.
    """

    def __init__(self, config, constrained: bool):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.dim = config.n_embd
        self.norm_eps = config.hc_eps
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.constrained = constrained
        mix_dim = (2 + self.hc_mult) * self.hc_mult
        flat_dim = self.hc_mult * self.dim
        self.fn = nn.Linear(flat_dim, mix_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(3))
        self.base = nn.Parameter(torch.zeros(mix_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.fn.weight)
        with torch.no_grad():
            h = self.hc_mult
            self.base.zero_()
            post_start = h
            comb_start = 2 * h
            self.base[post_start : post_start + h].fill_(4.0)
            eye = torch.eye(h).reshape(-1)
            self.base[comb_start : comb_start + h * h].copy_(eye * 4.0)

    def split(self, mixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.hc_mult
        pre_logits = mixes[..., :h] * self.scale[0]
        post_logits = mixes[..., h : 2 * h] * self.scale[1]
        comb_logits = mixes[..., 2 * h :] * self.scale[2]
        pre_logits = pre_logits + self.base[:h].to(dtype=mixes.dtype, device=mixes.device)
        post_logits = post_logits + self.base[h : 2 * h].to(dtype=mixes.dtype, device=mixes.device)
        comb_logits = comb_logits + self.base[2 * h :].to(dtype=mixes.dtype, device=mixes.device)
        comb_logits = comb_logits.view(*mixes.shape[:-1], h, h)

        pre = F.softmax(pre_logits.float(), dim=-1).to(mixes.dtype)
        post = torch.sigmoid(post_logits.float()).to(mixes.dtype)
        if self.constrained:
            comb = stable_sinkhorn(comb_logits, self.sinkhorn_iters, self.norm_eps)
        else:
            comb = F.softmax(comb_logits.float(), dim=-1).to(mixes.dtype)
        return pre, post, comb

    def forward(self, residual: torch.Tensor, update: Optional[torch.Tensor] = None):
        shape = residual.shape
        flat = residual.flatten(2).float()
        rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        mixes = self.fn(flat).float() * rms
        pre, post, comb = self.split(mixes)
        x = torch.sum(pre.unsqueeze(-1) * residual, dim=2)
        if update is None:
            return x.to(residual.dtype), post, comb
        y = post.unsqueeze(-1) * update.unsqueeze(2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(2), dim=3)
        return y.to(residual.dtype)


class HyperConnectionBlock(nn.Module):
    def __init__(self, config, constrained: bool):
        super().__init__()
        self.constrained = constrained
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.hc_attn = DynamicHCUnit(config, constrained=constrained)
        self.hc_mlp = DynamicHCUnit(config, constrained=constrained)
        self.last_aux = {}

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        residual = streams
        x, post, comb = self.hc_attn(residual)
        update = self.attn(self.ln_1(x))
        streams = self.hc_attn(residual, update)

        residual = streams
        x, post, comb = self.hc_mlp(residual)
        update = self.mlp(self.ln_2(x))
        streams = self.hc_mlp(residual, update)

        if self.training:
            row_err, col_err = matrix_errors(comb.detach())
            self.last_aux = {
                "activation_norm": streams.detach().float().norm(dim=-1).mean(),
                "stream_std": streams.detach().float().std(dim=2).mean(),
                "mixer_row_error": row_err,
                "mixer_col_error": col_err,
            }
        return streams


def build_block(config):
    if config.residual_mode == "vanilla":
        return VanillaBlock(config)
    if config.residual_mode == "hc":
        return HyperConnectionBlock(config, constrained=False)
    if config.residual_mode == "mhc":
        return HyperConnectionBlock(config, constrained=True)
    raise ValueError(f"unknown residual_mode: {config.residual_mode}")
