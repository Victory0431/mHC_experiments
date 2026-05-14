import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from residuals import build_block


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    residual_mode: str = "vanilla"  # vanilla | hc | mhc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 10
    hc_eps: float = 1e-6


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([build_block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        print(f"num decayed parameter tensors: {len(decay_params)}")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        return flops_achieved / (1979e12)

    def forward(self, idx, targets=None, return_aux: bool = False):
        device = idx.device
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.block_size
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        aux = {}
        if self.config.residual_mode == "vanilla":
            for i, block in enumerate(self.transformer.h):
                x = block(x)
                if self.training and hasattr(block, "last_aux"):
                    aux[f"layer_{i}_activation_norm"] = block.last_aux.get("activation_norm")
        else:
            streams = x.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
            for i, block in enumerate(self.transformer.h):
                streams = block(streams)
                if self.training and hasattr(block, "last_aux"):
                    for k, v in block.last_aux.items():
                        aux.setdefault(k, []).append(v)
            x = streams.mean(dim=2)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :] if targets is None else x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if return_aux:
            return logits, loss, aux
        return logits, loss

