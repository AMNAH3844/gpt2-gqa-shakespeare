"""GPT-2 style model with grouped-query attention (GQA).

Contents:
    GPTConfig            settings in one place
    CausalSelfAttention  attention with n_head query heads sharing n_kv_head key/value heads
    MLP                  per-token feed-forward layer
    Block                attention + MLP with layer norm and residual connections
    GPT                  embeddings + blocks + output layer, with the loss

Other files use it like this:
    from model import GPT, GPTConfig

Run `python model.py` to check that everything works (shapes, causality, parameter count, start loss).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256   # context length (max tokens the model sees at once)
    vocab_size: int = 512   # size of our BPE vocabulary
    n_layer: int = 6        # number of transformer blocks
    n_head: int = 6         # number of query heads
    n_kv_head: int = 2      # number of key/value heads (GQA); n_head // n_kv_head queries share each one
    n_embd: int = 384       # embedding size


class CausalSelfAttention(nn.Module):
    """Grouped-query attention.

    Standard multi-head attention has n_head query, key and value heads.
    Here there are still n_head query heads, but only n_kv_head key/value heads,
    so every group of n_head // n_kv_head query heads shares one key/value head.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head  # queries per key/value head

        # queries: one per head. keys and values: only n_kv_head of them (the GQA saving)
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        # lower-triangular mask: token i may only look at tokens 0..i
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape

        # split into heads: (B, T, heads * head_dim) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # repeat each key/value head so it lines up with its group of query heads:
        # [kv0, kv1] -> [kv0, kv0, kv0, kv1, kv1, kv1]
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # attention scores: (B, n_head, T, T)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (B, n_head, T, head_dim)

        # glue the heads back together: (B, T, n_embd), then mix them
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    """Per-token feed-forward layer: widen 4x, GELU, shrink back."""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Transformer block: pre-norm attention and MLP, each added back through a residual connection."""

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embedding
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # position embedding
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),                  # final layer norm
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: the token embedding and the output layer share one matrix
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 initialization: small random weights, and the layers that write into the
        # residual stream are scaled down by 1/sqrt(2 * n_layer)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence is longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Self-test. Expected values are for the default config.
    torch.manual_seed(1337)
    cfg = GPTConfig()
    print(cfg)

    # 1. attention keeps the shape, and is causal
    attn = CausalSelfAttention(cfg)
    x = torch.randn(4, 8, cfg.n_embd)
    y = attn(x)
    print("attention in/out:", tuple(x.shape), tuple(y.shape))
    assert y.shape == x.shape

    with torch.no_grad():
        x2 = x.clone()
        x2[:, -1, :] += 1.0  # change only the last token
        y1, y2 = attn(x), attn(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6), "mask is leaking future tokens"
    assert not torch.allclose(y1[:, -1], y2[:, -1]), "last token should change"
    print("causal mask: OK (earlier tokens do not see the last token)")

    # 2. GQA saves parameters compared with standard multi-head attention
    mha_params = 4 * cfg.n_embd * cfg.n_embd
    print("attention params:", count_parameters(attn), "| standard multi-head would be:", mha_params)
    assert count_parameters(attn) == 393_216

    # 3. full model
    model = GPT(cfg)
    n_params = count_parameters(model)
    print("total parameters:", n_params)
    assert n_params == 9_742_080
    assert model.lm_head.weight is model.transformer.wte.weight
    print("weights tied: OK")

    idx = torch.randint(0, cfg.vocab_size, (4, 8))
    targets = torch.randint(0, cfg.vocab_size, (4, 8))
    logits, loss = model(idx, targets)
    print("logits:", tuple(logits.shape))
    assert logits.shape == (4, 8, cfg.vocab_size)
    print("start loss:", round(loss.item(), 3), "| expected about ln(vocab) =", round(math.log(cfg.vocab_size), 3))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5

    print("all checks passed")
