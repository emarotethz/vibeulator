"""A small GPT. Pre-LayerNorm blocks, learned positions, tanh GELU, and output
weights tied to the token embedding. All of it is deliberately plain, because
web/engine.js has to reproduce it by hand."""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import TOKENS, BLOCK


@dataclass
class Config:
    vocab: int = len(TOKENS)
    block: int = BLOCK
    d: int = 192
    layers: int = 6
    heads: int = 6
    ff: int = 768

    def dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc1 = nn.Linear(c.d, c.ff)
        self.fc2 = nn.Linear(c.ff, c.d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate='tanh'))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.block, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / (2 * c.layers) ** 0.5)
            nn.init.normal_(b.fc2.weight, std=0.02 / (2 * c.layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.tok.weight.T

    def params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def greedy(model, prompts, max_new=None):
    """Greedy answers for a list of encoded prompts (each ending in '=')."""
    return answers(model, prompts, 0.0, max_new=max_new)


@torch.no_grad()
def answers(model, prompts, temp=0.0, seed=0, max_new=None):
    """Answers for a list of encoded prompts (each ending in '='): greedy at temp
    0, otherwise each token sampled at that temperature, the way the page asks.
    Prompts are grouped by length so no padding ever shifts a position."""
    from .vocab import EOS, MAX_ANS, decode
    max_new = max_new or MAX_ANS + 1
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = [None] * len(prompts)
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    for L, idx in by_len.items():
        for s in range(0, len(idx), 4096):
            chunk = idx[s:s + 4096]
            x = torch.tensor([prompts[i] for i in chunk], device=dev)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=dev)
            for _ in range(max_new):
                if x.shape[1] > model.c.block:
                    break
                logits = model(x)[:, -1].float()
                if temp:
                    nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)[:, 0]
                else:
                    nxt = logits.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                x = torch.cat([x, nxt[:, None]], 1)
                done |= nxt == EOS
                if done.all():
                    break
            for j, i in enumerate(chunk):
                ids = x[j, L:].tolist()
                if EOS in ids:
                    ids = ids[:ids.index(EOS)]
                out[i] = decode(ids)
    return out
