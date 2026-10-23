"""Train the vibeulator.

    python train.py                          # the shipped configuration
    python train.py --steps 50 --n 20000 --d 64 --layers 2 --heads 2 --ff 256 \
        --device cpu --out /tmp/smoke.pt     # a smoke test

Only the answer is scored: the loss covers the tokens after '=' and the
closing <eos>, never the expression itself.
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from vibeulator.data import build
from vibeulator.gen import sample
from vibeulator.model import GPT, Config, greedy
from vibeulator.vocab import PAD, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=12_000_000, help='training examples to generate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=60_000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--d', type=int, default=Config.d)
    ap.add_argument('--layers', type=int, default=Config.layers)
    ap.add_argument('--heads', type=int, default=Config.heads)
    ap.add_argument('--ff', type=int, default=Config.ff)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=None, help='processes for data generation')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--log', type=int, default=200)
    ap.add_argument('--eval', type=int, default=5000, help='steps between held-out checks')
    ap.add_argument('--out', default='checkpoints/vibeulator.pt')
    ap.add_argument('--cache', default='data', help='where generated datasets are kept')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    t0 = time.time()
    rows, eq = build(a.n, a.seed, a.workers, a.cache)
    print(f'data: {len(rows):,} rows in {time.time() - t0:.0f}s')

    dev = torch.device(a.device)
    rows, eq = torch.from_numpy(rows), torch.from_numpy(eq)   # uint8: 12M rows is 600 MB, not 4.7 GB
    if dev.type == 'cuda':
        rows, eq = rows.to(dev), eq.to(dev)

    cfg = Config(d=a.d, layers=a.layers, heads=a.heads, ff=a.ff)
    model = GPT(cfg).to(dev)
    print(f'model: {model.params():,} parameters, {cfg.dict()}')
    step_fn = torch.compile(model) if a.compile else model

    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() != 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd},
                             {'params': other, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), fused=dev.type == 'cuda')

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    rng = random.Random(f'holdout-{a.seed}')
    held = [sample(rng) for _ in range(2000)]
    held_prompts = [encode(e) for e, _ in held]

    pos = torch.arange(cfg.block, device=dev)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else torch.autocast('cpu', enabled=False)
    gen = torch.Generator(device=rows.device).manual_seed(a.seed)
    seen, t0, run = 0, time.time(), None
    for step in range(a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        i = torch.randint(len(rows), (a.batch,), device=rows.device, generator=gen)
        b, e = rows[i].to(dev).long(), eq[i].to(dev).long()
        x, y = b[:, :-1], b[:, 1:].clone()
        y[(pos[None, :] < e[:, None]) | (y == PAD)] = -100
        with amp:
            logits = step_fn(x)
        loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += a.batch
        run = loss.detach() if run is None else run.mul_(0.98).add_(loss.detach(), alpha=0.02)

        if (step + 1) % a.log == 0:
            dt = time.time() - t0
            print(f'step {step + 1:6d}  loss {run.item():.4f}  lr {lr_at(step):.2e}  '
                  f'{seen / len(rows):.2f} epochs  {(step + 1) / dt:.1f} it/s', flush=True)
        if (step + 1) % a.eval == 0 or step + 1 == a.steps:
            model.eval()
            got = greedy(model, held_prompts)
            model.train()
            acc = sum(g == t for g, (_, t) in zip(got, held)) / len(held)
            print(f'  held-out exact match {acc:.3f}', flush=True)
            save(model, cfg, a, step + 1)


def save(model, cfg, a, step):
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save({'config': cfg.dict(), 'state': model.state_dict(), 'step': step,
                'args': vars(a)}, a.out)


if __name__ == '__main__':
    main()
