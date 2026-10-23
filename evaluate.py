"""Score the shipped weights family by family and write weights/eval.json,
which the page draws as its "where it breaks" chart.

    python evaluate.py [--bin weights/vibeulator.bin] [--n 1000]

It loads the int8 file, not the checkpoint, so the numbers belong to the
model the browser runs. "close" means within 1% of the right answer, or the
right Error.
"""
import argparse
import json
import random

import torch

from vibeulator.calc import answer, ERR
from vibeulator.gen import binop, chain, dec, div_any, div_exact, g_pow, g_sqrt, g_trig, tree
from vibeulator.model import GPT, Config, greedy
from vibeulator.pack import read
from vibeulator.vocab import encode, MAX_EXPR, MAX_ANS

SUITE = [
    ('7+5', '1-digit + 1-digit', lambda r: binop(r, '+', 1, 1)),
    ('48+37', '2-digit + 2-digit', lambda r: binop(r, '+', 2, 2)),
    ('512+389', '3-digit + 3-digit', lambda r: binop(r, '+', 3, 3)),
    ('604183+275946', '6-digit + 6-digit', lambda r: binop(r, '+', 6, 6)),
    ('83-46', '2-digit − 2-digit', lambda r: binop(r, '-', 2, 2)),
    ('4012-7385', '4-digit − 4-digit', lambda r: binop(r, '-', 4, 4)),
    ('7*8', '1-digit × 1-digit', lambda r: binop(r, '*', 1, 1)),
    ('36*7', '2-digit × 1-digit', lambda r: binop(r, '*', 2, 1)),
    ('47*23', '2-digit × 2-digit', lambda r: binop(r, '*', 2, 2)),
    ('318*642', '3-digit × 3-digit', lambda r: binop(r, '*', 3, 3)),
    ('2718*3141', '4-digit × 4-digit', lambda r: binop(r, '*', 4, 4)),
    ('1302/42', 'exact division', div_exact),
    ('1/7', 'division with decimals', div_any),
    ('1.25+0.75', 'decimals', lambda r: f'{dec(r)}{r.choice("+-*")}{dec(r)}'),
    ('2^10', 'powers', g_pow),
    ('r2', 'square roots', g_sqrt),
    ('s(30)', 'sin and cos', g_trig),
    ('12+3*4', 'three terms', lambda r: chain(r, 3)),
    ('8*4-6/3+5', 'five terms', lambda r: chain(r, 5)),
    ('(3+4)*5', 'parentheses', lambda r: tree(r, 2, 0.1)[0]),
    ('r(9+(3-1)*8)/2', 'nested', lambda r: tree(r, 3, 0.05)[0]),
]


def close(pred, truth):
    if truth == ERR or pred == ERR:
        return pred == truth
    try:
        p, t = float(pred), float(truth)
    except ValueError:
        return False
    return p == t if t == 0 else abs(p - t) <= 0.01 * abs(t)


def cases(make, n, seed):
    rng, out = random.Random(seed), []
    while len(out) < n:
        e = make(rng)
        a = answer(e)
        if len(e) <= MAX_EXPR and len(a) <= MAX_ANS:
            out.append((e, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', default='weights/vibeulator.bin')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--out', default='weights/eval.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    head, state = read(a.bin)
    model = GPT(Config(**head['config']))
    model.load_state_dict(state)
    model.to(a.device).eval()

    rows = []
    for example, label, make in SUITE:
        cs = cases(make, a.n, f'eval-{label}')
        got = greedy(model, [encode(e) for e, _ in cs])
        exact = sum(g == t for g, (_, t) in zip(got, cs)) / len(cs)
        near = sum(close(g, t) for g, (_, t) in zip(got, cs)) / len(cs)
        rows.append({'label': label, 'example': example, 'exact': round(exact, 4),
                     'close': round(near, 4), 'n': len(cs)})
        print(f'{label:24s} exact {exact:6.1%}   close {near:6.1%}   e.g. {cs[0][0]} = {got[0]} ({cs[0][1]})')

    with open(a.out, 'w') as f:
        json.dump({'meta': head['meta'], 'rows': rows}, f, indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
