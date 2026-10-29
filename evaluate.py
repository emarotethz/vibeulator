"""Score the shipped weights and write weights/eval.json, which the page draws
as its "error by operations" table and "error by length" chart.

    python evaluate.py [--bin weights/vibeulator.bin] [--n 10000] [--per-length 10000] [--temp 1]

It loads the int8 file, not the checkpoint, so the numbers belong to the
model the browser runs, and it samples every token at temperature 1 the way
the page does (--temp 0 for greedy). "close" means within 1%, or the right
Error.

Errors are a share of the bigger of the two numbers, |said - right| / max(|said|,
|right|), so being too big and being too small count the same: 10 times off
either way is 90%, and 0, the wrong sign or something wildly too big are all
100%. Measured against the right answer alone, 0 for 100 would be 100% off
while 1e11 was 1e11% off, though both are as wrong as it gets.

It also scores random expressions at every length, the same number at each,
for the page's error-by-length chart: every kind of operation equally likely,
and only ones with a numeric answer, so each has a % error. Each length gets
a breakdown of how far off its answers are and two real examples, a typical
one and the worst.
"""
import argparse
import json
import math
import random
import re

import torch

from vibeulator.calc import answer, ERR, SYN
from vibeulator.gen import (FAMILIES, binop, nat, chain, dec, div_any, div_exact, fits, g_const, g_exp, g_log, g_pow,
                            g_sqrt, g_syntax, g_trig, moded, tree)
from vibeulator.model import GPT, Config, answers
from vibeulator.pack import read
from vibeulator.vocab import encode, MAX_EXPR, MAX_ANS

ADD, MUL, MORE = 'Adding and subtracting', 'Multiplying and dividing', 'Everything else'
# bigger than anything the first models were trained on: there the pale "within 1%" is what counts
HUGE = lambda r: '*'.join(str(nat(r, r.randint(4, 12))) for _ in range(r.randint(3, 6)))
SUITE = [
    ('7+5', '1 + 1 digits', ADD, lambda r: binop(r, '+', 1, 1)),
    ('48+37', '2 + 2 digits', ADD, lambda r: binop(r, '+', 2, 2)),
    ('512+389', '3 + 3 digits', ADD, lambda r: binop(r, '+', 3, 3)),
    ('604183+275946', '6 + 6 digits', ADD, lambda r: binop(r, '+', 6, 6)),
    ('81103553+68231808', '8 + 8 digits', ADD, lambda r: binop(r, '+', 8, 8)),
    ('83-46', '2 − 2 digits', ADD, lambda r: binop(r, '-', 2, 2)),
    ('4012-7385', '4 − 4 digits', ADD, lambda r: binop(r, '-', 4, 4)),
    ('7*8', '1 × 1 digits', MUL, lambda r: binop(r, '*', 1, 1)),
    ('36*7', '2 × 1 digits', MUL, lambda r: binop(r, '*', 2, 1)),
    ('47*23', '2 × 2 digits', MUL, lambda r: binop(r, '*', 2, 2)),
    ('318*642', '3 × 3 digits', MUL, lambda r: binop(r, '*', 3, 3)),
    ('2718*3141', '4 × 4 digits', MUL, lambda r: binop(r, '*', 4, 4)),
    ('693031*553681', '6 × 6 digits', MUL, lambda r: binop(r, '*', 6, 6)),
    ('13315664*33515252', '8 × 8 digits', MUL, lambda r: binop(r, '*', 8, 8)),
    ('3045*7428*6628*1030', 'huge products', MUL, HUGE),
    ('1302/42', 'exact division', MUL, div_exact),
    ('1/7', 'division with decimals', MUL, div_any),
    ('1.25+0.75', 'decimals', MORE, lambda r: f'{dec(r)}{r.choice("+-*")}{dec(r)}'),
    ('2^10', 'powers', MORE, g_pow),
    ('r2', 'square roots', MORE, g_sqrt),
    ('s(30)', 'trig in degrees', MORE, lambda r: g_trig(r, False)),
    ('Rs(p/6)', 'trig in radians', MORE, lambda r: g_trig(r, True)),
    ('l(1000)', 'logarithms', MORE, g_log),
    ('e^(2)', 'e to the x', MORE, g_exp),
    ('2*p', 'π and e', MORE, g_const),
    ('12+3*4', 'three terms', MORE, lambda r: chain(r, 3)),
    ('8*4-6/3+5', 'five terms', MORE, lambda r: chain(r, 5)),
    ('(3+4)*5', 'parentheses', MORE, lambda r: tree(r, 2, 0.1)[0]),
    ('r(9+(3-1)*8)/2', 'nested', MORE, lambda r: tree(r, 3, 0.05)[0]),
    ('2+*3', 'syntax errors', MORE, g_syntax),
]


def close(pred, truth):
    if truth in (ERR, SYN) or pred in (ERR, SYN):
        return pred == truth
    off = off_by(pred, truth)
    return off is not None and off <= 1


NUMBER = re.compile(r'^-?\d+(\.\d+)?(e-?\d+)?$')


def off_by(pred, truth):
    """How far off an answer is, in percent of the bigger of the two, the way the
    page's verdict reads it: 0 when exact, 100 at most, None when it isn't a
    number that can be compared."""
    if pred == truth:
        return 0.0
    if truth in (ERR, SYN) or pred in (ERR, SYN) or not NUMBER.match(pred):
        return None
    p, t = float(pred), float(truth)
    if not (math.isfinite(p) and math.isfinite(t)):
        return None
    # a wrong sign comes out above 100%: it is as wrong as 0
    return 0.0 if p == t else min(100.0, abs(p - t) / max(abs(p), abs(t)) * 100)


def any_op(rng):
    """A random valid expression: each kind of operation in the training mix
    equally likely, rather than weighted towards sums, and never one whose
    answer is Error."""
    makers = [make for name, make, _ in FAMILIES if name not in ('err', 'syntax')]
    while True:
        e = moded(rng, rng.choice(makers)(rng))
        a = answer(e)
        if a != ERR and fits(e, a):
            return e, a


def ask(model, prompts, temp, seed, rev):
    """The model's answers the right way round, for a model that writes them backwards too."""
    got = answers(model, prompts, temp, seed=seed)
    return [g[::-1] for g in got] if rev else got


def by_length(model, per, temp, most, rev, seed='eval-length'):
    """`per` random expressions at every length from 1 to `most` tokens. Returns
    a summary row per length."""
    rng, got = random.Random(seed), {n: [] for n in range(1, most + 1)}
    while any(len(v) < per for v in got.values()):
        e, t = any_op(rng)
        if len(e) <= most and len(got[len(e)]) < per:
            got[len(e)].append((e, t))
    cs = [c for n in sorted(got) for c in got[n]]
    ans = ask(model, [encode(e) for e, _ in cs], temp, len(SUITE), rev)
    offs = [off_by(g, t) for g, (_, t) in zip(ans, cs)]

    tidy = lambda x: None if x is None or math.isinf(x) else float(f'{x:.3g}')
    rank = lambda o: math.inf if o is None else o
    rows = []
    for n in sorted(got):
        # every case of this length, best answer first
        here = sorted((rank(o), e, g, t) for (e, t), g, o in zip(cs, ans, offs) if len(e) == n)
        at = lambda q: here[min(len(here) - 1, int(q * len(here)))]
        share = lambda lo, hi: round(sum(lo < h[0] <= hi for h in here) / len(here), 4)
        example = lambda h: {'expr': h[1], 'said': h[2], 'right': h[3], 'off': tidy(h[0])}
        # the examples are answers that were numbers: "said Error, right is 12.5" shows nothing about how far off
        numbers = [h for h in here if not math.isinf(h[0])] or here
        typical = at(0.5) if not math.isinf(at(0.5)[0]) else numbers[len(numbers) // 2]
        rows.append({'tokens': n, 'n': len(here), 'exact': share(-1, 0),
                     'median': tidy(at(0.5)[0]), 'p90': tidy(at(0.9)[0]),
                     # exact, within 1%, 1 to 10%, 10 to 90%, more than 10 times off (90 to 100%),
                     # and not a number at all (Error, Syntax error or garbage), which has no % to give
                     'split': [share(-1, 0), share(0, 1), share(1, 10), share(10, 90), share(90, 100),
                               round(sum(math.isinf(h[0]) for h in here) / len(here), 4)],
                     'typical': example(typical), 'worst': example(numbers[-1])})
        print(f'{n:2d} tokens   exact {rows[-1]["exact"]:6.1%}   median off {at(0.5)[0]:9.3g}%   '
              f'90th percentile {at(0.9)[0]:9.3g}%   e.g. {at(0.5)[1]} = {at(0.5)[2]} ({at(0.5)[3]})')
    return {'rows': rows}


def cases(make, n, seed, most=MAX_EXPR):
    rng, out = random.Random(seed), []
    while len(out) < n:
        e = make(rng)
        a = answer(e)
        if len(e) <= most and len(a) <= MAX_ANS:
            out.append((e, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', default='weights/vibeulator.bin')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--per-length', type=int, default=10000, help='sums at each expression length')
    ap.add_argument('--temp', type=float, default=1.0, help='sampling temperature, 0 for greedy')
    ap.add_argument('--out', default='weights/eval.json')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    head, state = read(a.bin)
    most = head['config']['block'] - MAX_ANS - 2        # the longest sum this model can read
    rev = head['meta'].get('reversed', False)
    model = GPT(Config(**head['config']))
    model.load_state_dict(state)
    model.to(a.device).eval()

    rows = []
    for k, (example, label, group, make) in enumerate(SUITE):
        cs = cases(make, a.n, f'eval-{label}', most)
        got = ask(model, [encode(e) for e, _ in cs], a.temp, k, rev)
        exact = sum(g == t for g, (_, t) in zip(got, cs)) / len(cs)
        near = sum(close(g, t) for g, (_, t) in zip(got, cs)) / len(cs)
        # the page hands out a random one of these when someone clicks the row
        rows.append({'label': label, 'group': group, 'example': example, 'exact': round(exact, 4),
                     'close': round(near, 4), 'n': len(cs), 'pool': [e for e, _ in cs[:40]]})
        print(f'{label:24s} exact {exact:6.1%}   close {near:6.1%}   e.g. {cs[0][0]} = {got[0]} ({cs[0][1]})')

    length = by_length(model, a.per_length, a.temp, most, rev)

    with open(a.out, 'w') as f:
        json.dump({'meta': head['meta'], 'temp': a.temp, 'rows': rows, 'by_length': length}, f, separators=(',', ':'))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
