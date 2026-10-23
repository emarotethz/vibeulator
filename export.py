"""Checkpoint -> weights/vibeulator.bin, plus tests/model-fixture.json, which
holds the logits PyTorch gets from the same int8 weights so tests/parity.mjs
can check that engine.js computes the same thing.

    python export.py [--ckpt checkpoints/vibeulator.pt] [--out weights/vibeulator.bin]
"""
import argparse
import json

import torch

from vibeulator.model import GPT, Config
from vibeulator.pack import read, write
from vibeulator.vocab import EOS, MAX_ANS, MAX_EXPR, decode, encode

PROBES = ['7+5', '48+37', '512+389', '604183+275946', '83-46', '7*8', '47*23', '318*642',
          '1302/42', '1/7', '2^10', 'r2', 's(30)', 'c(45)', '12+3*4', '(3+4)*5', '1/0',
          'r(-4)', '-5+2', '0.1+0.2', '2^-3', '9^99', 'r(9+(3-1)*8)/2', '((((((1))))))',
          '1+', '++', '123456789012345678901234567890']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/vibeulator.pt')
    ap.add_argument('--out', default='weights/vibeulator.bin')
    ap.add_argument('--fixture', default='tests/model-fixture.json')
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu')
    cfg, args = ck['config'], ck['args']
    model = GPT(Config(**cfg))
    model.load_state_dict(ck['state'])
    meta = {'params': model.params(), 'steps': ck['step'], 'batch': args['batch'],
            'examples': args['n'], 'seen': ck['step'] * args['batch'],
            'max_expr': MAX_EXPR, 'max_ans': MAX_ANS}
    write(a.out, cfg, {k: v for k, v in ck['state'].items()}, meta)
    print(f'wrote {a.out}: {model.params():,} parameters')

    # Greedy-decode the probes on the weights as stored, one token at a time,
    # and keep every step's logits.
    _, state = read(a.out)
    model.load_state_dict(state)
    model.eval()
    probes = []
    with torch.no_grad():
        for p in PROBES:
            ids, steps = encode(p), []
            while len(steps) <= MAX_ANS and len(ids) <= cfg['block']:
                logits = model(torch.tensor([ids]))[0, -1]
                nxt = int(logits.argmax())
                steps.append([round(float(v), 5) for v in logits])
                ids.append(nxt)
                if nxt == EOS:
                    break
            probes.append({'expr': p, 'answer': decode(ids[len(encode(p)):]), 'logits': steps})
    with open(a.fixture, 'w') as f:
        json.dump({'probes': probes}, f)
    print(f'wrote {a.fixture}')


if __name__ == '__main__':
    main()
