"""Builds the training set as one uint8 array of token rows, generated in
parallel and cached under data/ so a second run with the same size and seed
starts straight away."""
import os
from multiprocessing import Pool

import numpy as np

from .gen import VERSION, batch
from .vocab import BLOCK, PAD, encode

CHUNK = 50_000


def _chunk(args):
    seed, n, reverse = args
    rows = np.full((n, BLOCK + 1), PAD, dtype=np.uint8)
    eq = np.empty(n, dtype=np.uint8)
    for i, (expr, ans) in enumerate(batch(seed, n)):
        ids = encode(expr, ans[::-1] if reverse else ans)
        rows[i, :len(ids)] = ids
        eq[i] = len(expr) + 1
    return rows, eq


def build(n, seed, workers=None, cache='data', reverse=False):
    """n rows of [bos, expression, '=', answer, eos, pad...] and, per row, the
    index of its '=' (the first position whose prediction is scored). With
    reverse the answer is written backwards, least significant digit first."""
    path = os.path.join(cache, f'train-v{VERSION}{"-rev" if reverse else ""}-{n}-{seed}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['rows'], z['eq']
    jobs = [(seed * 1_000_003 + i, min(CHUNK, n - s), reverse) for i, s in enumerate(range(0, n, CHUNK))]
    with Pool(workers or max(1, os.cpu_count() // 2)) as pool:
        parts = pool.map(_chunk, jobs)
    rows = np.concatenate([p[0] for p in parts])
    eq = np.concatenate([p[1] for p in parts])
    os.makedirs(cache, exist_ok=True)
    np.savez(path, rows=rows, eq=eq)
    return rows, eq
