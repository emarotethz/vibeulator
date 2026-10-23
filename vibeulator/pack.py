"""The weights file the page downloads.

    "VIBU" | u32 header length | JSON header, space-padded to 4 bytes | tensors

The four big matrices in each block (qkv, proj, fc1, fc2) are stored as int8
with one float32 scale per output row, which takes the file from 10.6 MB of
float32 to about 2.7 MB. Everything else stays float32. The eval runs on the
same rounded weights (dequantized()) so the numbers on the page describe the
model the page actually runs.
"""
import json
import re
import struct

import numpy as np
import torch

from .vocab import TOKENS

QUANTIZED = re.compile(r'blocks\.\d+\.(qkv|proj|fc1|fc2)\.weight$')


def _q8(w):
    w = w.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-12) / 127
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantized(state):
    """A copy of state with every int8-stored matrix replaced by its rounded value."""
    out = {}
    for k, v in state.items():
        if QUANTIZED.search(k):
            q, s = _q8(v)
            out[k] = (q.float() * s[:, None]).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def write(path, cfg, state, meta):
    tensors, blobs, off = [], [], 0

    def put(arr):
        nonlocal off
        b = arr.tobytes()
        pad = (-len(b)) % 4
        blobs.append(b + b'\0' * pad)
        at = off
        off += len(b) + pad
        return at

    for k, v in state.items():
        if QUANTIZED.search(k):
            q, s = _q8(v)
            t = {'name': k, 'shape': list(v.shape), 'dtype': 'i8',
                 'offset': put(q.cpu().numpy()), 'scale': put(s.cpu().numpy().astype('<f4'))}
        else:
            t = {'name': k, 'shape': list(v.shape), 'dtype': 'f32',
                 'offset': put(v.float().cpu().numpy().astype('<f4'))}
        tensors.append(t)

    header = json.dumps({'config': cfg, 'vocab': TOKENS, 'tensors': tensors, 'meta': meta},
                        separators=(',', ':')).encode()
    header += b' ' * ((-len(header)) % 4)
    with open(path, 'wb') as f:
        f.write(b'VIBU' + struct.pack('<I', len(header)) + header)
        for b in blobs:
            f.write(b)


def read(path):
    """Header and float32 state dict, decoded the same way engine.js does it."""
    buf = open(path, 'rb').read()
    assert buf[:4] == b'VIBU'
    n = struct.unpack('<I', buf[4:8])[0]
    head = json.loads(buf[8:8 + n])
    base = 8 + n
    state = {}
    for t in head['tensors']:
        size = int(np.prod(t['shape']))
        if t['dtype'] == 'i8':
            q = np.frombuffer(buf, np.int8, size, base + t['offset']).astype(np.float32)
            s = np.frombuffer(buf, '<f4', t['shape'][0], base + t['scale'])
            a = q.reshape(t['shape']) * s[:, None]
        else:
            a = np.frombuffer(buf, '<f4', size, base + t['offset']).reshape(t['shape'])
        state[t['name']] = torch.from_numpy(a.copy())
    return head, state
