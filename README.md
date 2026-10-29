# vibeulator

The world's worst and most wasteful calculator. Every key is a token, and a
2.7M-parameter transformer reads the sum and writes the answer one token at a
time, the way a language model writes a sentence. 2 + 2 is probably 4. Short
sums come out right; long ones, big products and anything with a function in
them come out as guesses, usually about the right size and just as confident.

It runs in the browser at [ethanmarot.com/vibeulator](https://ethanmarot.com/vibeulator/).
This repo has everything behind it: the data, the training, the export, the eval
and the JavaScript that runs the model on the page.

## How it works

**Tokens.** 33 of them: the digits, `. + - * / ^ ( )`, `r` for √, `s` `c` `t` for
sin, cos and tan, `l` `n` for log and ln, `p` for π, `e` for the constant e (and
the exponent in answers like `1.5e12`), `=`, `E` for Error, `S` for Syntax error,
`R` in front of a sum whose angles are in radians, and `<pad> <bos> <eos>`.
A training example is

    <bos> 4 8 + 3 7 = 5 8 <eos>

with the answer written backwards, least significant digit first, the way you
would work it out on paper: each digit then only needs the carries from the
ones already written. The page flips it back. The loss covers only the answer
and the closing `<eos>`.

**Ground truth.** `vibeulator/calc.py` defines the grammar (usual precedence, `^`
right-associative, `-2^2 = -4`, angles in degrees unless the sum starts with
`R`) and the display format: 10 significant digits, fixed notation from 1e-4 up
to 1e10 and scientific notation outside that, `E` for division by zero, roots
and logs of negatives, tan at a right angle and overflow past 1e100, and `S` for
anything that doesn't parse. Every expression has exactly one right string. Two things side by
side multiply, so `2(3+4)`, `(1+2)(3+4)` and `2r9` work; the training data always
writes the `*`, so the page writes it in before the model sees the sum.

**Data.** `vibeulator/gen.py` generates random expressions from a fixed mix of
families: sums and differences, products, division, decimals, powers, roots,
trig in degrees and radians, logs, e^x, π and e, chains of terms, random
bracketed trees, long expressions spread evenly from 14 to 60 tokens, products
and powers that keep growing past 10^10 and on to Error, and about 3% that don't
parse (a valid sum with a slip in it, written exactly as the page would send it).
Numbers are picked by size first, 1 to 15 digits with small ones most common: a
model only gets big numbers the right size if it has seen them, and one trained
up to 4×4-digit products wrote every 5×5 one 100 times too small. The default
run generates 30M examples, which takes under a minute across all cores and is
cached under `data/`.

**Model.** `vibeulator/model.py` is a plain GPT: 6 pre-LayerNorm blocks, width
192, 6 heads, learned positions, tanh GELU and output weights tied to the
embedding. The context is 78 tokens: up to 60 for the sum and 16 for the answer.

**Export.** `vibeulator/pack.py` writes `weights/vibeulator.bin`: a JSON header
followed by the tensors, with the attention and MLP matrices in int8 (one scale
per row), which brings it to about 2.8 MB. The header also records that the
model writes its answers backwards.

**Eval.** `evaluate.py` runs on those int8 weights and samples every answer at
temperature 1, the way the page asks, so its numbers describe what the page
does: a thousand random cases of each kind of sum (how many come out exact and
how many within 1%), and 120 random sums at every length from 1 to 60 tokens
(the median error, a breakdown and real examples). It writes
`weights/eval.json`, which the page draws.

**In the browser.** `web/engine.js` reimplements the forward pass by hand with a
key/value cache and no dependencies. `web/calc.js` is a port of `calc.py` that the
page uses for the real answer. `tests/parity.mjs` checks both against Python:
the calculator has to agree on every fixture case, and the engine has to
reproduce PyTorch's logits step by step.

## Running it

    pip install -r requirements.txt

    python train.py                 # 100k steps of 1024 on 30M examples, writes checkpoints/vibeulator.pt
    python export.py                # -> weights/vibeulator.bin, tests/model-fixture.json
    python evaluate.py              # -> weights/eval.json
    node tests/parity.mjs           # engine.js and calc.js against PyTorch and calc.py

That is the shipped model; it trains in about 40 minutes on one RTX 5090.
`--forward` writes answers the usual way round instead: it gets big products
the right size just as well but loses more on small sums. `python train.py
--help` lists the rest of the knobs (model size, steps, batch, data size).

**On a CPU.** The model is small enough to train without a GPU. Smaller batches
and bf16 autocast make most of the difference: on a 32-core Zen 5 Threadripper a
batch of 256 in bf16 runs at about 4.2 steps/s, so the full 100k steps take
about 6.5 hours and see a quarter of the examples of the GPU run:

    nohup python train.py --device cpu --batch 256 > train.log 2>&1 &

`--threads` caps how many cores it takes (default: one per physical core), and
`--fp32` turns bf16 off on CPUs that don't support it natively, where it would
be slower.

A smoke test that finishes in under a minute on a CPU:

    python train.py --n 100000 --steps 300 --batch 128 --d 64 --layers 2 --heads 2 \
        --ff 256 --device cpu --eval 150 --out /tmp/smoke.pt --cache /tmp

`tests/calc-fixture.json` is generated by `python -m tests.calc_fixture`. Rerun it
after any change to `calc.py` or `gen.py`.

## Putting it on the site

The page lives in the website repo under `vibeulator/`. After training:

    cp weights/vibeulator.bin weights/eval.json web/engine.js web/calc.js \
       ../personnal_website/vibeulator/

The page reads from the weights' header how long a sum can be and whether the
answers come out backwards. Until `vibeulator.bin` is there it still works as a
plain calculator and says it has no weights yet. If `engine.js` or `calc.js`
change, bump their `?v=` in the page's `index.html`, since the site caches its
scripts forever.
