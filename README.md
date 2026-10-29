# vibeulator

A calculator where a small transformer guesses every answer.

Try it at [ethanmarot.com/vibeulator](https://ethanmarot.com/vibeulator/).

## Install

    pip install -r requirements.txt

## Run

    python train.py         # train the model (about 40 min on one GPU)
    python export.py        # write weights/vibeulator.bin
    python evaluate.py      # write weights/eval.json
    node tests/parity.mjs   # check the browser code against PyTorch
