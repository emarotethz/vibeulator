"""Writes tests/calc-fixture.json: expressions and the answer calc.py gives
(null for a syntax error), for tests/parity.mjs to replay through calc.js.

    python -m tests.calc_fixture
"""
import json
import random

from vibeulator.calc import answer, SyntaxErr
from vibeulator.gen import sample

EDGES = [
    '', '1+', '()', '1..2', '.5+.5', '5.', '.', '+1', '1+-2', '3--2', '--3', '2^3^2', '-2^2',
    '(-2)^2', '2^-1', '2^(-1)', '0^0', '0^-1', '(-8)^(1/3)', '(-8)^3', 'r16^2', 'rr16', 'r-4',
    's(180)', 's180', 'c(90)', 's(-30)', 'c(360)', 's(30)', 'c(60)', 's(45)', 'c(-720)', 's(0.5)',
    '1/0', '0/0', '1/3', '2/3', '10^10', '10^9', '9999999999+1', '9999999999.5', '0.1+0.2',
    '12345678905', '-12345678905', '0.00012345678905', '99999999995', '0.0001', '0.00001',
    '0.000099999999995', '9^99', '10^99', '10^100', '-10^100', 'r(-1)', 'r0', '1e5', '((1)',
    '(1))', '1(2)', '2r4', 's', 'r', '^2', '1*', '7/7', '6/4', '1-1', '0.5-0.5', '-0',
    '123456789012345678901234567890', '1.5^2', '16^0.5', '2^0.5', '17*0', '((((((1))))))',
]


def main():
    rng = random.Random('calc-fixture')
    exprs = EDGES + [sample(rng)[0] for _ in range(5000)]
    out = []
    for e in exprs:
        try:
            out.append([e, answer(e)])
        except SyntaxErr:
            out.append([e, None])
    with open('tests/calc-fixture.json', 'w') as f:
        json.dump(out, f)
    print(f'wrote tests/calc-fixture.json: {len(out)} cases')


if __name__ == '__main__':
    main()
