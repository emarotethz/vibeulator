"""Writes tests/calc-fixture.json: expressions and the answer calc.py gives
('S' for a syntax error), for tests/parity.mjs to replay through calc.js.

    python -m tests.calc_fixture
"""
import json
import random

from vibeulator.calc import answer
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
    '2(3+4)', '(1+2)(3+4)', '(2)3', '2(3)(4)', 'r4r4', '3s(30)', 's(30)2', '-2(3)', '2^3(2)',
    '1/2(4)', '.5(2)', '2.(3)', '(1)(2)(3)', '2r9+1', '2(', '2()', '(2)(', 'sc(60)', '2^(3)4',
    't(45)', 't(90)', 't(-90)', 't(135)', 't(180)', 't(89.9)', 'Rs(p/2)', 'Rc(p)', 'Rt(p/4)', 'Rs(1)',
    'Rt(p/2)', 'Rs(30)', 'R2+3', 's(R30)', 'RR', 'l(1000)', 'l(0)', 'l(-5)', 'l(2)', 'l100', 'n(e)',
    'n(1)', 'n(-1)', 'n(10)', 'e^2', 'e^-1', 'e^(1/2)', 'e', 'p', '2p', 'pe', 'p(2)', '2e', 'e2',
    'p^2', 'rp', 'l(e^3)', 'n(e^3)', 'ln', 'l', 'e^', '2l(100)', 'lp', 'Rs(p/6)+Rc(p/3)', 'Rs(2p)',
]


def main():
    rng = random.Random('calc-fixture')
    exprs = EDGES + [sample(rng)[0] for _ in range(6000)]
    out = [[e, answer(e)] for e in exprs]
    with open('tests/calc-fixture.json', 'w') as f:
        json.dump(out, f)
    print(f'wrote tests/calc-fixture.json: {len(out)} cases')


if __name__ == '__main__':
    main()
