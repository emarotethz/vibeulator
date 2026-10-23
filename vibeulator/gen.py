"""Random expressions for training, and the fixed families the eval reports on.

The mix leans on short sums because those are what should come out right.
Everything harder is in there too, so that on a long product or a cosine the
model still answers with something shaped like a number. It just isn't the
right one.
"""
import random

from .calc import answer
from .vocab import MAX_EXPR, MAX_ANS


def nat(rng, d):
    """A natural number with exactly d digits (d = 1 includes 0)."""
    if d == 1:
        return rng.randrange(10)
    return rng.randrange(10 ** (d - 1), 10 ** d)


def dec(rng, maxd=3):
    whole = nat(rng, rng.randint(1, maxd))
    places = rng.randint(1, 3)
    frac = ''.join(str(rng.randrange(10)) for _ in range(places - 1)) + str(rng.randrange(1, 10))
    return f'{whole}.{frac}'


def num(rng, maxd=2, p_dec=0.0):
    if rng.random() < p_dec:
        return dec(rng, maxd)
    return str(nat(rng, rng.randint(1, maxd)))


def wrap(rng, s, p):
    return f'({s})' if rng.random() < p else s


# ---- binary operations on two numbers ----------------------------------------

def binop(rng, op, d1, d2):
    return f'{nat(rng, d1)}{op}{nat(rng, d2)}'


def g_add(rng):
    return binop(rng, '+', rng.randint(1, 6), rng.randint(1, 6))


def g_sub(rng):
    return binop(rng, '-', rng.randint(1, 6), rng.randint(1, 6))


def g_mul(rng):
    return binop(rng, '*', rng.randint(1, 4), rng.randint(1, 4))


def div_exact(rng, dq=None):
    b = rng.randrange(1, 100) if rng.random() < 0.7 else rng.randrange(1, 10)
    q = nat(rng, dq or rng.randint(1, 3))
    return f'{b * q}/{b}'


def div_any(rng):
    return f'{nat(rng, rng.randint(1, 4))}/{rng.randrange(1, 10 ** rng.randint(1, 3))}'


def g_div(rng):
    return div_exact(rng) if rng.random() < 0.5 else div_any(rng)


def g_dec(rng):
    op = rng.choice('+-*')
    a = dec(rng, 3) if rng.random() < 0.8 else str(nat(rng, rng.randint(1, 3)))
    b = dec(rng, 3) if rng.random() < 0.8 else str(nat(rng, rng.randint(1, 3)))
    return a + op + b


# ---- the "scientific" keys ---------------------------------------------------

def g_pow(rng):
    r = rng.random()
    if r < 0.5:
        return f'{rng.randrange(13)}^{rng.randrange(11)}'
    if r < 0.7:
        return f'{rng.randrange(2, 100)}^{rng.randint(2, 4)}'
    if r < 0.85:
        e = rng.randint(1, 4)
        return f'{rng.randrange(1, 21)}^' + (f'(-{e})' if rng.random() < 0.3 else f'-{e}')
    if r < 0.93:
        return f'{dec(rng, 1)}^{rng.randint(2, 4)}'
    return f'{rng.randrange(1, 1000)}^0.5'


def g_sqrt(rng):
    r = rng.random()
    if r < 0.5:
        x = rng.randrange(100) ** 2
    elif r < 0.9:
        x = nat(rng, rng.randint(1, 5))
    else:
        x = dec(rng, 3)
    return 'r' + (f'({x})' if rng.random() < 0.5 else str(x))


NICE = list(range(0, 721, 15))


def g_trig(rng):
    f = rng.choice('sc')
    r = rng.random()
    if r < 0.4:
        a = str(rng.choice(NICE))
    elif r < 0.9:
        a = str(rng.randrange(361))
    elif r < 0.95:
        a = f'-{rng.randrange(1, 361)}'
    else:
        a = dec(rng, 2)
    if a.startswith('-') or rng.random() < 0.6:
        a = f'({a})'
    return f + a


# ---- longer expressions --------------------------------------------------------

def chain(rng, n, maxd=2, ops='++--*/'):
    s = num(rng, maxd)
    for _ in range(n - 1):
        s += rng.choice(ops) + num(rng, maxd)
    return s


def g_chain(rng):
    return chain(rng, rng.randint(3, 5), 2 if rng.random() < 0.8 else 3)


# precedence of what a node renders as: sums 1, products 2, negation 3,
# powers 4, functions and numbers 5
PREC = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 4}


def tree(rng, depth, p_leaf=0.3):
    """A random expression tree, rendered with the parentheses it needs and
    occasionally one it doesn't. Returns (string, precedence)."""
    if depth == 0 or rng.random() < p_leaf:
        return num(rng, 2 if rng.random() < 0.85 else 3, p_dec=0.08), 5
    r = rng.random()
    if r < 0.12:
        f = rng.choice('rsc')
        s, p = tree(rng, depth - 1, p_leaf + 0.2)
        return f + (s if p == 5 and rng.random() < 0.4 else f'({s})'), 5
    if r < 0.17:
        s, p = tree(rng, depth - 1, p_leaf + 0.2)
        return '-' + (s if p >= 4 else f'({s})'), 3
    op = rng.choice('++--**//^') if r < 0.97 else '^'
    pr = PREC[op]
    if op == '^':
        ls, lp = num(rng, 1 if rng.random() < 0.5 else 2), 5
        rs, rp = str(rng.randrange(5)), 5
        if rng.random() < 0.25:
            ls, lp = tree(rng, depth - 1, 0.6)
    else:
        ls, lp = tree(rng, depth - 1, p_leaf)
        rs, rp = tree(rng, depth - 1, p_leaf)
    if lp < pr or (op == '^' and lp <= pr):
        ls = f'({ls})'
    else:
        ls = wrap(rng, ls, 0.05)
    if rp < pr or (rp == pr and op != '^') or (op == '^' and rp < 3):
        rs = f'({rs})'
    else:
        rs = wrap(rng, rs, 0.05)
    return ls + op + rs, pr


def g_tree(rng):
    return tree(rng, rng.randint(2, 3))[0]


def g_neg(rng):
    r = rng.random()
    a, b = num(rng, 3), num(rng, 3)
    if r < 0.4:
        return f'-{a}{rng.choice("+-*/")}{b}'
    if r < 0.7:
        return f'{a}{rng.choice("+-*/")}-{b}'
    return f'-({a}{rng.choice("+-")}{b})'


def g_err(rng):
    r = rng.random()
    a = num(rng, 3)
    if r < 0.5:
        return f'{a}/0'
    if r < 0.7:
        b = num(rng, 2)
        return f'{a}/({b}-{b})'
    if r < 0.9:
        return f'r(-{rng.randrange(1, 1000)})'
    return f'0^-{rng.randint(1, 4)}'


FAMILIES = [
    ('add', g_add, 0.18),
    ('sub', g_sub, 0.13),
    ('mul', g_mul, 0.14),
    ('div', g_div, 0.08),
    ('dec', g_dec, 0.06),
    ('pow', g_pow, 0.07),
    ('sqrt', g_sqrt, 0.06),
    ('trig', g_trig, 0.05),
    ('chain', g_chain, 0.10),
    ('tree', g_tree, 0.09),
    ('neg', g_neg, 0.03),
    ('err', g_err, 0.01),
]


def fits(expr, ans):
    return len(expr) <= MAX_EXPR and len(ans) <= MAX_ANS


def sample(rng):
    """One (expression, answer) pair from the training mix."""
    names, fns, weights = zip(*FAMILIES)
    while True:
        expr = rng.choices(fns, weights)[0](rng)
        ans = answer(expr)
        if fits(expr, ans):
            return expr, ans


def batch(seed, n):
    rng = random.Random(seed)
    return [sample(rng) for _ in range(n)]
