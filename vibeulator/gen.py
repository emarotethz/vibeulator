"""Random expressions for training, and the fixed families the eval reports on.

The mix leans on short sums because those are what should come out right.
Everything harder is in there too, so that on a long product or a cosine the
model still answers with something shaped like a number. It just isn't the
right one. Numbers are picked by size first, up to 15 digits, so that even
when it is wrong it is about the right size.
"""
import random

from .calc import RAD, SYN, answer, close, explicit
from .vocab import MAX_EXPR, MAX_ANS

# bump when the mix or the vocabulary changes, so data/ doesn't serve a stale set
VERSION = 8


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


# Training picks each number's size first, from 1 to 15 digits, small ones most
# often. Left to each operation, big numbers barely came up (7 digits or more in
# 0.2% of sums), and the model can't stretch past what it has seen: trained up to
# 4x4-digit products, it wrote every 5x5 one as if it were 4x4, 100 times too
# small. big=True turns this on; the eval's families keep their fixed sizes.
SIZES = list(range(1, 16))
SIZE_WEIGHTS = [16, 16, 15, 14, 12, 11, 9, 8, 6, 5, 4, 3, 3, 2, 2]


def size(rng, most=15):
    """How many digits a number gets, up to `most`."""
    return rng.choices(SIZES[:most], SIZE_WEIGHTS[:most])[0]


def num(rng, maxd=2, p_dec=0.0, big=False):
    if rng.random() < p_dec:
        return dec(rng, maxd)
    return str(nat(rng, size(rng, 6) if big else rng.randint(1, maxd)))


def wrap(rng, s, p):
    return f'({s})' if rng.random() < p else s


# ---- binary operations on two numbers ----------------------------------------

def binop(rng, op, d1, d2):
    return f'{nat(rng, d1)}{op}{nat(rng, d2)}'


def g_add(rng):
    return binop(rng, '+', size(rng), size(rng))


def g_sub(rng):
    return binop(rng, '-', size(rng), size(rng))


def g_mul(rng):
    return binop(rng, '*', size(rng), size(rng))


def g_huge(rng):
    """Products and powers that keep growing: past 10^10 into scientific
    notation and on past 10^100, where the calculator says Error, so the model
    learns how fast they blow up and where they stop."""
    r = rng.random()
    if r < 0.55:
        return '*'.join(str(nat(rng, size(rng))) for _ in range(rng.randint(3, 8)))
    if r < 0.85:
        return f'{nat(rng, size(rng, 8))}^{rng.randint(2, 50)}'
    return f'{nat(rng, size(rng, 8))}^{rng.randint(2, 6)}*{nat(rng, size(rng))}'


def div_exact(rng, dq=None):
    b = rng.randrange(1, 100) if rng.random() < 0.7 else rng.randrange(1, 10)
    q = nat(rng, dq or rng.randint(1, 3))
    return f'{b * q}/{b}'


def div_any(rng):
    return f'{nat(rng, rng.randint(1, 4))}/{rng.randrange(1, 10 ** rng.randint(1, 3))}'


def g_div(rng):
    if rng.random() < 0.5:
        return div_exact(rng, size(rng, 6))
    return f'{nat(rng, size(rng))}/{rng.randrange(1, 10 ** size(rng, 4))}'


def g_dec(rng):
    op = rng.choice('+-*')
    a = dec(rng, size(rng, 5)) if rng.random() < 0.8 else str(nat(rng, size(rng, 5)))
    b = dec(rng, size(rng, 5)) if rng.random() < 0.8 else str(nat(rng, size(rng, 5)))
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
NICE_RAD = ['p', 'p/2', 'p/3', 'p/4', 'p/6', '2*p', '3*p/2', '2*p/3', '3*p/4', '5*p/6', '-p/2', '-p/4']


def g_trig(rng, rad=None):
    """sin, cos or tan of an angle, in degrees, or about a third of the time in
    radians with the leading R the page adds in RAD mode."""
    f = rng.choice('sct')
    if rad if rad is not None else rng.random() < 0.35:
        r = rng.random()
        if r < 0.45:
            a = rng.choice(NICE_RAD)
        elif r < 0.8:
            a = dec(rng, 1)
        elif r < 0.9:
            a = str(rng.randrange(10))
        else:
            a = f'-{dec(rng, 1)}'
        return RAD + f + (a if len(a) == 1 or a.isdigit() and rng.random() < 0.4 else f'({a})')
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


def g_log(rng):
    """log or ln: powers of ten, e to a power, plain numbers, and now and then
    zero or a negative, which is Error."""
    f = rng.choice('ln')
    r = rng.random()
    if r < 0.25:
        x = '1' + '0' * rng.randrange(8)
    elif r < 0.35:
        x = 'e' if rng.random() < 0.5 else f'e^{rng.randrange(-3, 10)}'
    elif r < 0.8:
        x = str(nat(rng, rng.randint(1, 5)))
    elif r < 0.93:
        x = dec(rng, 3)
    else:
        x = '0' if rng.random() < 0.4 else f'-{nat(rng, rng.randint(1, 3))}'
    return f + (x if x.isdigit() and rng.random() < 0.4 else f'({x})')


def g_exp(rng):
    """e to the x, from the e^x key, which types e^(."""
    r = rng.random()
    if r < 0.6:
        x = str(rng.randrange(-6, 16))
    elif r < 0.85:
        x = dec(rng, 1)
    else:
        x = f'{rng.randrange(1, 10)}{rng.choice("+-*/")}{rng.randrange(1, 10)}'
    return 'e^' + (x if x.isdigit() and rng.random() < 0.3 else f'({x})')


def g_const(rng):
    """pi and e in ordinary arithmetic."""
    k = rng.choice('pe')
    r = rng.random()
    if r < 0.12:
        return k
    if r < 0.4:
        return f'{num(rng, 2)}*{k}'
    if r < 0.65:
        return f'{k}{rng.choice("+-*/")}{num(rng, 2, 0.2)}'
    if r < 0.8:
        return f'{k}^{rng.randint(2, 4)}'
    return f'{num(rng, 3)}{rng.choice("+-*/")}{k}'


# ---- longer expressions --------------------------------------------------------

def chain(rng, n, maxd=2, ops='++--*/', big=False):
    s = num(rng, maxd, big=big)
    for _ in range(n - 1):
        s += rng.choice(ops) + num(rng, maxd, big=big)
    return s


def g_chain(rng):
    return chain(rng, rng.randint(3, 5), 2 if rng.random() < 0.8 else 3, big=True)


# precedence of what a node renders as: sums 1, products 2, negation 3,
# powers 4, functions and numbers 5
PREC = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 4}


def tree(rng, depth, p_leaf=0.3, big=False):
    """A random expression tree, rendered with the parentheses it needs and
    occasionally one it doesn't. Returns (string, precedence)."""
    if depth == 0 or rng.random() < p_leaf:
        if rng.random() < 0.04:
            return rng.choice('pe'), 5
        return num(rng, 2 if rng.random() < 0.85 else 3, p_dec=0.08, big=big), 5
    r = rng.random()
    if r < 0.12:
        f = rng.choice('rrsctln')
        s, p = tree(rng, depth - 1, p_leaf + 0.2, big)
        return f + (s if p == 5 and rng.random() < 0.4 else f'({s})'), 5
    if r < 0.17:
        s, p = tree(rng, depth - 1, p_leaf + 0.2, big)
        return '-' + (s if p >= 4 else f'({s})'), 3
    op = rng.choice('++--**//^') if r < 0.97 else '^'
    pr = PREC[op]
    if op == '^':
        ls, lp = num(rng, 1 if rng.random() < 0.5 else 2), 5
        rs, rp = str(rng.randrange(5)), 5
        if rng.random() < 0.25:
            ls, lp = tree(rng, depth - 1, 0.6, big)
    else:
        ls, lp = tree(rng, depth - 1, p_leaf, big)
        rs, rp = tree(rng, depth - 1, p_leaf, big)
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
    return tree(rng, rng.randint(2, 3), big=True)[0]


def g_long(rng):
    """A long expression, every length from 14 tokens to the most the keypad takes
    about equally likely. Left alone the other families thin out fast past 13."""
    target = rng.randint(14, MAX_EXPR)
    depth = 3 if target < 24 else 4 if target < 42 else 5
    while True:
        if rng.random() < 0.6:
            e = tree(rng, depth, 0.15, big=True)[0]
        else:
            e = chain(rng, max(2, target // rng.choice([3, 5]) + rng.randint(-2, 2)), 2 if rng.random() < 0.75 else 3, big=True)
        # a few tokens short: one more operator and a number exactly that long lands it
        # (with numbers up to 8 digits, waiting for an exact hit took dozens of tries)
        gap = target - len(e) - 1
        if 1 <= gap <= 8:
            e += rng.choice('++--*/') + str(nat(rng, gap))
        if len(e) == target:
            return e


def g_neg(rng):
    r = rng.random()
    a, b = num(rng, 3, big=True), num(rng, 3, big=True)
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


# ---- things that don't parse ---------------------------------------------------

def _broken(rng, e):
    """e with one slip of the finger in it."""
    i = rng.randrange(len(e) + 1)
    r = rng.random()
    if r < 0.22 and e:                                    # a key missed
        j = rng.randrange(len(e))
        return e[:j] + e[j + 1:]
    if r < 0.44:                                          # an extra operator
        return e[:i] + rng.choice('+*/^.') + e[i:]
    if r < 0.56:                                          # a stray closing bracket
        return e[:i] + ')' + e[i:]
    if r < 0.64:                                          # empty brackets
        return e[:i] + '()' + e[i:]
    if r < 0.80:                                          # left hanging
        return e + rng.choice('+-*/^.') if rng.random() < 0.8 else e + rng.choice('rsc')
    if r < 0.90:                                          # started with an operator
        return rng.choice('*/^).') + e
    if e:                                                 # a key swapped
        j = rng.randrange(len(e))
        return e[:j] + rng.choice('+*/^.()') + e[j + 1:]
    return e


def g_syntax(rng):
    """Something that doesn't parse, written the way the page sends it: implied
    multiplication spelled out and open brackets closed. Mostly a valid sum
    with one slip in it, sometimes two, sometimes a stub like '+' or 'r'."""
    while True:
        if rng.random() < 0.08:
            e = rng.choice(['+', '-', '*', '/', '^', '.', 'r', 's', 'c', '()', '..', ')', '(', '*2', '2*'])
        else:
            e = rng.choices(_SOURCES, _SOURCE_WEIGHTS)[0](rng)
            for _ in range(1 if rng.random() < 0.8 else 2):
                e = _broken(rng, e)
        e = close(explicit(e), MAX_EXPR)
        if e and len(e) <= MAX_EXPR and answer(e) == SYN:
            return e


FAMILIES = [
    ('add', g_add, 0.18),
    ('sub', g_sub, 0.13),
    ('mul', g_mul, 0.14),
    ('huge', g_huge, 0.05),
    ('div', g_div, 0.08),
    ('dec', g_dec, 0.06),
    ('pow', g_pow, 0.07),
    ('sqrt', g_sqrt, 0.06),
    ('trig', g_trig, 0.07),
    ('log', g_log, 0.04),
    ('exp', g_exp, 0.03),
    ('const', g_const, 0.03),
    ('chain', g_chain, 0.10),
    ('tree', g_tree, 0.09),
    ('long', g_long, 0.30),
    ('neg', g_neg, 0.03),
    ('err', g_err, 0.01),
    ('syntax', g_syntax, 0.04),
]

_SOURCES, _SOURCE_WEIGHTS = zip(*[(f, w) for name, f, w in FAMILIES if name not in ('err', 'syntax')])


def fits(expr, ans):
    return len(expr) <= MAX_EXPR and len(ans) <= MAX_ANS


def moded(rng, e):
    """e as the page sends it: multiplication written out, and now and then in
    radians, which the page marks with a leading R when there's a trig key in it."""
    e = explicit(e)
    if not e.startswith(RAD) and any(f in e for f in 'sct') and len(e) < MAX_EXPR and rng.random() < 0.25:
        e = RAD + e
    return e


def sample(rng):
    """One (expression, answer) pair from the training mix."""
    names, fns, weights = zip(*FAMILIES)
    while True:
        expr = moded(rng, rng.choices(fns, weights)[0](rng))
        ans = answer(expr)
        if fits(expr, ans):
            return expr, ans


def batch(seed, n):
    rng = random.Random(seed)
    return [sample(rng) for _ in range(n)]
