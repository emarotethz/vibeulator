"""The ground truth: the grammar, the arithmetic and the display format the
model is trained to imitate. web/calc.js is a port of this file, and
tests/parity.mjs checks that the two agree on every case in the fixture.

Everything is evaluated in float64 and shown the way a 10-digit calculator
would show it, so the model has a single right string for each expression.

Grammar, loosest to tightest:
    expr  := term (('+' | '-') term)*
    term  := unary (('*' | '/') unary)*
    unary := '-' unary | power
    power := prim ('^' unary)?          right-associative, so 2^3^2 = 2^9
    prim  := func prim | '(' expr ')' | number
    func  := 'r' (square root) | 's' (sine) | 'c' (cosine), angles in degrees

so -2^2 = -4, and a function grabs only the next primary: r16^2 = (r16)^2.
"""
import math
from decimal import Decimal, ROUND_HALF_UP, localcontext

LIMIT = 1e100          # anything this big or bigger is an overflow, like a real calculator
SIG = 10               # significant digits on the display
ERR = 'E'              # the one token for "Error"


class MathError(Exception):
    pass


class SyntaxErr(Exception):
    pass


def _check(v):
    if not math.isfinite(v) or abs(v) >= LIMIT:
        raise MathError()
    return v


def _pow(b, e):
    if b == 0 and e < 0:
        raise MathError()
    if b < 0 and e != math.floor(e):
        raise MathError()
    try:
        return _check(math.pow(b, e))
    except (OverflowError, ValueError):
        raise MathError()


def _trig(f, x):
    r = x % 360.0
    v = f(r * (math.pi / 180.0))
    # sin(180) is 1.2e-16 in floating point and 0 on every calculator
    return math.floor(v * 1e12 + 0.5) / 1e12


def _func(c, x):
    if c == 'r':
        if x < 0:
            raise MathError()
        return math.sqrt(x)
    if c == 's':
        return _trig(math.sin, x)
    return _trig(math.cos, x)


class _Parser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ''

    def expr(self):
        v = self.term()
        while self.peek() in ('+', '-'):
            op = self.s[self.i]
            self.i += 1
            w = self.term()
            v = _check(v + w if op == '+' else v - w)
        return v

    def term(self):
        v = self.unary()
        while self.peek() in ('*', '/'):
            op = self.s[self.i]
            self.i += 1
            w = self.unary()
            if op == '*':
                v = _check(v * w)
            else:
                if w == 0:
                    raise MathError()
                v = _check(v / w)
        return v

    def unary(self):
        if self.peek() == '-':
            self.i += 1
            return -self.unary()
        return self.power()

    def power(self):
        b = self.prim()
        if self.peek() == '^':
            self.i += 1
            return _pow(b, self.unary())
        return b

    def prim(self):
        c = self.peek()
        if c in ('r', 's', 'c'):
            self.i += 1
            return _check(_func(c, self.prim()))
        if c == '(':
            self.i += 1
            v = self.expr()
            if self.peek() != ')':
                raise SyntaxErr()
            self.i += 1
            return v
        j = self.i
        while self.peek() in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.'):
            self.i += 1
        num = self.s[j:self.i]
        if num == '' or num == '.' or num.count('.') > 1:
            raise SyntaxErr()
        return float(num)


def evaluate(s):
    """Value of the expression. Raises MathError or SyntaxErr."""
    p = _Parser(s)
    v = p.expr()
    if p.i != len(s):
        raise SyntaxErr()
    return v


def fmt(x):
    """Show x the way a 10-digit calculator does. Halves round away from zero,
    which is what JavaScript's toExponential does, so calc.js can match this
    exactly."""
    if x == 0:
        return '0'
    with localcontext() as ctx:
        ctx.prec = 1100                          # enough to hold any float64 exactly
        d = abs(Decimal(x))
        exp = d.adjusted()
        q = Decimal(1).scaleb(-(SIG - 1))
        m = d.scaleb(-exp).quantize(q, rounding=ROUND_HALF_UP)
        if m >= 10:
            exp += 1
            m = (m / 10).quantize(q, rounding=ROUND_HALF_UP)
    digits = str(m).replace('.', '')
    if -4 <= exp < SIG:
        if exp >= 0:
            whole, frac = digits[:exp + 1], digits[exp + 1:]
        else:
            whole, frac = '0', '0' * (-exp - 1) + digits
        frac = frac.rstrip('0')
        out = whole + ('.' + frac if frac else '')
    else:
        mant = (digits[0] + '.' + digits[1:]).rstrip('0').rstrip('.')
        out = mant + 'e' + str(exp)
    return ('-' if x < 0 else '') + out


def answer(s):
    """The string the model should produce for s ('E' for any math error)."""
    try:
        return fmt(evaluate(s))
    except MathError:
        return ERR
