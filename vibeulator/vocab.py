"""Every calculator key is one token, and so is everything the display can show.

r, s, c and t are the square root, sine, cosine and tangent keys, l and n are
log and ln, and p is pi. e is the constant e in a sum and the exponent in an
answer like 1.5e12. E is "Error", S is "Syntax error", and an R in front of a
sum means its angles are in radians. The web page maps them back to √, sin,
cos and so on.
"""

PAD, BOS, EOS = 0, 1, 2
TOKENS = ['<pad>', '<bos>', '<eos>'] + list('0123456789.+-*/^()rsc=eEStlnpR')
ID = {t: i for i, t in enumerate(TOKENS)}
EQ = ID['=']

MAX_EXPR = 60          # tokens of input the page accepts
MAX_ANS = 16           # the longest display string, e.g. -1.234567891e-12
BLOCK = 1 + MAX_EXPR + 1 + MAX_ANS   # bos, expression, '=', answer; eos is only ever a target


def encode(expr, ans=None):
    ids = [BOS] + [ID[ch] for ch in expr] + [EQ]
    if ans is not None:
        ids += [ID[ch] for ch in ans] + [EOS]
    return ids


def decode(ids):
    return ''.join(TOKENS[i] for i in ids if i > EOS)
