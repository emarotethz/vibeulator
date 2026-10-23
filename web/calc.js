// The real answer, for comparison. A port of vibeulator/calc.py, which is what
// the model was trained on: same grammar, float64 arithmetic, and the same
// 10-digit display with halves rounded away from zero (toExponential's rule,
// which calc.py copies). tests/parity.mjs checks the two agree.
(function (root) {
'use strict';

var LIMIT = 1e100, SIG = 10, ERR = 'E';

function MathError() {}
function SyntaxErr() {}

function check(v) {
  if (!isFinite(v) || Math.abs(v) >= LIMIT) throw new MathError();
  return v;
}

function pow(b, e) {
  if (b === 0 && e < 0) throw new MathError();
  if (b < 0 && e !== Math.floor(e)) throw new MathError();
  return check(Math.pow(b, e));
}

function trig(f, x) {
  var r = x % 360;
  if (r < 0) r += 360;
  var v = f(r * (Math.PI / 180));
  return Math.floor(v * 1e12 + 0.5) / 1e12;
}

function func(c, x) {
  if (c === 'r') {
    if (x < 0) throw new MathError();
    return Math.sqrt(x);
  }
  return trig(c === 's' ? Math.sin : Math.cos, x);
}

function Parser(s) { this.s = s; this.i = 0; }
Parser.prototype.peek = function () { return this.i < this.s.length ? this.s[this.i] : ''; };
Parser.prototype.expr = function () {
  var v = this.term();
  while (this.peek() === '+' || this.peek() === '-') {
    var op = this.s[this.i++], w = this.term();
    v = check(op === '+' ? v + w : v - w);
  }
  return v;
};
Parser.prototype.term = function () {
  var v = this.unary();
  while (this.peek() === '*' || this.peek() === '/') {
    var op = this.s[this.i++], w = this.unary();
    if (op === '*') v = check(v * w);
    else {
      if (w === 0) throw new MathError();
      v = check(v / w);
    }
  }
  return v;
};
Parser.prototype.unary = function () {
  if (this.peek() === '-') { this.i++; return -this.unary(); }
  return this.power();
};
Parser.prototype.power = function () {
  var b = this.prim();
  if (this.peek() === '^') { this.i++; return pow(b, this.unary()); }
  return b;
};
Parser.prototype.prim = function () {
  var c = this.peek();
  if (c === 'r' || c === 's' || c === 'c') { this.i++; return check(func(c, this.prim())); }
  if (c === '(') {
    this.i++;
    var v = this.expr();
    if (this.peek() !== ')') throw new SyntaxErr();
    this.i++;
    return v;
  }
  var j = this.i;
  while (/[0-9.]/.test(this.peek())) this.i++;
  var num = this.s.slice(j, this.i);
  if (num === '' || num === '.' || num.indexOf('.') !== num.lastIndexOf('.')) throw new SyntaxErr();
  return parseFloat(num.charAt(0) === '.' ? '0' + num : num);
};

function evaluate(s) {
  var p = new Parser(s), v = p.expr();
  if (p.i !== s.length) throw new SyntaxErr();
  return v;
}

function fmt(x) {
  if (x === 0) return '0';
  var s = Math.abs(x).toExponential(SIG - 1), k = s.indexOf('e');
  var digits = s.slice(0, k).replace('.', ''), exp = +s.slice(k + 1), out;
  if (exp >= -4 && exp < SIG) {
    var whole, frac;
    if (exp >= 0) { whole = digits.slice(0, exp + 1); frac = digits.slice(exp + 1); }
    else { whole = '0'; frac = new Array(-exp).join('0') + digits; }
    frac = frac.replace(/0+$/, '');
    out = whole + (frac ? '.' + frac : '');
  } else {
    out = (digits[0] + '.' + digits.slice(1)).replace(/0+$/, '').replace(/\.$/, '') + 'e' + exp;
  }
  return (x < 0 ? '-' : '') + out;
}

// The display string for s, 'E' for a math error, null if s doesn't parse.
function answer(s) {
  try {
    return fmt(evaluate(s));
  } catch (e) {
    if (e instanceof MathError) return ERR;
    if (e instanceof SyntaxErr) return null;
    throw e;
  }
}

root.VibCalc = { answer: answer, evaluate: evaluate, fmt: fmt, ERR: ERR };

})(typeof self !== 'undefined' ? self : this);
