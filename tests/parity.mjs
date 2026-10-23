// Holds the browser code to the Python it mirrors.
//
//   calc.js   must give calc.py's display string for every case in calc-fixture.json
//   engine.js must reproduce PyTorch's logits, step by step, for every probe in
//             model-fixture.json, and decode to the same answers
//
//   node tests/parity.mjs [weights.bin] [model-fixture.json]
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const load = (file) => {
  const ctx = { self: {}, TextDecoder };
  vm.runInNewContext(fs.readFileSync(here(file), 'utf8'), ctx);
  return ctx.self;
};

let failures = 0;
const fail = (msg) => { failures++; if (failures <= 20) console.log('  FAIL  ' + msg); };

// ---- calc ---------------------------------------------------------------------
const { VibCalc } = load('../web/calc.js');
const cases = JSON.parse(fs.readFileSync(here('calc-fixture.json'), 'utf8'));
for (const [expr, want] of cases) {
  const got = VibCalc.answer(expr);
  if (got !== want) fail(`calc ${JSON.stringify(expr)}: got ${got}, want ${want}`);
}
console.log(`calc: ${cases.length} cases, ${failures} mismatches`);

// ---- engine ---------------------------------------------------------------------
const binPath = process.argv[2] || here('../weights/vibeulator.bin');
const fixPath = process.argv[3] || here('model-fixture.json');
if (!fs.existsSync(binPath) || !fs.existsSync(fixPath)) {
  console.log(`engine: skipped, no ${fs.existsSync(binPath) ? fixPath : binPath} (train and export first)`);
} else {
  const before = failures;
  const { Vibeulator } = load('../web/engine.js');
  const b = fs.readFileSync(binPath);
  const model = new Vibeulator.Model(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  const { probes } = JSON.parse(fs.readFileSync(fixPath, 'utf8'));
  let worst = 0;
  for (const pr of probes) {
    const ids = model.encode(pr.expr.split(''));
    let lg = model.prompt(ids);
    for (let s = 0; s < pr.logits.length; s++) {
      const want = pr.logits[s];
      for (let v = 0; v < want.length; v++) worst = Math.max(worst, Math.abs(lg[v] - want[v]));
      let arg = 0;
      for (let v = 1; v < want.length; v++) if (want[v] > want[arg]) arg = v;
      if (s + 1 < pr.logits.length) lg = model.step(arg);
    }
    const got = model.answer(pr.expr.split(''), 0).text;
    if (got !== pr.answer) fail(`engine ${pr.expr}: answered ${got}, PyTorch ${pr.answer}`);
  }
  if (worst > 2e-3) fail(`engine logits differ from PyTorch by up to ${worst}`);
  console.log(`engine: ${probes.length} probes, max logit difference ${worst.toExponential(2)}, ${failures - before} mismatches`);
}

if (failures) { console.log(`${failures} failures`); process.exit(1); }
console.log('all good');
