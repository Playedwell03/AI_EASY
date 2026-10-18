import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { makeLoanCases, FIELDS } from '../js/modules/jev/data.js';
import jev from '../js/modules/jev/index.js';

test('makeLoanCases is reproducible with fields in range', () => {
  const a = makeLoanCases(300, mulberry32(2));
  assert.deepEqual(a, makeLoanCases(300, mulberry32(2)));
  assert.equal(a.fields.length, 300);
  for (const f of a.fields) {
    for (const def of FIELDS) {
      const v = f[def.key];
      if (def.type === 'num') assert.ok(v >= def.min && v <= def.max, `${def.key}=${v}`);
      else assert.ok(Number.isInteger(v) && v >= 0 && v < def.options.length);
    }
  }
});

test('labels: approve 0/1 with about 10% noise, all three routes, risk in [0,1]', () => {
  const { answers, clean } = makeLoanCases(4000, mulberry32(5));
  const flips = answers.filter((a, i) => a.approve !== clean[i].approve).length / answers.length;
  assert.ok(flips > 0.07 && flips < 0.13, `flip rate ${flips}`);
  const routes = new Set(answers.map((a) => a.route));
  assert.deepEqual([...routes].sort(), [0, 1, 2]);
  assert.ok(answers.every((a) => a.risk >= 0 && a.risk <= 1));
  const approveRate = answers.filter((a) => a.approve).length / answers.length;
  assert.ok(approveRate > 0.3 && approveRate < 0.7, `approve rate ${approveRate}`);
});

test('makeData splits and reports counts', () => {
  const d = jev.makeData({ ...jev.defaults, n: 200 });
  assert.equal(d.meta.counts.train + d.meta.counts.val + d.meta.counts.test, 200);
  assert.ok(jev.baselineLoss(jev.defaults, d) > 0);
});

test('default Jev-style model is accurate and well calibrated', slow, async () => {
  const r = await trainModule(jev);
  const res = await r.session.evaluateTest();
  assert.ok(res.acc >= 0.8, `yes/no acc ${res.acc}`);
  assert.ok(res.ece <= 0.08, `ECE ${res.ece}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard)`);
  const [p] = r.session.predict([r.data.test.fields[0]]);
  assert.ok(p.approve >= 0 && p.approve <= 1);
  assert.equal(p.route.length, 3);
  assert.ok(Math.abs(p.route.reduce((a, b) => a + b, 0) - 1) < 1e-4);
  r.session.dispose();
});

for (const p of jev.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(jev, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
