import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { makePoints } from '../js/modules/mlp/data.js';
import mlp from '../js/modules/mlp/index.js';

test('makePoints: balanced labels inside [-1,1], reproducible', () => {
  for (const kind of ['circle', 'xor', 'spiral', 'moons']) {
    const a = makePoints(kind, 200, 0, mulberry32(1));
    const b = makePoints(kind, 200, 0, mulberry32(1));
    assert.deepEqual(a, b);
    assert.equal(a.x.length, 200);
    const ones = a.y.filter((v) => v === 1).length;
    assert.ok(ones >= 60 && 200 - ones >= 60, `${kind} balance ${ones}`);
    for (const [x, y] of a.x) assert.ok(Math.abs(x) <= 1 && Math.abs(y) <= 1, kind);
  }
});

test('makeData splits into train/val/test with counts', () => {
  const d = mlp.makeData(mlp.defaults);
  const { train, val, test: te } = d.meta.counts;
  assert.equal(train + val + te, mlp.defaults.n);
  assert.equal(d.train.x.length, train);
});

test('default MLP learns the circle well', slow, async () => {
  const r = await trainModule(mlp);
  const res = await r.session.evaluateTest();
  r.session.dispose();
  assert.ok(res.acc >= 0.9, `test acc ${res.acc}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 30, `took ${r.seconds}s`);
});

for (const p of mlp.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(mlp, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
