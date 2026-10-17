import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { makeSeries, windows } from '../js/modules/timeseries/data.js';
import ts from '../js/modules/timeseries/index.js';

test('makeSeries is reproducible and has the requested length', () => {
  const opts = { trend: 1, season: 1, noise: 0.2 };
  const a = makeSeries(300, opts, mulberry32(4));
  assert.equal(a.length, 300);
  assert.deepEqual(a, makeSeries(300, opts, mulberry32(4)));
  assert.ok(a.every(Number.isFinite));
});

test('windows pairs each window with the next value', () => {
  const s = [0, 1, 2, 3, 4, 5];
  const { x, y, t } = windows(s, 3);
  assert.deepEqual(x, [[0, 1, 2], [1, 2, 3], [2, 3, 4]]);
  assert.deepEqual(y, [3, 4, 5]);
  assert.deepEqual(t, [3, 4, 5]);
});

test('makeData splits in time order without shuffling', () => {
  const d = ts.makeData(ts.defaults);
  const maxTrain = Math.max(...d.train.t);
  const minVal = Math.min(...d.val.t);
  const maxVal = Math.max(...d.val.t);
  const minTest = Math.min(...d.test.t);
  assert.ok(maxTrain < minVal && maxVal < minTest);
  assert.equal(d.meta.counts.train, d.train.x.length);
});

test('default model beats the naive "repeat last value" forecast on the test set', async () => {
  const r = await trainModule(ts);
  const res = await r.session.evaluateTest();
  r.session.dispose();
  assert.ok(res.mae < 0.8 * res.naiveMae, `mae ${res.mae} vs naive ${res.naiveMae}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 60, `took ${r.seconds}s`);
});

for (const p of ts.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(ts, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
