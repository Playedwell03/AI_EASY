import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { renderDigit, makeDigits, DIGIT_SIZE } from '../js/shared/digits.js';
import cnn from '../js/modules/cnn/index.js';

test('renderDigit returns a size² image in [0,1] with some ink', () => {
  for (let d = 0; d < 10; d++) {
    const img = renderDigit(d, mulberry32(d + 1));
    assert.equal(img.length, DIGIT_SIZE * DIGIT_SIZE);
    let ink = 0;
    for (const v of img) { assert.ok(v >= 0 && v <= 1); ink += v; }
    assert.ok(ink > 15 && ink < 200, `digit ${d} ink ${ink}`);
  }
});

test('renderDigit is reproducible and varies between draws', () => {
  assert.deepEqual(renderDigit(3, mulberry32(9)), renderDigit(3, mulberry32(9)));
  assert.notDeepEqual(renderDigit(3, mulberry32(9)), renderDigit(3, mulberry32(10)));
});

test('makeDigits gives balanced labels', () => {
  const { x, y } = makeDigits(500, mulberry32(1));
  assert.equal(x.length, 500);
  for (let d = 0; d < 10; d++) assert.equal(y.filter((v) => v === d).length, 50);
});

test('default CNN reaches 90% test accuracy and a good diagnosis', slow, async () => {
  const r = await trainModule(cnn);
  const res = await r.session.evaluateTest();
  r.session.dispose();
  assert.ok(res.acc >= 0.9, `test acc ${res.acc}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard; the 1-minute target is checked in the browser)`);
});

for (const p of cnn.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(cnn, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
