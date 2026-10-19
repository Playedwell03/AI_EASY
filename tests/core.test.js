import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, randn, shuffle } from '../js/core/rng.js';
import { splitIndices } from '../js/core/split.js';
import { accuracy, confusionMatrix, reliabilityBins, ece, mse, mae, perplexity } from '../js/core/metrics.js';

test('mulberry32 is reproducible per seed', () => {
  const a = mulberry32(42), b = mulberry32(42), c = mulberry32(43);
  const va = [a(), a(), a(), a(), a()];
  const vb = [b(), b(), b(), b(), b()];
  const vc = [c(), c(), c(), c(), c()];
  assert.deepEqual(va, vb);
  assert.notDeepEqual(va, vc);
  for (const v of va) assert.ok(v >= 0 && v < 1);
});

test('randn has roughly zero mean and unit variance', () => {
  const r = mulberry32(7);
  const xs = Array.from({ length: 5000 }, () => randn(r));
  const m = xs.reduce((s, x) => s + x, 0) / xs.length;
  const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
  assert.ok(Math.abs(m) < 0.06);
  assert.ok(Math.abs(v - 1) < 0.08);
});

test('shuffle permutes in place', () => {
  const arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const out = shuffle(arr, mulberry32(3));
  assert.equal(out, arr);
  assert.deepEqual([...arr].sort((x, y) => x - y), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test('splitIndices partitions 0..n-1 by ratio', () => {
  const s = splitIndices(100, [0.7, 0.15, 0.15], mulberry32(1));
  assert.equal(s.train.length, 70);
  assert.equal(s.val.length, 15);
  assert.equal(s.test.length, 15);
  const all = [...s.train, ...s.val, ...s.test].sort((a, b) => a - b);
  assert.deepEqual(all, Array.from({ length: 100 }, (_, i) => i));
});

test('accuracy and confusion matrix', () => {
  assert.equal(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75);
  assert.deepEqual(confusionMatrix([0, 1, 1], [0, 1, 0], 2), [[1, 0], [1, 1]]);
});

test('ece and reliability bins', () => {
  assert.ok(Math.abs(ece([0.9, 0.9, 0.9, 0.9], [true, true, true, false]) - 0.15) < 1e-9);
  assert.equal(ece([0.5, 0.5], [true, false]), 0);
  const bins = reliabilityBins([0.05, 0.95], [false, true], 10);
  assert.equal(bins.length, 10);
  assert.equal(bins[0].count, 1);
  assert.equal(bins[9].count, 1);
  assert.equal(bins[9].accuracy, 1);
});

test('regression metrics and perplexity', () => {
  assert.equal(mse([1, 2], [1, 4]), 2);
  assert.equal(mae([1, 2], [1, 4]), 1);
  assert.ok(Math.abs(perplexity(Math.log(5)) - 5) < 1e-9);
});
