// Pure decision logic used inside the "Use" tab apps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toPlane } from '../js/modules/mlp/uses.js';

test('mlp toPlane maps a feature range onto [-1, 1] and clamps', () => {
  assert.equal(toPlane(0, 0, 20), -1);
  assert.equal(toPlane(20, 0, 20), 1);
  assert.equal(toPlane(10, 0, 20), 0);
  assert.equal(toPlane(30, 0, 20), 1);
});

import { readNumber, calc } from '../js/modules/cnn/uses.js';

test('cnn readNumber joins digits and flags unsure boxes', () => {
  const r = readNumber([{ digit: 1, prob: 0.99 }, { digit: 2, prob: 0.4 }, { digit: 7, prob: 0.9 }], 0.6);
  assert.equal(r.text, '127');
  assert.equal(r.value, 127);
  assert.deepEqual(r.unsure, [1]);
  assert.equal(readNumber([], 0.6).value, null);
  assert.equal(readNumber([null, { digit: 3, prob: 1 }], 0.6).text, '?3');
});

test('cnn calc handles the four operations and division by zero', () => {
  assert.equal(calc(6, '+', 7), 13);
  assert.equal(calc(6, '-', 7), -1);
  assert.equal(calc(6, '×', 7), 42);
  assert.equal(calc(8, '÷', 2), 4);
  assert.equal(calc(8, '÷', 0), null);
});

import { pixelError, percentile, addNoise, lerp2 } from '../js/modules/vae/uses.js';
import { mulberry32 } from '../js/core/rng.js';

test('vae pixelError is mean squared difference', () => {
  assert.equal(pixelError([0, 1, 0, 1], [0, 1, 0, 1]), 0);
  assert.equal(pixelError([0, 0], [1, 1]), 1);
  assert.equal(pixelError([0, 0], [0.5, 0]), 0.125);
});

test('vae percentile picks the q-th value', () => {
  const xs = Array.from({ length: 101 }, (_, i) => i);
  assert.equal(percentile(xs, 0.95), 95);
  assert.equal(percentile([3, 1, 2], 0), 1);
});

test('vae addNoise is reproducible and stays in [0,1]', () => {
  const img = new Float32Array(400).fill(0.5);
  const a = addNoise(img, 0.6, mulberry32(1));
  assert.deepEqual(a, addNoise(img, 0.6, mulberry32(1)));
  assert.ok(a.every((v) => v >= 0 && v <= 1));
  assert.notDeepEqual(Array.from(a), Array.from(img));
});

test('vae lerp2 interpolates latent points', () => {
  assert.deepEqual(lerp2([0, 0], [2, 4], 0.5), [1, 2]);
  assert.deepEqual(lerp2([1, 1], [3, 3], 0), [1, 1]);
});

import { orderQuantity, hoursAbove, outsideBand } from '../js/modules/timeseries/uses.js';

test('timeseries orderQuantity covers forecast + safety minus stock, never negative', () => {
  assert.equal(orderQuantity([10, 20, 30], 25, 5), 40);
  assert.equal(orderQuantity([1, 1], 100, 0), 0);
  assert.equal(orderQuantity([10.2], 0, 0), 11);
});

test('timeseries hoursAbove lists indices over the limit', () => {
  assert.deepEqual(hoursAbove([1, 5, 3, 7], 4), [1, 3]);
});

test('timeseries outsideBand flags points far from the forecast', () => {
  assert.deepEqual(outsideBand([1, 2, 10, 4], [1, 2.5, 3, 4.2], 1), [2]);
});

import { topK, charProbs, textPerplexity } from '../js/modules/rnn/uses.js';

test('rnn topK returns the k most likely indices in order', () => {
  assert.deepEqual(topK([0.1, 0.5, 0.05, 0.35], 2).map((x) => x.i), [1, 3]);
});

// A fake model that always predicts 'a' with 0.9 and 'b' with 0.1.
const fakeNext = () => [0.9, 0.1];
const stoi = { a: 0, b: 1 };

test('rnn charProbs scores each character given the ones before it', () => {
  const ps = charProbs('aab', fakeNext, stoi);
  assert.equal(ps[0], null); // the first character has no context
  assert.equal(ps[1], 0.9);
  assert.equal(ps[2], 0.1);
  assert.equal(charProbs('ax', fakeNext, stoi)[1], 0); // unknown character
});

test('rnn textPerplexity is exp of mean negative log probability', () => {
  assert.ok(Math.abs(textPerplexity('aaa', fakeNext, stoi) - 1 / 0.9) < 1e-9);
});

import { rankByScore } from '../js/modules/transformer/uses.js';

test('transformer rankByScore sorts candidates by ascending perplexity', () => {
  const ppl = { a: 3, b: 1.5, c: 9 };
  assert.deepEqual(rankByScore(['a', 'b', 'c'], (s) => ppl[s]).map((r) => r.text), ['b', 'a', 'c']);
  assert.equal(rankByScore(['a'], () => 2)[0].score, 2);
});

import { cosine, shouldFix, nearest } from '../js/modules/bert/uses.js';

test('bert cosine similarity', () => {
  assert.equal(cosine([1, 0], [1, 0]), 1);
  assert.equal(cosine([1, 0], [0, 1]), 0);
  assert.ok(Math.abs(cosine([1, 1], [-1, -1]) + 1) < 1e-9);
});

test('bert shouldFix only replaces unlikely characters with a much likelier one', () => {
  assert.equal(shouldFix(0.01, { c: 'x', p: 0.9 }), true);
  assert.equal(shouldFix(0.4, { c: 'x', p: 0.5 }), false);
  assert.equal(shouldFix(0.01, { c: 'x', p: 0.03 }), false);
});

test('bert nearest ranks items by similarity to the query vector', () => {
  const items = [{ text: 'a', v: [1, 0] }, { text: 'b', v: [0.7, 0.7] }, { text: 'c', v: [0, 1] }];
  assert.deepEqual(nearest([1, 0.1], items, 2).map((x) => x.text), ['a', 'b']);
});
