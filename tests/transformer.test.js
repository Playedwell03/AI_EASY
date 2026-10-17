import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { initBlockParams, multiHeadAttention, transformerBlock, disposeParams, createEncoder } from '../js/shared/attention.js';
import transformer from '../js/modules/transformer/index.js';

function attentionWeights(causal) {
  const p = initBlockParams(tf, 16, 4, 32, 3);
  const x = tf.randomNormal([2, 6, 16], 0, 1, 'float32', 5);
  const { out, weights } = multiHeadAttention(tf, p, x, { heads: 4, causal });
  assert.deepEqual(out.shape, [2, 6, 16]);
  assert.deepEqual(weights.shape, [2, 4, 6, 6]);
  const w = weights.arraySync();
  tf.dispose([x, out, weights]);
  disposeParams(p);
  return w;
}

test('causal attention never looks ahead and each row sums to 1', () => {
  const w = attentionWeights(true);
  for (const b of w) for (const head of b) head.forEach((row, i) => {
    assert.ok(Math.abs(row.reduce((a, v) => a + v, 0) - 1) < 1e-5);
    for (let j = i + 1; j < row.length; j++) assert.ok(row[j] < 1e-6, `look-ahead ${row[j]}`);
  });
});

test('bidirectional attention can look both ways', () => {
  const w = attentionWeights(false);
  const row = w[0][0][0];
  assert.ok(Math.abs(row.reduce((a, v) => a + v, 0) - 1) < 1e-5);
  assert.ok(row.slice(1).some((v) => v > 1e-3));
});

test('createEncoder leaves no tensors behind after dispose', () => {
  const before = tf.memory().numTensors;
  const enc = createEncoder(tf, { vocab: 10, dModel: 8, heads: 2, ff: 16, blocks: 2, maxLen: 6, seed: 1 });
  enc.dispose();
  assert.equal(tf.memory().numTensors, before);
});

test('transformerBlock keeps the shape', () => {
  const p = initBlockParams(tf, 16, 2, 32, 1);
  const x = tf.randomNormal([1, 5, 16], 0, 1, 'float32', 2);
  const { out, weights } = transformerBlock(tf, p, x, { heads: 2, causal: true });
  assert.deepEqual(out.shape, [1, 5, 16]);
  tf.dispose([x, out, weights]);
  disposeParams(p);
});

test('default Transformer LM learns the corpus and exposes attention maps', slow, async () => {
  const r = await trainModule(transformer);
  const res = await r.session.evaluateTest();
  const V = r.data.vocab.itos.length;
  assert.ok(res.perplexity < 0.5 * V, `ppl ${res.perplexity} vocab ${V}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard; the 1-minute target is checked in the browser)`);
  assert.equal(typeof r.session.generate('the ', 10, 0, mulberry32(1)), 'string');
  const maps = r.session.attentionMaps('고양이는 집에서');
  assert.equal(maps.length, r.hp.blocks);
  assert.equal(maps[0].length, r.hp.heads);
  r.session.dispose();
});

for (const p of transformer.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(transformer, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
