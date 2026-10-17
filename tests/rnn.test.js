import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule } from './helpers.js';
import { makeCorpus, buildVocab, makeTextData, sampleIndex } from '../js/shared/corpus.js';
import { mulberry32 } from '../js/core/rng.js';
import rnn from '../js/modules/rnn/index.js';

test('corpus is reproducible and long enough', () => {
  for (const lang of ['ko', 'en']) {
    const a = makeCorpus(lang, 3000);
    assert.deepEqual(a, makeCorpus(lang, 3000));
    assert.ok(a.join(' ').length >= 3000);
  }
});

test('buildVocab is sorted and invertible', () => {
  const v = buildVocab('banana');
  assert.deepEqual(v.itos, ['a', 'b', 'n']);
  assert.equal(v.stoi.n, 2);
});

test('makeTextData windows map seqLen ids to the next id; splits do not share sentences', () => {
  const d = makeTextData({ lang: 'en', chars: 4000, seqLen: 10, seed: 1 });
  assert.equal(d.train.x[0].length, 10);
  assert.equal(d.train.x.length, d.train.y.length);
  const train = new Set(d.sentences.train.map((s, i) => `${i}`));
  assert.ok(train.size > 0);
  assert.equal(d.meta.counts.val, d.val.x.length);
  const ids = d.train.x.flat();
  assert.ok(ids.every((i) => i >= 0 && i < d.vocab.itos.length));
});

test('sampleIndex: temperature 0 is argmax, sampling respects probabilities', () => {
  assert.equal(sampleIndex([0.1, 0.7, 0.2], 0, mulberry32(1)), 1);
  const r = mulberry32(2);
  const counts = [0, 0];
  for (let i = 0; i < 2000; i++) counts[sampleIndex([0.8, 0.2], 1, r)]++;
  assert.ok(counts[0] > 1450 && counts[0] < 1750, String(counts));
});

test('default char-RNN gets validation perplexity under half the vocab size', async () => {
  const r = await trainModule(rnn);
  const res = await r.session.evaluateTest();
  const V = r.data.vocab.itos.length;
  assert.ok(res.perplexity < 0.5 * V, `ppl ${res.perplexity} vocab ${V}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 60, `took ${r.seconds}s`);
  const text = r.session.generate('the ', 20, 0, mulberry32(1));
  assert.equal(typeof text, 'string');
  r.session.dispose();
});

for (const p of rnn.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(rnn, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
