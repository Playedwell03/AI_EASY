import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { maskTokens } from '../js/modules/bert/model.js';
import bert from '../js/modules/bert/index.js';

const MASK = 99;
const PAD = 98;

test('maskTokens hides about 15% of real tokens with the 80/10/10 rule', () => {
  const rand = mulberry32(3);
  let real = 0;
  let masked = 0;
  let asMask = 0;
  let kept = 0;
  for (let n = 0; n < 400; n++) {
    const ids = Array.from({ length: 30 }, (_, i) => (i < 25 ? 1 + (i % 50) : PAD));
    const { input, labels } = maskTokens(ids, rand, { maskProb: 0.15, maskId: MASK, padId: PAD, vocabSize: 60 });
    ids.forEach((id, i) => {
      if (id === PAD) {
        assert.equal(labels[i], -1, 'padding is never a target');
        assert.equal(input[i], PAD);
        return;
      }
      real++;
      if (labels[i] === -1) {
        assert.equal(input[i], id, 'unselected tokens are untouched');
        return;
      }
      masked++;
      assert.equal(labels[i], id, 'label is the original token');
      if (input[i] === MASK) asMask++;
      else if (input[i] === id) kept++;
    });
  }
  assert.ok(Math.abs(masked / real - 0.15) < 0.02, `masked ${masked / real}`);
  assert.ok(Math.abs(asMask / masked - 0.8) < 0.05, `as [MASK] ${asMask / masked}`);
  assert.ok(Math.abs(kept / masked - 0.1) < 0.04, `kept ${kept / masked}`);
});

test('maskTokens always masks at least one token', () => {
  const { labels } = maskTokens([5, 6, 7], mulberry32(1), { maskProb: 0.0001, maskId: MASK, padId: PAD, vocabSize: 60 });
  assert.equal(labels.filter((l) => l !== -1).length, 1);
});

test('makeData pads sentences and reserves [PAD] and [MASK]', () => {
  const d = bert.makeData(bert.defaults);
  const { itos } = d.vocab;
  assert.ok(itos.includes('[MASK]') && itos.includes('[PAD]'));
  assert.equal(d.train.ids[0].length, bert.defaults.maxLen);
  assert.equal(d.meta.counts.train, d.train.ids.length);
});

test('default BERT fills masks well above chance', slow, async () => {
  const r = await trainModule(bert);
  const res = await r.session.evaluateTest();
  assert.ok(res.acc >= 0.35, `top-1 ${res.acc}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard)`);
  const fill = r.session.fillMask('고양이는 공원에서 우유를 마신다.', [10]);
  assert.equal(fill.length, 1);
  assert.equal(fill[0].top.length, 5);
  assert.equal(r.session.sentenceVector('고양이는 집에서 책을 읽는다.').length, r.hp.dModel);
  r.session.dispose();
});

for (const p of bert.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(bert, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
