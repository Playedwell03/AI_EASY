import { test } from 'node:test';
import assert from 'node:assert/strict';
import { diagnoseCurves } from '../js/core/diagnose.js';

const hist = (train, val = train) => train.map((t, i) => ({ trainLoss: t, valLoss: val[i] }));

test('NaN loss is diverged', () => {
  const d = diagnoseCurves(hist([2, NaN]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'diverged');
  assert.ok(d.advice.includes('adv.lowerLr'));
});

test('loss growing past 3x the start is diverged', () => {
  assert.equal(diagnoseCurves(hist([1, 2, 4]), { baselineLoss: 0.69 }).verdict, 'diverged');
});

test('fewer than 3 epochs is insufficient, not a verdict', () => {
  const d = diagnoseCurves(hist([0.6, 0.3]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'insufficient');
  assert.deepEqual(d.advice, ['adv.trainMore']);
});

test('validation rising after its minimum while train keeps falling is overfitting', () => {
  const d = diagnoseCurves(
    hist([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], [1.0, 0.6, 0.4, 0.5, 0.55, 0.6]),
    { baselineLoss: 0.69 },
  );
  assert.equal(d.verdict, 'overfitting');
  assert.equal(d.bestEpoch, 3);
  const f = d.findings.find((x) => x.key === 'f.valRoseAfter');
  assert.equal(f.vars.epoch, 3);
  assert.equal(f.vars.pct, 50);
  assert.deepEqual(d.advice, ['adv.earlyStop', 'adv.regularize', 'adv.moreData']);
});

test('loss stuck near the random-guess baseline is underfitting', () => {
  const d = diagnoseCurves(hist([0.68, 0.67, 0.66]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'underfitting');
  assert.deepEqual(d.advice, ['adv.biggerModel', 'adv.trainMore', 'adv.raiseLr']);
});

test('healthy curves are good and flag still-improving', () => {
  const d = diagnoseCurves(hist([0.6, 0.3, 0.15, 0.1], [0.62, 0.32, 0.18, 0.14]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'good');
  assert.equal(d.stillImproving, true);
  assert.ok(d.advice.includes('adv.trainMore'));
  const b = d.findings.find((x) => x.key === 'f.beatsBaseline');
  assert.ok(Math.abs(b.vars.ratio - 0.69 / 0.14) < 0.01);
  assert.ok(d.findings.some((x) => x.key === 'f.gapSmall'));
});

test('good curve that plateaued is not still improving', () => {
  const d = diagnoseCurves(hist([0.6, 0.2, 0.1, 0.1, 0.1, 0.1], [0.6, 0.2, 0.12, 0.12, 0.12, 0.12]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'good');
  assert.equal(d.stillImproving, false);
  assert.deepEqual(d.advice, []);
});
