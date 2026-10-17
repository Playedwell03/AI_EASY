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

test('tiny jitter after near-zero loss is not overfitting', () => {
  // Relative rise is large (0.003 → 0.011) but absolute rise is negligible vs. the baseline.
  const train = [0.4, 0.1, 0.02, 0.005, 0.004, 0.003, 0.003, 0.003];
  const val = [0.39, 0.1, 0.03, 0.006, 0.003, 0.009, 0.012, 0.011];
  assert.equal(diagnoseCurves(hist(train, val), { baselineLoss: 0.69 }).verdict, 'good');
});

test('loss stuck far above random guessing is diverged', () => {
  const d = diagnoseCurves(hist([8.4, 8.4, 8.4, 8.4], [6.9, 6.9, 6.9, 6.9]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'diverged');
  assert.ok(d.findings.some((f) => f.key === 'f.worseThanBaseline'));
});

test('negligible absolute drop near zero is not "still improving"', () => {
  const d = diagnoseCurves(hist([0.4, 0.05, 0.005, 0.004, 0.003, 0.003], [0.4, 0.05, 0.006, 0.006, 0.005, 0.004]), { baselineLoss: 0.69 });
  assert.equal(d.verdict, 'good');
  assert.equal(d.stillImproving, false);
});

test('a large train/val gap with near-zero train loss is overfitting even if val plateaus', () => {
  const train = [2.2, 1.0, 0.3, 0.05, 0.01, 0.002, 0.001, 0.001];
  const val = [2.3, 1.2, 0.6, 0.4, 0.36, 0.37, 0.37, 0.37];
  const d = diagnoseCurves(hist(train, val), { baselineLoss: Math.log(10) });
  assert.equal(d.verdict, 'overfitting');
  assert.ok(d.findings.some((f) => f.key === 'f.bigGap'));
});

test('a moderate gap while both losses are still falling stays good', () => {
  const train = [2.0, 1.0, 0.5, 0.3, 0.2];
  const val = [2.1, 1.1, 0.65, 0.48, 0.38];
  assert.equal(diagnoseCurves(hist(train, val), { baselineLoss: Math.log(10) }).verdict, 'good');
});

test('loss that exploded far above the baseline at some point is diverged even if it settles', () => {
  const train = [45.7, 14.1, 3.8, 1.7, 0.31, 0.22, 0.24, 0.29, 0.22];
  const val = [45.4, 14.3, 3.7, 1.75, 0.3, 0.22, 0.23, 0.3, 0.22];
  const d = diagnoseCurves(hist(train, val), { baselineLoss: 0.217 });
  assert.equal(d.verdict, 'diverged');
  assert.ok(d.findings.some((f) => f.key === 'f.spiked'));
});
