// Reads train/validation loss curves and answers "did this train well?",
// with the evidence behind the answer. Finding/advice keys are i18n keys.

const finite = (x) => Number.isFinite(x);

export function diagnoseCurves(history, { baselineLoss }) {
  const n = history.length;
  const train = history.map((h) => h.trainLoss);
  const val = history.map((h) => h.valLoss);
  const result = { verdict: 'good', stillImproving: false, bestEpoch: 0, findings: [], advice: [] };

  if (n && (history.some((h) => !finite(h.trainLoss) || !finite(h.valLoss)) || train[n - 1] > 3 * train[0])) {
    const bad = history.findIndex((h) => !finite(h.trainLoss) || !finite(h.valLoss));
    result.verdict = 'diverged';
    result.findings.push(bad >= 0
      ? { level: 'bad', key: 'f.lossNaN', vars: { epoch: bad + 1 } }
      : { level: 'bad', key: 'f.lossExploded', vars: { from: round(train[0]), to: round(train[n - 1]) } });
    result.advice = ['adv.lowerLr'];
    return result;
  }

  if (n && train[n - 1] > 3 * baselineLoss) {
    result.verdict = 'diverged';
    result.findings.push({ level: 'bad', key: 'f.worseThanBaseline', vars: { loss: round(train[n - 1]), baseline: round(baselineLoss) } });
    result.advice = ['adv.lowerLr'];
    return result;
  }

  if (n < 3) {
    result.verdict = 'insufficient';
    result.findings.push({ level: 'info', key: 'f.tooFewEpochs', vars: { n } });
    result.advice = ['adv.trainMore'];
    return result;
  }

  const last = n - 1;
  let m = 0;
  for (let i = 1; i < n; i++) if (val[i] < val[m]) m = i;
  result.bestEpoch = m + 1;

  // Overfitting needs a real rise: 10% relative AND at least 3% of the baseline in absolute terms,
  // so jitter around a near-zero loss is not mistaken for it.
  const rise = val[last] - val[m];
  if (val[last] > 1.1 * val[m] && rise > 0.03 * baselineLoss && last - m >= 3 && train[last] < train[m]) {
    result.verdict = 'overfitting';
    result.findings.push({
      level: 'bad', key: 'f.valRoseAfter',
      vars: { epoch: m + 1, pct: Math.round(((val[last] - val[m]) / val[m]) * 100) },
    });
    result.findings.push({ level: 'warn', key: 'f.trainKeptFalling' });
    result.advice = ['adv.earlyStop', 'adv.regularize', 'adv.moreData'];
    return result;
  }

  // Memorized the training set: train loss far below validation loss, and validation has stopped improving.
  const gap = val[last] - train[last];
  const valFlat = n >= 4 && val[last] >= 0.98 * val[last - 3];
  if (gap > 0.12 * baselineLoss && train[last] < 0.25 * val[last] && valFlat) {
    result.verdict = 'overfitting';
    result.findings.push({ level: 'bad', key: 'f.bigGap', vars: { train: round(train[last]), val: round(val[last]), gapPct: Math.round((gap / baselineLoss) * 100) } });
    result.findings.push({ level: 'warn', key: 'f.valStalled' });
    result.advice = ['adv.earlyStop', 'adv.regularize', 'adv.moreData'];
    return result;
  }

  const improvement = (train[0] - train[last]) / train[0];
  if (train[last] > 0.8 * baselineLoss || improvement < 0.1) {
    result.verdict = 'underfitting';
    result.findings.push({
      level: 'warn', key: 'f.nearBaseline',
      vars: { loss: round(train[last]), baseline: round(baselineLoss), pct: Math.round(improvement * 100) },
    });
    result.advice = ['adv.biggerModel', 'adv.trainMore', 'adv.raiseLr'];
  } else {
    result.findings.push({ level: 'good', key: 'f.beatsBaseline', vars: { ratio: baselineLoss / val[last] } });
    // Gap measured against the baseline so it stays meaningful when losses get tiny.
    const gapPct = Math.round(((val[last] - train[last]) / baselineLoss) * 100);
    result.findings.push(gapPct <= 10
      ? { level: 'good', key: 'f.gapSmall', vars: { gapPct } }
      : { level: 'warn', key: 'f.gapLarge', vars: { gapPct } });
  }

  result.stillImproving = n >= 4 && val[last] < 0.98 * val[last - 3] && val[last - 3] - val[last] > 0.01 * baselineLoss;
  if (result.stillImproving && !result.advice.includes('adv.trainMore')) {
    result.findings.push({ level: 'info', key: 'f.stillImproving' });
    result.advice.push('adv.trainMore');
  }
  return result;
}

function round(x) {
  return Math.round(x * 1000) / 1000;
}
