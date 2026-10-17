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

  if (val[last] > 1.1 * val[m] && last - m >= 3 && train[last] < train[m]) {
    result.verdict = 'overfitting';
    result.findings.push({
      level: 'bad', key: 'f.valRoseAfter',
      vars: { epoch: m + 1, pct: Math.round(((val[last] - val[m]) / val[m]) * 100) },
    });
    result.findings.push({ level: 'warn', key: 'f.trainKeptFalling' });
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

  result.stillImproving = n >= 4 && val[last] < 0.98 * val[last - 3];
  if (result.stillImproving && !result.advice.includes('adv.trainMore')) {
    result.findings.push({ level: 'info', key: 'f.stillImproving' });
    result.advice.push('adv.trainMore');
  }
  return result;
}

function round(x) {
  return Math.round(x * 1000) / 1000;
}
