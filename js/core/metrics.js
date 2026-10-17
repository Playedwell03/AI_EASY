export function accuracy(yTrue, yPred) {
  let ok = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) ok++;
  return yTrue.length ? ok / yTrue.length : 0;
}

// Rows = true class, columns = predicted class.
export function confusionMatrix(yTrue, yPred, k) {
  const m = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) m[yTrue[i]][yPred[i]]++;
  return m;
}

// Group predictions by stated confidence and compare to how often they were right.
export function reliabilityBins(probs, correct, nBins = 10) {
  const bins = Array.from({ length: nBins }, (_, b) => ({
    lo: b / nBins, hi: (b + 1) / nBins, count: 0, avgConf: 0, accuracy: 0,
  }));
  for (let i = 0; i < probs.length; i++) {
    const b = Math.min(nBins - 1, Math.floor(probs[i] * nBins));
    bins[b].count++;
    bins[b].avgConf += probs[i];
    bins[b].accuracy += correct[i] ? 1 : 0;
  }
  for (const bin of bins) {
    if (bin.count) {
      bin.avgConf /= bin.count;
      bin.accuracy /= bin.count;
    }
  }
  return bins;
}

// Expected calibration error: count-weighted gap between confidence and accuracy.
export function ece(probs, correct, nBins = 10) {
  if (!probs.length) return 0;
  let sum = 0;
  for (const b of reliabilityBins(probs, correct, nBins)) {
    sum += b.count * Math.abs(b.avgConf - b.accuracy);
  }
  return sum / probs.length;
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

export function mae(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
}

export function perplexity(meanCrossEntropyNats) {
  return Math.exp(meanCrossEntropyNats);
}
