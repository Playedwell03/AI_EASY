// Synthetic time series (trend + daily-like seasonality + noise) and sliding windows.
import { mulberry32, randn } from '../../core/rng.js';

export const PERIOD = 24;

export function makeSeries(len, { trend, season, noise }, rand) {
  const phase = rand() * Math.PI * 2;
  const out = [];
  for (let t = 0; t < len; t++) {
    const s = Math.sin((2 * Math.PI * t) / PERIOD + phase) + 0.4 * Math.sin((4 * Math.PI * t) / PERIOD + phase * 2);
    out.push(trend * (2 * t) / len + season * s + noise * randn(rand));
  }
  return out;
}

// Each window of w consecutive values is paired with the value right after it (at index t).
export function windows(series, w) {
  const x = [];
  const y = [];
  const t = [];
  for (let i = 0; i + w < series.length; i++) {
    x.push(series.slice(i, i + w));
    y.push(series[i + w]);
    t.push(i + w);
  }
  return { x, y, t };
}

// Time-ordered split: targets in the first 70% train, next 15% validate, last 15% test. No shuffling.
export function makeData(hp) {
  const rand = mulberry32(hp.seed);
  const raw = makeSeries(hp.len, { trend: hp.trend, season: hp.season, noise: hp.noise }, rand);
  const trainEnd = Math.floor(hp.len * 0.7);
  const valEnd = Math.floor(hp.len * 0.85);
  const head = raw.slice(0, trainEnd);
  const mean = head.reduce((a, b) => a + b, 0) / head.length;
  const std = Math.sqrt(head.reduce((a, b) => a + (b - mean) ** 2, 0) / head.length) || 1;
  const series = raw.map((v) => (v - mean) / std);
  const all = windows(series, hp.window);
  const parts = { train: { x: [], y: [], t: [] }, val: { x: [], y: [], t: [] }, test: { x: [], y: [], t: [] } };
  all.t.forEach((ti, i) => {
    const p = ti < trainEnd ? parts.train : ti < valEnd ? parts.val : parts.test;
    p.x.push(all.x[i]);
    p.y.push(all.y[i]);
    p.t.push(ti);
  });
  return {
    ...parts, series, raw, mean, std, trainEnd, valEnd,
    meta: { counts: { train: parts.train.x.length, val: parts.val.x.length, test: parts.test.x.length } },
  };
}

// Mean squared / absolute error of predicting "same as the last value in the window".
export function naiveErrors(part) {
  let se = 0;
  let ae = 0;
  part.x.forEach((w, i) => {
    const e = part.y[i] - w[w.length - 1];
    se += e * e;
    ae += Math.abs(e);
  });
  return { mse: se / part.x.length, mae: ae / part.x.length };
}
