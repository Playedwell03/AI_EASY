// Module 7 — Time series forecasting with an LSTM or 1D convolution.
import { h, fmt, pct } from '../../ui/dom.js';
import { lineChart, chartBox } from '../../ui/chart.js';
import { mulberry32 } from '../../core/rng.js';
import { makeData, makeSeries, windows, naiveErrors } from './data.js';
import { createSession, architecture } from './model.js';
import { useCases, USE_TEXT } from './uses.js';

// Values at the given indices, NaN elsewhere (so a line only covers that part of the axis).
const only = (series, from, to) => series.map((v, i) => (i >= from && i < to ? v : NaN));

export default {
  id: 'timeseries',
  title: { ko: '시계열 예측', en: 'Time series forecasting' },
  summary: {
    ko: '추세와 주기가 있는 데이터를 보고 다음 값을 예측합니다. "그냥 직전 값을 따라 하기"보다 정말 나은지 확인하는 것이 핵심입니다.',
    en: 'Predict the next value of data with a trend and a cycle. The key question: is it really better than just repeating the last value?',
  },
  archHint: 'Window(24) → LSTM | Conv1D → Dense(1)',
  synthetic: true,
  preferBackend: 'cpu',
  defaults: { seed: 1, len: 600, trend: 1, season: 1, noise: 0.25, window: 24, arch: 'lstm', units: 16, lr: 0.01, epochs: 30, batchSize: 32 },
  dataKeys: ['seed', 'len', 'trend', 'season', 'noise', 'window'],
  controls: [
    { key: 'arch', type: 'select', label: 'c.arch', where: 'arch', options: [{ value: 'lstm', label: 'LSTM' }, { value: 'conv1d', label: 'Conv1D' }] },
    { key: 'units', type: 'select', label: 'c.units', where: 'arch', options: [4, 8, 16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    { key: 'window', type: 'range', min: 8, max: 48, step: 4, label: 'c.window', where: 'arch' },
    { key: 'len', type: 'range', min: 120, max: 1200, step: 40, label: 'c.len', where: 'data' },
    { key: 'noise', type: 'range', min: 0, max: 1, step: 0.05, label: 'c.noise', where: 'data' },
    { key: 'trend', type: 'range', min: 0, max: 3, step: 0.25, label: 'c.trend', where: 'data' },
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 3, max: 200, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { len: 140, noise: 0.5, arch: 'conv1d', units: 64, epochs: 150 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 3 }, expect: 'diverged' },
  ],
  metrics: [{ key: 'mae', label: 'm.valMae', format: 'num' }],
  testMetrics: [
    { key: 'mae', label: 'metric.mae', format: 'num', sub: 'm.maeSub' },
    { key: 'naiveMae', label: 'm.naiveMae', format: 'num' },
    { key: 'mse', label: 'metric.mse', format: 'num' },
  ],
  baselineLoss: (hp, data) => naiveErrors(data.val).mse,
  baselineLabel: 'm.baseline',
  dataWhy: 'm.dataWhy',
  architecture,
  makeData,
  createSession,
  useCases,

  renderData(el, data, ctx) {
    const box = chartBox();
    el.append(h('p', { class: 'help' }, ctx.t('m.dataIntro')), box.el);
    const n = data.raw.length;
    lineChart(box.canvas, { xLabel: ctx.t('m.time'), xStart: 0, height: 240 }).update([
      { name: ctx.t('data.train'), values: only(data.raw, 0, data.trainEnd), color: '--series-1' },
      { name: ctx.t('data.val'), values: only(data.raw, data.trainEnd, data.valEnd), color: '--series-2' },
      { name: ctx.t('data.test'), values: only(data.raw, data.valEnd, n), color: '--series-3' },
    ]);
  },

  renderTrainViz(el, ctx) {
    const box = chartBox();
    el.append(h('p', { class: 'help' }, ctx.t('m.vizIntro')), box.el);
    const { val, series } = ctx.data;
    const from = val.t[0] - ctx.hp.window;
    const actual = series.slice(from, val.t.at(-1) + 1);
    const chart = lineChart(box.canvas, { xLabel: ctx.t('m.time'), xStart: from, height: 220 });
    return {
      update() {
        const s = ctx.session;
        const pred = new Array(actual.length).fill(NaN);
        if (s) s.predictWindows(val.x).forEach((p, i) => { pred[val.t[i] - from] = p; });
        chart.update([
          { name: ctx.t('chart.truth'), values: actual, color: '--series-1' },
          { name: ctx.t('m.pred1'), values: pred, color: '--series-2' },
        ]);
      },
    };
  },

  async diagnosePanel(el, ctx, history) {
    const last = history.at(-1).metrics;
    const naive = naiveErrors(ctx.data.val);
    const ratio = last.mae / naive.mae;
    const box = chartBox();
    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('m.vsNaive')),
      h('p', { class: 'help' }, ctx.t('m.vsNaiveHelp')),
      box.el));
    lineChart(box.canvas, { xLabel: ctx.t('train.epoch'), height: 200 }).update([
      { name: ctx.t('m.valMae'), values: history.map((x) => x.metrics.mae), color: '--series-2' },
      { name: ctx.t('m.naiveMae'), values: history.map(() => naive.mae), color: '--muted', dashed: true },
    ]);
    if (ratio < 0.8) return [{ level: 'good', key: 'm.beatsNaive', vars: { pct: pct(1 - ratio) } }];
    if (ratio < 1) return [{ level: 'warn', key: 'm.barelyBeatsNaive', vars: { pct: pct(1 - ratio) } }];
    return [{ level: 'bad', key: 'm.worseThanNaive', vars: { m: fmt(last.mae), n: fmt(naive.mae) } }];
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const { hp, data } = ctx;
    const { test, series } = data;
    const from = test.t[0] - hp.window;
    const actual = series.slice(from);

    // One-step predictions on the test range.
    const oneStep = box();
    const pred = new Array(actual.length).fill(NaN);
    s.predictWindows(test.x).forEach((p, i) => { pred[test.t[i] - from] = p; });

    // Autoregressive forecast from the start of the test range.
    const multi = box();
    const steps = Math.min(30, test.x.length);
    const fc = new Array(actual.length).fill(NaN);
    s.forecast(test.x[0], steps).forEach((p, i) => { fc[hp.window + i] = p; });

    // A fresh series with adjustable noise, normalized with the training statistics.
    const fresh = box();
    const noiseVal = h('span', { class: 'val' }, String(hp.noise));
    const freshStats = h('p', { class: 'small muted' });
    const slider = h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: hp.noise });

    el.append(
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.oneStep')), h('p', { class: 'help' }, ctx.t('m.oneStepHelp')), oneStep.el),
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.multi')), h('p', { class: 'help' }, ctx.t('m.multiHelp', { n: steps })), multi.el),
      h('div', { class: 'card' }, h('h3', null, ctx.t('test.try')), h('p', { class: 'help' }, ctx.t('m.freshHelp')),
        h('div', { class: 'control', style: { maxWidth: '320px' } }, h('label', null, h('span', null, ctx.t('c.noise')), noiseVal), slider),
        fresh.el, freshStats));

    lineChart(oneStep.canvas, { xLabel: ctx.t('m.time'), xStart: from, height: 220 }).update([
      { name: ctx.t('chart.truth'), values: actual, color: '--series-1' },
      { name: ctx.t('m.pred1'), values: pred, color: '--series-2' },
    ]);
    lineChart(multi.canvas, { xLabel: ctx.t('m.time'), xStart: from, height: 220 }).update([
      { name: ctx.t('chart.truth'), values: actual.slice(0, hp.window + steps), color: '--series-1' },
      { name: ctx.t('m.predMulti'), values: fc.slice(0, hp.window + steps), color: '--series-4' },
    ]);
    const freshChart = lineChart(fresh.canvas, { xLabel: ctx.t('m.time'), xStart: 0, height: 200 });
    const drawFresh = (noise) => {
      const raw = makeSeries(200, { trend: hp.trend, season: hp.season, noise }, mulberry32(hp.seed + 777));
      const norm = raw.map((v) => (v - data.mean) / data.std);
      const w = windows(norm, hp.window);
      const p = s.predictWindows(w.x);
      const predLine = new Array(norm.length).fill(NaN);
      p.forEach((v, i) => { predLine[w.t[i]] = v; });
      const mae = p.reduce((a, v, i) => a + Math.abs(v - w.y[i]), 0) / p.length;
      const naive = naiveErrors(w).mae;
      freshChart.update([
        { name: ctx.t('chart.truth'), values: norm, color: '--series-1' },
        { name: ctx.t('m.pred1'), values: predLine, color: '--series-2' },
      ]);
      freshStats.textContent = ctx.t('m.freshStats', { m: fmt(mae), n: fmt(naive) });
    };
    slider.addEventListener('input', () => { noiseVal.textContent = slider.value; drawFresh(Number(slider.value)); });
    drawFresh(hp.noise);

    function box() { return chartBox(); }
  },

  text: {
    ko: {
      ...USE_TEXT.ko,
      'c.arch': '모델 종류',
      'c.units': '유닛(필터) 수',
      'c.window': '창 크기 (몇 개를 보고 예측할지)',
      'c.len': '시계열 길이',
      'c.noise': '노이즈',
      'c.trend': '추세 세기',
      'p.overfit': '짧고 시끄러운 데이터에 큰 모델 → 과적합',
      'p.diverge': '학습률 3 → 발산',
      'm.valMae': '검증 MAE',
      'm.naiveMae': '나이브 예측 MAE',
      'm.maeSub': '나이브 예측보다 낮아야 의미가 있음',
      'm.baseline': '기준선: 직전 값 그대로 예측(나이브)',
      'm.time': '시간',
      'm.pred1': '예측 (한 칸 앞)',
      'm.predMulti': '예측 (자기 예측을 이어 붙임)',
      'm.dataIntro': '시간에 따라 변하는 값입니다. 천천히 올라가는 추세, 24칸마다 반복되는 주기, 그리고 무작위 노이즈가 섞여 있습니다.',
      'm.dataWhy': '시계열은 무작위로 섞어서 나누면 안 됩니다. 미래 데이터로 과거를 맞히는 "커닝"이 되기 때문입니다. 그래서 앞쪽 70%로 학습하고, 그다음 15%로 검증하고, 가장 마지막 15%로 시험합니다. 실제로 미래를 예측하는 상황과 같습니다.',
      'm.vizIntro': '검증 구간의 실제 값과 모델의 한 칸 앞 예측입니다. 학습이 진행될수록 주황 선이 파란 선의 모양을 따라갑니다.',
      'm.vsNaive': '나이브 예측과 비교',
      'm.vsNaiveHelp': '시계열에서 가장 중요한 비교입니다. "다음 값은 지금 값과 같다"고 찍기만 해도 꽤 잘 맞습니다. 모델이 이것보다 확실히 낫지 않다면, 아무리 손실이 내려가도 쓸모가 없습니다.',
      'm.beatsNaive': '검증 MAE가 나이브 예측보다 {pct} 낮습니다. 단순히 따라 하는 것 이상을 배웠습니다.',
      'm.barelyBeatsNaive': '나이브 예측보다 {pct}밖에 낫지 않습니다. 패턴을 충분히 배우지 못했을 수 있습니다.',
      'm.worseThanNaive': '모델의 검증 MAE({m})가 직전 값을 그대로 쓰는 나이브 예측({n})보다 나쁩니다. 이 모델은 쓸모가 없습니다.',
      'm.oneStep': '한 칸 앞 예측 (테스트 구간)',
      'm.oneStepHelp': '매번 실제 과거 값을 보고 바로 다음 값 하나만 예측합니다.',
      'm.multi': '여러 칸 앞 예측',
      'm.multiHelp': '자기가 예측한 값을 다시 입력으로 넣으며 {n}칸 앞까지 내다봅니다. 오차가 쌓이기 때문에 멀리 갈수록 실제와 벌어지기 쉽습니다.',
      'm.freshHelp': '같은 규칙으로 새 시계열을 만들어 예측해 봅니다. 노이즈를 키우면 예측이 얼마나 어려워지는지 보세요.',
      'm.freshStats': '모델 MAE {m} · 나이브 MAE {n}',
    },
    en: {
      ...USE_TEXT.en,
      'c.arch': 'Model type',
      'c.units': 'Units (filters)',
      'c.window': 'Window size (how many past values it sees)',
      'c.len': 'Series length',
      'c.noise': 'Noise',
      'c.trend': 'Trend strength',
      'p.overfit': 'Big model on short, noisy data → overfitting',
      'p.diverge': 'Learning rate 3 → diverges',
      'm.valMae': 'Validation MAE',
      'm.naiveMae': 'Naive forecast MAE',
      'm.maeSub': 'Only meaningful if lower than the naive forecast',
      'm.baseline': 'Baseline: repeat the last value (naive)',
      'm.time': 'Time',
      'm.pred1': 'Prediction (one step ahead)',
      'm.predMulti': 'Prediction (fed its own predictions)',
      'm.dataIntro': 'A value that changes over time: a slowly rising trend, a cycle that repeats every 24 steps, and random noise.',
      'm.dataWhy': 'Never shuffle a time series before splitting — that would let the model peek at the future to guess the past. Instead the first 70% trains, the next 15% validates, and the final 15% tests. That matches really forecasting the future.',
      'm.vizIntro': 'Actual values in the validation range, with the model\'s one-step-ahead predictions. As training goes on, the orange line follows the shape of the blue one.',
      'm.vsNaive': 'Compared with the naive forecast',
      'm.vsNaiveHelp': 'The most important check for time series. Just guessing "the next value equals the current one" already works fairly well. If the model is not clearly better than that, it is useless no matter how low its loss goes.',
      'm.beatsNaive': 'Validation MAE is {pct} lower than the naive forecast. It learned more than just copying.',
      'm.barelyBeatsNaive': 'Only {pct} better than the naive forecast. It may not have learned the pattern well.',
      'm.worseThanNaive': 'The model\'s validation MAE ({m}) is worse than simply repeating the last value ({n}). This model is useless.',
      'm.oneStep': 'One-step-ahead forecast (test range)',
      'm.oneStepHelp': 'Each time it sees the real past values and predicts only the next one.',
      'm.multi': 'Forecasting several steps ahead',
      'm.multiHelp': 'It feeds its own predictions back in to look {n} steps ahead. Errors pile up, so the forecast tends to drift from reality the further it goes.',
      'm.freshHelp': 'A new series made by the same rules. Turn up the noise and see how much harder forecasting gets.',
      'm.freshStats': 'Model MAE {m} · naive MAE {n}',
    },
  },
};
