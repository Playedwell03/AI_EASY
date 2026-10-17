// Module 1 — Basic neural network (MLP) on 2D point classification.
import { h, pct } from '../../ui/dom.js';
import { scatter } from '../../ui/chart.js';
import { makeData } from './data.js';
import { createSession, architecture } from './model.js';
import { useCases, USE_TEXT } from './uses.js';

const GRID = 41;

const toPoints = (part, extra = {}) => part.x.map(([x, y], i) => ({ x, y, c: part.y[i], ...extra }));

function plane() {
  const canvas = h('canvas', { style: { maxWidth: '420px', aspectRatio: '1 / 1', borderRadius: '10px', cursor: 'crosshair' } });
  return { el: h('div', { class: 'chart-wrap', style: { maxWidth: '420px' } }, canvas), canvas };
}

export default {
  id: 'mlp',
  title: { ko: '기초 신경망 (MLP)', en: 'Basic neural network (MLP)' },
  summary: {
    ko: '평면 위의 점을 두 색으로 나누는 문제로 뉴런, 층, 손실, 학습률의 기본을 익힙니다.',
    en: 'Split points on a plane into two colors to learn neurons, layers, loss, and learning rate.',
  },
  archHint: 'Input(2) → Dense × N → Sigmoid(1)',
  synthetic: true,
  defaults: { seed: 1, dataset: 'circle', n: 400, noise: 0.1, layers: 2, units: 8, activation: 'relu', lr: 0.03, epochs: 60, batchSize: 32, l2: 0 },
  dataKeys: ['seed', 'dataset', 'n', 'noise'],
  controls: [
    { key: 'layers', type: 'range', min: 1, max: 4, label: 'c.layers', where: 'arch' },
    { key: 'units', type: 'range', min: 2, max: 32, label: 'c.units', where: 'arch' },
    {
      key: 'activation', type: 'select', label: 'c.activation', where: 'arch',
      options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'tanh' }, { value: 'sigmoid', label: 'sigmoid' }],
    },
    {
      key: 'dataset', type: 'select', label: 'c.dataset', where: 'data',
      options: [
        { value: 'circle', label: 'ds.circle' }, { value: 'xor', label: 'ds.xor' },
        { value: 'moons', label: 'ds.moons' }, { value: 'spiral', label: 'ds.spiral' },
      ],
    },
    { key: 'n', type: 'range', min: 40, max: 800, step: 20, label: 'c.n', where: 'data' },
    { key: 'noise', type: 'range', min: 0, max: 0.6, step: 0.05, label: 'c.noise', where: 'data' },
    { key: 'lr', type: 'range', min: -3, max: 1, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 5, max: 300, step: 5, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
    { key: 'l2', type: 'range', min: 0, max: 0.02, step: 0.001, label: 'c.l2', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { n: 60, noise: 0.35, layers: 4, units: 32, epochs: 300, lr: 0.01 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 5 }, expect: 'diverged' },
    { id: 'tiny', label: 'p.tiny', hp: { dataset: 'spiral', layers: 1, units: 2, epochs: 40 }, expect: 'underfitting' },
  ],
  metrics: [
    { key: 'train_acc', label: 'metric.trainAccuracy', format: 'pct' },
    { key: 'acc', label: 'metric.valAccuracy', format: 'pct' },
  ],
  testMetrics: [
    { key: 'acc', label: 'metric.testAccuracy', format: 'pct', sub: 'm.testAccSub' },
    { key: 'loss', label: 'metric.testLoss', format: 'num' },
  ],
  baselineLoss: () => Math.log(2),
  architecture,
  makeData,
  createSession,
  useCases,

  renderData(el, data, ctx) {
    const p = plane();
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      p.el,
      h('p', { class: 'small muted', style: { marginTop: '8px' } }, ctx.t('m.dataLegend')));
    scatter(p.canvas, [...toPoints(data.train), ...toPoints(data.val, { alpha: 0.35 })]);
  },

  renderTrainViz(el, ctx) {
    const p = plane();
    el.append(h('p', { class: 'help' }, ctx.t('m.vizIntro')), p.el);
    const chart = scatter(p.canvas, toPoints(ctx.data.train));
    return {
      update() {
        const s = ctx.session;
        chart.update({ field: s ? { n: GRID, values: s.grid(GRID) } : null });
      },
    };
  },

  async diagnosePanel(el, ctx, history) {
    const last = history[history.length - 1].metrics;
    const gap = last.train_acc - last.acc;
    const findings = [{ level: 'info', key: 'm.accPair', vars: { train: pct(last.train_acc), val: pct(last.acc) } }];
    if (gap > 0.1) findings.push({ level: 'warn', key: 'm.accGap', vars: { gap: pct(gap) } });
    if (last.acc < 0.75) findings.push({ level: 'bad', key: 'm.lowAcc' });
    return findings;
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const test = ctx.data.test;
    const probs = s.predictPoints(test.x);
    const pts = toPoints(test).map((p, i) => ({ ...p, ring: (probs[i] >= 0.5 ? 1 : 0) !== p.c }));
    const wrong = pts.filter((p) => p.ring).length;
    const p = plane();
    const readout = h('div', { class: 'stat', style: { maxWidth: '420px' } },
      h('div', { class: 'label' }, ctx.t('m.clickHint')), h('div', { class: 'value' }, '—'));
    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('test.try')),
      h('p', { class: 'help' }, ctx.t('m.testIntro', { wrong, total: pts.length })),
      p.el, h('div', { style: { marginTop: '10px' } }, readout)));
    const chart = scatter(p.canvas, pts, { field: { n: GRID, values: s.grid(GRID) } });
    p.canvas.addEventListener('click', (ev) => {
      const pt = chart.toData(ev);
      const prob = s.predictPoints([[pt.x, pt.y]])[0];
      chart.update({ marker: pt });
      readout.lastChild.textContent = ctx.t('m.clickResult', { p1: pct(prob), p0: pct(1 - prob) });
    });
  },

  text: {
    ko: {
      ...USE_TEXT.ko,
      'c.layers': '은닉층 수',
      'c.units': '층마다 뉴런 수',
      'c.activation': '활성화 함수',
      'c.dataset': '데이터 모양',
      'c.n': '점 개수',
      'c.noise': '노이즈',
      'c.l2': 'L2 규제(가중치 벌점)',
      'ds.circle': '원 (안쪽/바깥쪽)',
      'ds.xor': 'XOR (대각선 사분면)',
      'ds.moons': '두 개의 반달',
      'ds.spiral': '나선 (어려움)',
      'p.overfit': '점은 적고 모델은 크게 → 과적합',
      'p.diverge': '학습률 5 → 발산',
      'p.tiny': '나선 데이터에 뉴런 2개 → 학습 부족',
      'm.dataIntro': '각 점은 (x, y) 좌표 두 개로 이루어져 있고, 색이 정답(클래스)입니다. 모델은 좌표만 보고 색을 맞혀야 합니다.',
      'm.dataLegend': '진한 점: 학습용 데이터 · 흐린 점: 검증용 데이터',
      'm.vizIntro': '배경색은 모델이 평면의 각 위치를 어느 색이라고 생각하는지 보여 줍니다(결정 경계). 학습이 진행될수록 경계가 점들의 모양을 따라갑니다.',
      'm.accPair': '정확도: 학습용 {train}, 검증용 {val}',
      'm.accGap': '학습 정확도가 검증 정확도보다 {gap} 높습니다. 학습 데이터를 외우고 있을 수 있습니다.',
      'm.lowAcc': '검증 정확도가 75%보다 낮습니다. 무작위로 찍으면 50%입니다.',
      'm.testAccSub': '무작위로 찍으면 50%',
      'm.testIntro': '테스트용 점 {total}개 중 {wrong}개를 틀렸습니다(빨간 테두리). 평면 아무 곳이나 눌러 그 위치에 대한 모델의 예측을 확인해 보세요.',
      'm.clickHint': '평면을 눌러 보세요',
      'm.clickResult': '주황 {p1} · 파랑 {p0}',
    },
    en: {
      ...USE_TEXT.en,
      'c.layers': 'Hidden layers',
      'c.units': 'Neurons per layer',
      'c.activation': 'Activation',
      'c.dataset': 'Data shape',
      'c.n': 'Number of points',
      'c.noise': 'Noise',
      'c.l2': 'L2 regularization (weight penalty)',
      'ds.circle': 'Circle (inside/outside)',
      'ds.xor': 'XOR (diagonal quadrants)',
      'ds.moons': 'Two moons',
      'ds.spiral': 'Spiral (hard)',
      'p.overfit': 'Few points, big model → overfitting',
      'p.diverge': 'Learning rate 5 → diverges',
      'p.tiny': '2 neurons on a spiral → underfitting',
      'm.dataIntro': 'Each point is two coordinates (x, y); its color is the answer (class). The model must guess the color from the coordinates alone.',
      'm.dataLegend': 'Solid dots: training data · faint dots: validation data',
      'm.vizIntro': 'The background shows which color the model thinks each spot on the plane is (its decision boundary). As training goes on, the boundary follows the shape of the points.',
      'm.accPair': 'Accuracy: train {train}, validation {val}',
      'm.accGap': 'Training accuracy is {gap} higher than validation accuracy. It may be memorizing the training data.',
      'm.lowAcc': 'Validation accuracy is below 75%. Random guessing gets 50%.',
      'm.testAccSub': 'Random guessing: 50%',
      'm.testIntro': 'It got {wrong} of {total} test points wrong (red rings). Click anywhere on the plane to see the model\'s prediction for that spot.',
      'm.clickHint': 'Click the plane',
      'm.clickResult': 'Orange {p1} · Blue {p0}',
    },
  },
};
