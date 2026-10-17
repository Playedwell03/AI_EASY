// Module 2 — Image classification with a small CNN on synthetic handwritten digits.
import { h, drawGray, probBars } from '../../ui/dom.js';
import { heatmap } from '../../ui/chart.js';
import { confusionMatrix } from '../../core/metrics.js';
import { DIGIT_SIZE as S, frameDrawing } from '../../shared/digits.js';
import { makeData, createSession, architecture } from './model.js';

const DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

export function imgCell(pixels, caption, { wrong = false, size = S } = {}) {
  return h('div', { class: `img-cell${wrong ? ' wrong' : ''}` }, drawGray(h('canvas'), pixels, size), caption);
}

// A mouse/touch drawing pad. onChange(imageData) fires after each stroke.
export function drawPad(onChange, { size = 168, brush = 13 } = {}) {
  const canvas = h('canvas', { width: size, height: size, style: { width: `${size}px`, height: `${size}px`, background: '#000', borderRadius: '10px', touchAction: 'none', cursor: 'crosshair' } });
  const ctx = canvas.getContext('2d');
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = brush;
  let drawing = false;
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * (size / r.width), (e.clientY - r.top) * (size / r.height)]; };
  canvas.addEventListener('pointerdown', (e) => { drawing = true; canvas.setPointerCapture(e.pointerId); const [x, y] = pos(e); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.1, y); ctx.stroke(); });
  canvas.addEventListener('pointermove', (e) => { if (!drawing) return; const [x, y] = pos(e); ctx.lineTo(x, y); ctx.stroke(); });
  const end = () => { if (!drawing) return; drawing = false; onChange(ctx.getImageData(0, 0, size, size)); };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointerleave', end);
  return { canvas, clear() { ctx.clearRect(0, 0, size, size); } };
}

export default {
  id: 'cnn',
  title: { ko: '이미지 분류 (CNN)', en: 'Image classification (CNN)' },
  summary: {
    ko: '손글씨 숫자 이미지를 합성곱 신경망으로 분류합니다. 필터가 무엇을 보는지 직접 확인하고, 숫자를 그려서 시험해 봅니다.',
    en: 'Classify handwritten digit images with a convolutional network. See what its filters look at, and draw your own digits to test it.',
  },
  archHint: 'Conv → Pool → Conv → Pool → Dense → Softmax(10)',
  synthetic: true,
  defaults: { seed: 1, n: 2000, filters: 8, dense: 32, dropout: 0, lr: 0.003, epochs: 6, batchSize: 32 },
  cpuDefaults: { n: 1000, epochs: 6, lr: 0.006 },
  dataKeys: ['seed', 'n'],
  controls: [
    { key: 'filters', type: 'select', label: 'c.filters', where: 'arch', options: [4, 8, 16].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dense', type: 'select', label: 'c.dense', where: 'arch', options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dropout', type: 'select', label: 'c.dropout', where: 'arch', options: [0, 0.25, 0.5].map((v) => ({ value: v, label: v ? String(v) : 'c.off' })) },
    { key: 'n', type: 'range', min: 200, max: 5000, step: 100, label: 'c.n', where: 'data' },
    { key: 'lr', type: 'range', min: -4, max: 0, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 1, max: 60, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { n: 200, epochs: 30, dense: 64 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 1 }, expect: 'diverged' },
  ],
  metrics: [
    { key: 'train_acc', label: 'metric.trainAccuracy', format: 'pct' },
    { key: 'acc', label: 'metric.valAccuracy', format: 'pct' },
  ],
  testMetrics: [
    { key: 'acc', label: 'metric.testAccuracy', format: 'pct', sub: 'm.chance' },
    { key: 'loss', label: 'metric.testLoss', format: 'num' },
  ],
  baselineLoss: () => Math.log(10),
  architecture,
  makeData,
  createSession,

  renderData(el, data, ctx) {
    const byClass = DIGITS.map((_, d) => data.train.x.filter((__, i) => data.train.y[i] === d).slice(0, 3));
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(10, minmax(0, 1fr))' } },
        [0, 1, 2].flatMap((row) => byClass.map((imgs, d) => imgCell(imgs[row], row === 2 ? String(d) : '')))));
  },

  renderTrainViz(el, ctx) {
    const sample = ctx.data.val.x[3];
    const filtersEl = h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(8, minmax(0, 1fr))' } });
    const mapsEl = h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(8, minmax(0, 1fr))' } });
    el.append(
      h('div', { class: 'grid-2' },
        h('div', null, h('h3', { class: 'small' }, ctx.t('m.filters')), h('p', { class: 'help' }, ctx.t('m.filtersHelp')), filtersEl),
        h('div', null, h('h3', { class: 'small' }, ctx.t('m.maps')), h('p', { class: 'help' }, ctx.t('m.mapsHelp')),
          h('div', { class: 'row', style: { alignItems: 'flex-start' } }, h('div', { style: { width: '56px' } }, imgCell(sample, ctx.t('m.input'))), h('div', { style: { flex: 1 } }, mapsEl)))));
    return {
      update() {
        const s = ctx.session;
        if (!s) { filtersEl.replaceChildren(); mapsEl.replaceChildren(); return; }
        const { filters, maps } = s.firstLayer(sample);
        filtersEl.replaceChildren(...filters.map((f, i) => {
          const lo = Math.min(...f);
          const hi = Math.max(...f);
          return imgCell(f.map((v) => (v - lo) / (hi - lo || 1)), `#${i + 1}`, { size: 3 });
        }));
        const n = maps.length;
        mapsEl.replaceChildren(...filters.map((_, fi) => {
          const px = new Float32Array(n * n);
          let hi = 1e-6;
          for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { px[y * n + x] = maps[y][x][fi]; hi = Math.max(hi, maps[y][x][fi]); }
          return imgCell(px.map((v) => v / hi), `#${fi + 1}`, { size: n });
        }));
      },
    };
  },

  async diagnosePanel(el, ctx) {
    const { val } = ctx.data;
    const probs = ctx.session.predictImages(val.x);
    const pred = probs.map((p) => p.indexOf(Math.max(...p)));
    const cm = confusionMatrix(val.y, pred, 10);
    let worst = null;
    for (let a = 0; a < 10; a++) for (let b = 0; b < 10; b++) if (a !== b && cm[a][b] && (!worst || cm[a][b] > worst.n)) worst = { a, b, n: cm[a][b] };
    const canvas = h('canvas');
    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('m.confusion')),
      h('p', { class: 'help' }, ctx.t('m.confusionHelp')),
      h('div', { class: 'chart-wrap', style: { maxWidth: '460px' } }, canvas)));
    heatmap(canvas, cm, { rowLabels: DIGITS, colLabels: DIGITS, showValues: true, height: 330 });
    return worst ? [{ level: 'info', key: 'm.worstPair', vars: worst }] : [{ level: 'good', key: 'm.noConfusion' }];
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const bars = h('div', null, h('p', { class: 'muted small' }, ctx.t('m.drawHint')));
    const framed = h('div', { style: { width: '56px' } });
    const pad = drawPad((imgData) => {
      const img = frameDrawing(imgData);
      framed.replaceChildren(imgCell(img, ctx.t('m.modelSees')));
      bars.replaceChildren(probBars(DIGITS, s.predictImages([img])[0]));
    });
    const { test } = ctx.data;
    const probs = s.predictImages(test.x);
    const wrong = [];
    probs.forEach((p, i) => { const k = p.indexOf(Math.max(...p)); if (k !== test.y[i]) wrong.push({ i, k }); });
    el.append(
      h('div', { class: 'card' },
        h('h3', null, ctx.t('test.try')),
        h('div', { class: 'grid-2' },
          h('div', { class: 'stack', style: { justifyItems: 'start' } }, pad.canvas,
            h('div', { class: 'row' }, h('button', { class: 'btn btn-sm', type: 'button', onclick: () => { pad.clear(); framed.replaceChildren(); bars.replaceChildren(h('p', { class: 'muted small' }, ctx.t('m.drawHint'))); } }, ctx.t('m.clear')), framed)),
          bars),
        h('p', { class: 'help', style: { marginTop: '10px' } }, ctx.t('m.drawNote'))),
      h('div', { class: 'card' },
        h('h3', null, ctx.t('test.mistakes')),
        h('p', { class: 'help' }, ctx.t('m.mistakesHelp', { n: wrong.length, total: test.x.length })),
        h('div', { class: 'img-grid' }, wrong.slice(0, 24).map(({ i, k }) => imgCell(test.x[i], `${test.y[i]}→${k}`, { wrong: true })))));
  },

  text: {
    ko: {
      'c.filters': '첫 합성곱층 필터 수',
      'c.dense': 'Dense 층 뉴런 수',
      'c.dropout': '드롭아웃',
      'c.off': '끔',
      'c.n': '이미지 개수',
      'p.overfit': '이미지 200장으로 30에폭 → 과적합',
      'p.diverge': '학습률 1 → 발산',
      'm.chance': '무작위로 찍으면 10%',
      'm.dataIntro': '0~9 숫자 이미지입니다. 같은 숫자라도 기울기, 크기, 굵기, 위치가 조금씩 다르게 만들어졌습니다. 모델은 이 차이에도 불구하고 숫자를 알아봐야 합니다.',
      'm.filters': '1층 필터 (3×3)',
      'm.filtersHelp': '밝은 칸은 "여기가 밝으면 반응"한다는 뜻입니다. 학습 전에는 무작위 무늬지만, 학습하면서 특정 방향의 선이나 모서리를 찾는 모양으로 바뀝니다.',
      'm.maps': '특징 지도',
      'm.mapsHelp': '왼쪽 이미지를 각 필터에 통과시킨 결과입니다. 밝은 부분이 필터가 반응한 곳입니다.',
      'm.input': '입력',
      'm.confusion': '혼동 행렬 (검증 데이터)',
      'm.confusionHelp': '행은 정답, 열은 모델의 예측입니다. 대각선에 숫자가 몰려 있을수록 좋습니다. 대각선 밖의 숫자는 "어떤 숫자를 무엇으로 착각했는지"를 보여 줍니다.',
      'm.worstPair': '가장 많이 헷갈린 경우: 정답 {a}를 {b}로 {n}번 착각했습니다.',
      'm.noConfusion': '검증 데이터에서 헷갈린 숫자가 하나도 없습니다.',
      'm.drawHint': '왼쪽 검은 칸에 숫자 하나를 그려 보세요.',
      'm.modelSees': '모델 입력',
      'm.clear': '지우기',
      'm.drawNote': '그림은 자동으로 잘라내고 가운데 맞춘 뒤 20×20으로 줄여서 모델에 넣습니다("모델 입력"). 모델은 합성된 숫자로만 배웠기 때문에, 여러분의 글씨체가 많이 다르면 틀릴 수 있습니다. 이것도 중요한 교훈입니다: 학습 데이터와 다른 데이터에서는 성능이 떨어집니다.',
      'm.mistakesHelp': '테스트 이미지 {total}장 중 {n}장을 틀렸습니다. "정답→예측" 형식으로 표시합니다.',
    },
    en: {
      'c.filters': 'Filters in first conv layer',
      'c.dense': 'Dense layer neurons',
      'c.dropout': 'Dropout',
      'c.off': 'off',
      'c.n': 'Number of images',
      'p.overfit': '200 images for 30 epochs → overfitting',
      'p.diverge': 'Learning rate 1 → diverges',
      'm.chance': 'Random guessing: 10%',
      'm.dataIntro': 'Images of digits 0–9. Each copy of a digit has a slightly different tilt, size, thickness, and position. The model has to recognize the digit despite these differences.',
      'm.filters': 'Layer-1 filters (3×3)',
      'm.filtersHelp': 'A bright cell means "respond when this spot is bright". Before training they are random; as training goes on they turn into detectors for strokes or edges in particular directions.',
      'm.maps': 'Feature maps',
      'm.mapsHelp': 'The image on the left passed through each filter. Bright areas are where the filter responded.',
      'm.input': 'Input',
      'm.confusion': 'Confusion matrix (validation data)',
      'm.confusionHelp': 'Rows are the true digit, columns the model\'s prediction. The more the counts sit on the diagonal, the better. Off-diagonal counts show which digit was mistaken for which.',
      'm.worstPair': 'Most common mix-up: a true {a} was read as {b}, {n} time(s).',
      'm.noConfusion': 'No digits were mixed up on the validation data.',
      'm.drawHint': 'Draw a single digit in the black box on the left.',
      'm.modelSees': 'Model input',
      'm.clear': 'Clear',
      'm.drawNote': 'Your drawing is cropped, centered, and shrunk to 20×20 before it goes into the model ("Model input"). The model only learned from synthetic digits, so if your handwriting looks very different it may get it wrong. That is a lesson too: performance drops on data unlike the training data.',
      'm.mistakesHelp': 'It got {n} of {total} test images wrong. Shown as "true→predicted".',
    },
  },
};
