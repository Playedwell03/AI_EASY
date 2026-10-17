// Module 6 — Generative models: autoencoder / VAE with a 2D latent space.
import { h, fmt, drawGray } from '../../ui/dom.js';
import { scatter } from '../../ui/chart.js';
import { DIGIT_SIZE as S } from '../../shared/digits.js';
import { imgCell } from '../cnn/index.js';
import { makeData, createSession, architecture, meanImageLoss } from './model.js';

// Ten categorical colors readable on light and dark backgrounds (one per digit).
export const DIGIT_COLORS = ['#4c6ef5', '#f76707', '#37b24d', '#e64980', '#7950f2', '#15aabf', '#f59f00', '#868e96', '#d6336c', '#82c91e'];

function latentPoints(s, data, limit = 400) {
  const imgs = data.val.x.slice(0, limit);
  const z = s.encode(imgs);
  return z.map(([x, y], i) => ({ x, y, c: data.val.y[i], r: 2.6, alpha: 0.85 }));
}

function domainOf(points) {
  const m = Math.max(1, ...points.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]));
  return [-m * 1.1, m * 1.1];
}

function legend() {
  return h('div', { class: 'legend' }, DIGIT_COLORS.map((c, d) => h('span', null, h('i', { style: { background: c } }), String(d))));
}

export default {
  id: 'vae',
  title: { ko: '생성 모델 (오토인코더 / VAE)', en: 'Generative model (Autoencoder / VAE)' },
  summary: {
    ko: '이미지를 숫자 두 개로 압축했다가 다시 복원합니다. 그 두 숫자가 만드는 "잠재공간"을 돌아다니며 새로운 숫자를 만들어 봅니다.',
    en: 'Compress an image into two numbers and rebuild it. Then wander the latent space those two numbers form to generate new digits.',
  },
  archHint: 'Encoder → z (2D) → Decoder',
  synthetic: true,
  defaults: { seed: 1, n: 2000, hidden: 128, variational: true, beta: 1, lr: 0.01, epochs: 20, batchSize: 32 },
  cpuDefaults: { epochs: 15 },
  dataKeys: ['seed', 'n'],
  controls: [
    { key: 'variational', type: 'toggle', label: 'c.variational', where: 'arch' },
    { key: 'hidden', type: 'select', label: 'c.hidden', where: 'arch', options: [2, 16, 64, 128, 256].map((v) => ({ value: v, label: String(v) })) },
    { key: 'beta', type: 'select', label: 'c.beta', where: 'train', options: [0.1, 0.5, 1, 2, 5].map((v) => ({ value: v, label: String(v) })) },
    { key: 'n', type: 'range', min: 200, max: 4000, step: 100, label: 'c.n', where: 'data' },
    { key: 'lr', type: 'range', min: -4, max: 0, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 80, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 16, max: 256, step: 16, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'diverge', label: 'p.diverge', hp: { lr: 0.5 }, expect: 'diverged' },
    { id: 'tiny', label: 'p.tiny', hp: { hidden: 2, epochs: 10 }, expect: 'underfitting' },
  ],
  metrics: [
    { key: 'recon', label: 'm.recon', format: 'num' },
    { key: 'kl', label: 'm.kl', format: 'num', dashed: true },
  ],
  testMetrics: [
    { key: 'recon', label: 'm.recon', format: 'num', sub: 'm.reconSub' },
    { key: 'kl', label: 'm.kl', format: 'num' },
  ],
  baselineLoss: (hp, data) => meanImageLoss(data),
  baselineLabel: 'm.baseline',
  architecture,
  makeData,
  createSession,

  renderData(el, data, ctx) {
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(10, minmax(0, 1fr))' } },
        data.train.x.slice(0, 20).map((img) => imgCell(img, ''))));
  },

  renderTrainViz(el, ctx) {
    const pairs = h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(8, minmax(0, 1fr))' } });
    const canvas = h('canvas', { style: { aspectRatio: '1 / 1', borderRadius: '10px' } });
    el.append(h('div', { class: 'grid-2' },
      h('div', null, h('h3', { class: 'small' }, ctx.t('m.pairs')), h('p', { class: 'help' }, ctx.t('m.pairsHelp')), pairs),
      h('div', null, h('h3', { class: 'small' }, ctx.t('m.latent')), h('p', { class: 'help' }, ctx.t('m.latentHelp')),
        h('div', { class: 'chart-wrap', style: { maxWidth: '360px' } }, canvas), legend())));
    const chart = scatter(canvas, [], { colors: DIGIT_COLORS });
    const sample = ctx.data.val.x.slice(0, 8);
    return {
      update() {
        const s = ctx.session;
        if (!s) { pairs.replaceChildren(); chart.update({ points: [] }); return; }
        const rec = s.reconstruct(sample);
        pairs.replaceChildren(...sample.map((img) => imgCell(img, ctx.t('m.orig'))), ...rec.map((img) => imgCell(img, ctx.t('m.rec'))));
        const pts = latentPoints(s, ctx.data);
        chart.update({ points: pts, domain: domainOf(pts) });
      },
    };
  },

  async diagnosePanel(el, ctx, history) {
    const last = history.at(-1).metrics;
    const base = meanImageLoss(ctx.data);
    const findings = [];
    if (last.recon <= 0.5 * base) findings.push({ level: 'good', key: 'm.reconGood', vars: { r: fmt(last.recon), b: fmt(base) } });
    else findings.push({ level: 'warn', key: 'm.reconMeh', vars: { r: fmt(last.recon), b: fmt(base) } });
    if (ctx.hp.variational && last.kl < 0.01) findings.push({ level: 'warn', key: 'm.collapse' });
    if (ctx.hp.variational && last.kl >= 0.01) findings.push({ level: 'info', key: 'm.klInfo', vars: { kl: fmt(last.kl) } });
    return findings;
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const pts = latentPoints(s, ctx.data);
    const domain = domainOf(pts);
    const canvas = h('canvas', { style: { aspectRatio: '1 / 1', borderRadius: '10px', cursor: 'crosshair' } });
    const out = h('div', { style: { width: '120px' } });
    const coords = h('div', { class: 'mono small muted' }, ctx.t('m.clickLatent'));
    const chart = scatter(canvas, pts.map((p) => ({ ...p, alpha: 0.35 })), { colors: DIGIT_COLORS, domain });
    const show = (z) => {
      chart.update({ marker: { x: z[0], y: z[1] } });
      out.replaceChildren(imgCell(s.decode([z])[0], ''));
      coords.textContent = `z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
    };
    canvas.addEventListener('click', (ev) => { const p = chart.toData(ev); show([p.x, p.y]); });

    // 8×8 grid of decoded images across the latent plane.
    const G = 8;
    const zs = [];
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) zs.push([domain[0] + ((i + 0.5) / G) * (domain[1] - domain[0]), domain[1] - ((j + 0.5) / G) * (domain[1] - domain[0])]);
    const imgs = s.decode(zs);
    const grid = h('div', { style: { display: 'grid', gridTemplateColumns: `repeat(${G}, 1fr)`, gap: '2px', maxWidth: '360px' } },
      imgs.map((img) => drawGray(h('canvas', { style: { width: '100%', imageRendering: 'pixelated', display: 'block' } }), img, S)));

    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('test.try')),
      h('p', { class: 'help' }, ctx.t('m.testIntro')),
      h('div', { class: 'grid-2' },
        h('div', null, h('div', { class: 'chart-wrap', style: { maxWidth: '360px' } }, canvas), legend()),
        h('div', { class: 'stack', style: { alignContent: 'start' } }, out, coords))),
    h('div', { class: 'card' },
      h('h3', null, ctx.t('m.gridTitle')),
      h('p', { class: 'help' }, ctx.t(ctx.hp.variational ? 'm.gridHelpVae' : 'm.gridHelpAe')),
      grid));
    show([0, 0]);
  },

  text: {
    ko: {
      'c.variational': 'VAE 모드 (끄면 일반 오토인코더)',
      'c.hidden': '은닉층 뉴런 수',
      'c.beta': 'β (KL 항의 무게)',
      'c.n': '이미지 개수',
      'p.diverge': '학습률 0.5 → 발산',
      'p.tiny': '은닉층 뉴런 2개 → 학습 부족',
      'm.recon': '재구성 오차',
      'm.kl': 'KL 발산',
      'm.reconSub': '평균 이미지로 복원할 때보다 낮을수록 좋음',
      'm.baseline': '기준선: 평균 이미지로 복원했을 때',
      'm.dataIntro': '이 모듈은 정답(라벨) 없이 이미지만으로 학습합니다. 목표는 "입력 이미지를 그대로 다시 그려 내는 것"입니다. 중간에 숫자 두 개짜리 좁은 통로를 지나야 하므로, 모델은 이미지의 핵심만 요약하는 법을 배워야 합니다.',
      'm.pairs': '원본 vs 복원',
      'm.pairsHelp': '윗줄이 원본, 아랫줄이 숫자 두 개에서 복원한 이미지입니다.',
      'm.orig': '원본',
      'm.rec': '복원',
      'm.latent': '잠재공간 (검증 이미지)',
      'm.latentHelp': '각 이미지를 인코딩한 두 숫자(z₁, z₂)를 점으로 찍었습니다. 색은 실제 숫자입니다. 정답을 알려 준 적이 없는데도 같은 숫자끼리 모이면, 모델이 스스로 의미 있는 요약을 찾은 것입니다.',
      'm.reconGood': '재구성 오차({r})가 평균 이미지로 복원할 때({b})의 절반 이하입니다. 이미지의 특징을 잘 요약하고 있습니다.',
      'm.reconMeh': '재구성 오차({r})가 평균 이미지로 복원할 때({b})와 큰 차이가 없습니다.',
      'm.collapse': 'KL이 거의 0입니다. 인코더가 입력을 무시하고 있을 수 있습니다(사후 붕괴). β를 낮춰 보세요.',
      'm.klInfo': 'KL 발산은 {kl}입니다. 잠재공간이 표준정규분포 모양에서 얼마나 벗어났는지를 나타내며, 재구성 품질과 서로 줄다리기를 합니다.',
      'm.clickLatent': '왼쪽 평면을 눌러 보세요',
      'm.testIntro': '잠재공간의 아무 곳이나 누르면, 그 위치의 두 숫자를 디코더에 넣어 새 이미지를 만듭니다. 학습 데이터에 없던 이미지가 "생성"됩니다.',
      'm.gridTitle': '잠재공간 지도',
      'm.gridHelpVae': '잠재공간을 8×8 격자로 나눠 각 위치를 디코딩했습니다. VAE는 잠재공간이 매끄럽게 채워져 있어서, 옆으로 갈수록 숫자가 자연스럽게 다른 숫자로 바뀝니다.',
      'm.gridHelpAe': '잠재공간을 8×8 격자로 나눠 각 위치를 디코딩했습니다. 일반 오토인코더는 빈틈이 있어서 어떤 위치에서는 알아보기 힘든 그림이 나옵니다. VAE 모드와 비교해 보세요.',
    },
    en: {
      'c.variational': 'VAE mode (off = plain autoencoder)',
      'c.hidden': 'Hidden layer neurons',
      'c.beta': 'β (weight of the KL term)',
      'c.n': 'Number of images',
      'p.diverge': 'Learning rate 0.5 → diverges',
      'p.tiny': '2 hidden neurons → underfitting',
      'm.recon': 'Reconstruction error',
      'm.kl': 'KL divergence',
      'm.reconSub': 'Lower than rebuilding with the average image is better',
      'm.baseline': 'Baseline: rebuild with the average image',
      'm.dataIntro': 'This module learns from images alone, with no labels. The goal is to redraw the input image. Since everything must pass through a narrow two-number bottleneck, the model has to learn to summarize the essence of each image.',
      'm.pairs': 'Original vs. rebuilt',
      'm.pairsHelp': 'Top row: originals. Bottom row: images rebuilt from just two numbers.',
      'm.orig': 'orig',
      'm.rec': 'rebuilt',
      'm.latent': 'Latent space (validation images)',
      'm.latentHelp': 'Each image is encoded into two numbers (z₁, z₂) and plotted as a dot, colored by its real digit. The model was never told the digits — if same digits still cluster together, it found a meaningful summary on its own.',
      'm.reconGood': 'Reconstruction error ({r}) is under half of rebuilding with the average image ({b}). It summarizes the images well.',
      'm.reconMeh': 'Reconstruction error ({r}) is not much better than rebuilding with the average image ({b}).',
      'm.collapse': 'KL is almost zero. The encoder may be ignoring its input (posterior collapse). Try a lower β.',
      'm.klInfo': 'KL divergence is {kl}: how far the latent space strays from a standard normal shape. It trades off against reconstruction quality.',
      'm.clickLatent': 'Click the plane on the left',
      'm.testIntro': 'Click anywhere in the latent space: those two numbers go through the decoder to make a new image — one that was never in the training data.',
      'm.gridTitle': 'Latent space map',
      'm.gridHelpVae': 'The latent plane split into an 8×8 grid, each spot decoded. A VAE fills its latent space smoothly, so digits morph naturally into one another as you move across.',
      'm.gridHelpAe': 'The latent plane split into an 8×8 grid, each spot decoded. A plain autoencoder leaves gaps, so some spots decode to unclear pictures. Compare with VAE mode.',
    },
  },
};
