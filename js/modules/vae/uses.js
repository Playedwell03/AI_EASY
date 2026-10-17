// "Use" tab apps for the autoencoder / VAE.
import { h, fmt } from '../../ui/dom.js';
import { mulberry32, randn } from '../../core/rng.js';
import { renderDigit, frameDrawing } from '../../shared/digits.js';
import { step, lt } from '../../ui/useView.js';
import { drawPad, imgCell } from '../../ui/pad.js';

export function pixelError(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

export function percentile(xs, q) {
  const s = [...xs].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];
}

// Salt-and-pepper style noise plus gaussian jitter, clipped to [0, 1].
export function addNoise(img, amount, rand) {
  return Float32Array.from(img, (v) => {
    let x = v + randn(rand) * amount * 0.35;
    if (rand() < amount * 0.15) x = rand() < 0.5 ? 0 : 1;
    return Math.max(0, Math.min(1, x));
  });
}

export const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

const errorOf = (s, img) => pixelError(img, s.reconstruct([img])[0]);

const anomaly = {
  id: 'anomaly',
  icon: '🔍',
  title: lt('불량품 탐지', 'Defect detector'),
  intro: lt('정상 제품만 보고 배운 모델이 "잘 복원되지 않는 것"을 불량으로 찾아냅니다.', 'A model that only ever saw normal items flags whatever it cannot rebuild well.'),
  realWorld: lt('공장·카드 결제·서버 로그에서 불량 사례는 드물어 정답을 모으기 어렵습니다. 그래서 정상만으로 오토인코더를 학습하고 복원 오차로 이상을 찾는 방법을 실제로 많이 씁니다.', 'In factories, card payments or server logs, defects are too rare to label. Training an autoencoder on normal data and flagging high rebuild error is a common real approach.'),
  pipeline: [
    step('input', '제품 사진', 'Item image', '검사할 이미지가 들어옵니다(정상 숫자 또는 이상한 이미지).', 'An image to inspect comes in (a normal digit or something odd).'),
    step('pre', '400픽셀로 펼치기', 'Flatten', '20×20 이미지를 400개 숫자로 펼칩니다.', 'The 20×20 image becomes 400 numbers.'),
    step('model', '압축 후 복원', 'Compress & rebuild', '학습한 오토인코더가 숫자 2개로 압축했다가 다시 그립니다.', 'Your autoencoder squeezes it into 2 numbers and redraws it.'),
    step('post', '복원 오차 비교', 'Compare error', '원본과 복원본의 차이(오차)를 재고, 정상 데이터 95%가 넘지 않는 선을 기준으로 삼습니다.', 'Measures how different the rebuild is, against a threshold that 95% of normal items stay under.'),
    step('output', '정상 / 불량 경보', 'OK / Alarm', '기준을 넘으면 불량 경보를 냅니다.', 'Above the threshold, it raises a defect alarm.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const normals = ctx.data.val.x.slice(0, 60);
    const threshold = percentile(normals.map((img) => errorOf(s, img)), 0.95);
    const rand = mulberry32(ctx.hp.seed + 99);
    const samples = [
      { name: lt('정상 숫자', 'Normal digit'), img: ctx.data.test.x[0] },
      { name: lt('정상 숫자', 'Normal digit'), img: ctx.data.test.x[1] },
      { name: lt('얼룩진 제품', 'Smudged item'), img: addNoise(ctx.data.test.x[2], 0.8, rand) },
      { name: lt('찌그러진 제품', 'Warped item'), img: renderDigit(5, rand, 20, { variation: 2.5 }) },
      { name: lt('엉뚱한 물건', 'Wrong object'), img: Float32Array.from({ length: 400 }, (_, i) => ((i % 20) > 5 && (i % 20) < 14 && i > 100 && i < 300 ? 1 : 0)) },
    ];
    const result = h('div');
    const check = (img, name) => {
      flow.run();
      const rec = s.reconstruct([img])[0];
      const err = pixelError(img, rec);
      const bad = err > threshold;
      result.replaceChildren(h('div', { class: 'row', style: { gap: '16px', alignItems: 'center' } },
        h('div', { style: { width: '72px' } }, imgCell(img, ctx.t('use.vae.input'))),
        h('div', { style: { width: '72px' } }, imgCell(rec, ctx.t('use.vae.rebuilt'))),
        h('div', null,
          h('div', { class: 'big-result' }, bad ? h('span', { class: 'tag-bad' }, `🚨 ${ctx.t('use.vae.defect')}`) : h('span', { class: 'tag-good' }, `✅ ${ctx.t('use.vae.ok')}`)),
          h('div', { class: 'small muted' }, ctx.t('use.vae.errLine', { e: fmt(err), t: fmt(threshold), name }))),
      ));
    };
    const pad = drawPad((img) => check(frameDrawing(img), ctx.t('use.vae.drawn')), { size: 120, brush: 10 });
    el.append(
      h('p', { class: 'help' }, ctx.t('use.vae.anomalyHelp')),
      h('div', { class: 'row', style: { gap: '8px' } }, samples.map((smp) => h('button', { class: 'chip', type: 'button', onclick: () => check(smp.img, ctx.L(smp.name)) },
        h('span', { style: { width: '28px', display: 'inline-block' } }, imgCell(smp.img, '')), ctx.L(smp.name)))),
      h('div', { class: 'row', style: { gap: '12px', alignItems: 'center' } }, pad.canvas, h('button', { class: 'btn btn-sm', type: 'button', onclick: () => pad.clear() }, ctx.t('use.vae.clear')), h('span', { class: 'small muted' }, ctx.t('use.vae.drawHint'))),
      result);
    check(samples[0].img, ctx.L(samples[0].name));
  },
};

const denoise = {
  id: 'denoise',
  icon: '🧽',
  title: lt('노이즈 제거기', 'Noise cleaner'),
  intro: lt('지저분한 스캔 이미지를 넣으면 깨끗한 숫자로 다시 그려 줍니다.', 'Feed in a messy scan and get a clean digit back.'),
  realWorld: lt('사진 앱의 노이즈 제거, 오래된 문서 복원은 노이즈가 있는 이미지와 깨끗한 이미지 쌍으로 따로 학습한 더 큰 모델을 씁니다. 여기서는 정상 숫자만 배운 모델이 "아는 모양"으로 끌어당겨 지저분함이 사라지는 효과를 봅니다.', 'Photo denoisers and document restorers train bigger models on pairs of noisy and clean images. Here, a model that only knows clean digits pulls the input toward shapes it knows, so the mess disappears.'),
  pipeline: [
    step('input', '지저분한 스캔', 'Noisy scan', '노이즈가 섞인 숫자 이미지입니다.', 'A digit image with noise mixed in.'),
    step('pre', '펼치기', 'Flatten', '400개 숫자로 펼칩니다.', 'Flattened into 400 numbers.'),
    step('model', '아는 모양으로 복원', 'Rebuild as known shape', '오토인코더는 깨끗한 숫자만 봤기 때문에, 숫자 2개로 요약하는 과정에서 노이즈가 버려집니다.', 'The autoencoder only saw clean digits, so summarizing into 2 numbers drops the noise.'),
    step('post', '이미지로 되돌리기', 'Back to image', '400개 숫자를 다시 20×20 이미지로 바꿉니다.', 'The 400 numbers become a 20×20 image again.'),
    step('output', '깨끗한 결과', 'Clean result', '정리된 이미지를 보여 줍니다.', 'Shows the cleaned image.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    let amount = 0.5;
    let idx = 0;
    const out = h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(6, minmax(0, 1fr))', maxWidth: '520px' } });
    const val = h('span', { class: 'val' }, '50%');
    const run = () => {
      flow.run();
      const rand = mulberry32(ctx.hp.seed * 3 + idx);
      const cleanImgs = ctx.data.test.x.slice(idx, idx + 3);
      const noisy = cleanImgs.map((img) => addNoise(img, amount, rand));
      const rec = s.reconstruct(noisy);
      out.replaceChildren(...noisy.flatMap((n, i) => [imgCell(n, ctx.t('use.vae.noisy')), imgCell(rec[i], ctx.t('use.vae.cleaned'))]));
    };
    el.append(
      h('div', { class: 'control', style: { maxWidth: '320px' } }, h('label', null, h('span', null, ctx.t('use.vae.noise')), val),
        h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: amount, oninput: (e) => { amount = Number(e.target.value); val.textContent = `${Math.round(amount * 100)}%`; run(); } })),
      h('div', null, h('button', { class: 'btn', type: 'button', onclick: () => { idx = (idx + 3) % 30; run(); } }, ctx.t('use.vae.another'))),
      out);
    run();
  },
};

const morph = {
  id: 'morph',
  icon: '🎨',
  title: lt('새 글자 생성기', 'New glyph maker'),
  intro: lt('두 숫자를 고르면, 그 사이를 잇는 처음 보는 글자들을 만들어 냅니다.', 'Pick two digits and it invents in-between glyphs nobody drew.'),
  realWorld: lt('글꼴 디자인 도구, 게임 캐릭터 생성기는 같은 원리(잠재공간 사이를 잇기)를 훨씬 큰 생성 모델로 씁니다.', 'Font design tools and character creators use the same idea — walking between points in a latent space — with much larger generative models.'),
  pipeline: [
    step('input', '두 숫자 고르기', 'Pick two digits', '시작 숫자와 끝 숫자를 고릅니다.', 'Choose a start digit and an end digit.'),
    step('pre', '대표 이미지 찾기', 'Find examples', '데이터에서 각 숫자의 예시 이미지를 찾습니다.', 'Finds an example image of each digit in the data.'),
    step('model', '인코딩 · 디코딩', 'Encode · decode', '인코더로 두 점(z)을 얻고, 그 사이 점들을 디코더로 그립니다.', 'The encoder gives two points (z); the decoder draws the points between them.'),
    step('post', '사이 점 계산', 'Points between', '두 점을 잇는 직선 위에 같은 간격으로 점을 찍습니다.', 'Places evenly spaced points on the line between them.'),
    step('output', '변신 애니메이션', 'Morph strip', '처음 보는 중간 글자들이 나타납니다.', 'New in-between glyphs appear.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    let from = 1;
    let to = 7;
    const strip = h('div', { class: 'img-grid', style: { gridTemplateColumns: 'repeat(8, minmax(0, 1fr))' } });
    const exampleOf = (d) => ctx.data.val.x[ctx.data.val.y.indexOf(d)];
    const run = () => {
      flow.run();
      const [za, zb] = s.encode([exampleOf(from), exampleOf(to)]);
      const zs = Array.from({ length: 8 }, (_, i) => lerp2(za, zb, i / 7));
      strip.replaceChildren(...s.decode(zs).map((img, i) => imgCell(img, i === 0 ? String(from) : i === 7 ? String(to) : '')));
    };
    const picker = (get, set) => h('div', { class: 'row', style: { gap: '4px' } }, Array.from({ length: 10 }, (_, d) =>
      h('button', { class: `chip${get() === d ? ' tag-good' : ''}`, type: 'button', onclick: (e) => { set(d); e.target.parentElement.querySelectorAll('.chip').forEach((c, j) => c.classList.toggle('tag-good', j === d)); run(); } }, String(d))));
    el.append(
      h('div', { class: 'row' }, h('b', null, ctx.t('use.vae.from')), picker(() => from, (d) => { from = d; })),
      h('div', { class: 'row' }, h('b', null, ctx.t('use.vae.to')), picker(() => to, (d) => { to = d; })),
      strip);
    run();
  },
};

const compress = {
  id: 'compress',
  icon: '📡',
  title: lt('초압축 전송', 'Ultra-compressed transfer'),
  intro: lt('이미지를 숫자 두 개로 줄여 보내고, 받는 쪽에서 다시 그립니다.', 'Send an image as just two numbers and redraw it on the other side.'),
  realWorld: lt('영상 통화·위성 통신의 "학습 기반 압축"은 보내는 쪽에 인코더, 받는 쪽에 디코더를 두는 같은 구조를 씁니다. 실제로는 숫자를 훨씬 많이 써서 화질을 지킵니다.', 'Learned compression for video calls or satellite links puts the encoder at the sender and the decoder at the receiver — with far more numbers, to keep quality.'),
  pipeline: [
    step('input', '보낼 이미지', 'Image to send', '20×20 = 400픽셀 이미지입니다.', 'A 20×20 = 400-pixel image.'),
    step('model', '인코더(보내는 쪽)', 'Encoder (sender)', '학습한 인코더가 400개 값을 숫자 2개로 요약합니다. 디코더(받는 쪽)도 같은 모델의 나머지 절반입니다.', 'Your trained encoder summarizes 400 values into 2 numbers. The decoder (receiver) is the other half of the same model.'),
    step('pre', '전송', 'Transmit', '숫자 2개만 네트워크로 보냅니다.', 'Only 2 numbers travel over the network.'),
    step('post', '디코더(받는 쪽)', 'Decoder (receiver)', '받은 숫자 2개로 이미지를 다시 그립니다.', 'Redraws the image from the 2 numbers.'),
    step('output', '받은 이미지', 'Received image', '원본과 얼마나 비슷한지 비교합니다.', 'Compare how close it is to the original.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    let idx = 0;
    const view = h('div');
    const run = () => {
      flow.run();
      const img = ctx.data.test.x[idx];
      const z = s.encode([img])[0];
      const rec = s.decode([z])[0];
      view.replaceChildren(h('div', { class: 'row', style: { gap: '14px', alignItems: 'center' } },
        h('div', { style: { width: '80px' } }, imgCell(img, ctx.t('use.vae.sent'))),
        h('div', { class: 'stack', style: { gap: '2px', textAlign: 'center' } }, h('span', { class: 'mono' }, `[${z.map((v) => v.toFixed(2)).join(', ')}]`), h('span', { class: 'small muted' }, '→ 📡 →')),
        h('div', { style: { width: '80px' } }, imgCell(rec, ctx.t('use.vae.received')))),
      h('p', { class: 'small', style: { marginTop: '10px' } }, ctx.t('use.vae.ratio', { bytes: 400, z: 8, x: 50, e: fmt(pixelError(img, rec)) })));
    };
    el.append(h('div', null, h('button', { class: 'btn', type: 'button', onclick: () => { idx = (idx + 1) % ctx.data.test.x.length; run(); } }, ctx.t('use.vae.another'))), view);
    run();
  },
};

export const useCases = [anomaly, denoise, morph, compress];

export const USE_TEXT = {
  ko: {
    'use.vae.input': '입력',
    'use.vae.rebuilt': '복원',
    'use.vae.defect': '불량 의심',
    'use.vae.ok': '정상',
    'use.vae.errLine': '{name} · 복원 오차 {e} (기준선 {t})',
    'use.vae.anomalyHelp': '아래 제품을 하나씩 검사해 보세요. 정상 숫자는 잘 복원되지만, 처음 보는 이상한 것은 복원이 엉망이라 오차가 커집니다. 직접 그려서 검사해도 됩니다.',
    'use.vae.drawn': '직접 그린 이미지',
    'use.vae.clear': '지우기',
    'use.vae.drawHint': '그린 것을 검사합니다',
    'use.vae.noise': '노이즈 양',
    'use.vae.noisy': '지저분',
    'use.vae.cleaned': '정리됨',
    'use.vae.another': '다른 이미지',
    'use.vae.from': '시작',
    'use.vae.to': '끝',
    'use.vae.sent': '보낸 이미지',
    'use.vae.received': '받은 이미지',
    'use.vae.ratio': '원본 {bytes}바이트 → 전송 {z}바이트 (약 {x}배 압축) · 차이 {e}',
  },
  en: {
    'use.vae.input': 'input',
    'use.vae.rebuilt': 'rebuilt',
    'use.vae.defect': 'Possible defect',
    'use.vae.ok': 'Normal',
    'use.vae.errLine': '{name} · rebuild error {e} (threshold {t})',
    'use.vae.anomalyHelp': 'Inspect the items below one by one. Normal digits rebuild well; strange things it never saw rebuild badly, so the error jumps. You can also draw something to inspect.',
    'use.vae.drawn': 'Your drawing',
    'use.vae.clear': 'Clear',
    'use.vae.drawHint': 'inspects what you draw',
    'use.vae.noise': 'Noise amount',
    'use.vae.noisy': 'noisy',
    'use.vae.cleaned': 'cleaned',
    'use.vae.another': 'Another image',
    'use.vae.from': 'From',
    'use.vae.to': 'To',
    'use.vae.sent': 'sent',
    'use.vae.received': 'received',
    'use.vae.ratio': '{bytes} bytes → {z} bytes sent (about {x}× smaller) · difference {e}',
  },
};
