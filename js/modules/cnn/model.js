import { layersSession, evalCategorical, init } from '../../core/tfSession.js';
import { makeDigitSplits, DIGIT_SIZE as S } from '../../shared/digits.js';

export function makeData(hp) {
  return makeDigitSplits(hp.n, hp.seed);
}

// Stack a list of Float32Array images into a [n, S, S, 1] tensor.
export function imagesTensor(tf, images) {
  const buf = new Float32Array(images.length * S * S);
  images.forEach((img, i) => buf.set(img, i * S * S));
  return tf.tensor4d(buf, [images.length, S, S, 1]);
}

export function buildModel(tf, hp) {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape: [S, S, 1], filters: hp.filters, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 0) }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: hp.filters * 2, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 1) }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: hp.dense, activation: 'relu', kernelInitializer: init(tf, hp.seed, 2) }));
  if (hp.dropout) m.add(tf.layers.dropout({ rate: hp.dropout, seed: hp.seed }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelInitializer: init(tf, hp.seed, 3) }));
  m.compile({ optimizer: tf.train.adam(hp.lr), loss: 'categoricalCrossentropy' });
  return m;
}

export function createSession(tf, hp, data) {
  const toT = (part) => ({ x: imagesTensor(tf, part.x), y: tf.tidy(() => tf.oneHot(tf.tensor1d(part.y, 'int32'), 10).toFloat()) });
  const s = layersSession(tf, {
    model: buildModel(tf, hp),
    train: toT(data.train), val: toT(data.val), test: toT(data.test),
    batchSize: hp.batchSize, seed: hp.seed,
    evaluate: (m, x, y) => evalCategorical(tf, m, x, y),
  });
  let probe = null;
  // Class probabilities for a list of images.
  s.predictImages = (images) => tf.tidy(() => {
    const p = s.model.predict(imagesTensor(tf, images));
    return Array.from({ length: images.length }, (_, i) => Array.from(p.slice([i, 0], [1, 10]).dataSync()));
  });
  // First conv layer: filters as 3×3 arrays, and its feature maps for one image.
  s.firstLayer = (image) => {
    const kernel = s.model.layers[0].getWeights()[0];
    const k = kernel.arraySync(); // [3][3][1][F]
    const filters = Array.from({ length: hp.filters }, (_, f) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => k[r][c][0][f])));
    probe ??= tf.model({ inputs: s.model.inputs, outputs: s.model.layers[0].output });
    const maps = tf.tidy(() => probe.predict(imagesTensor(tf, [image])).arraySync()[0]); // [18][18][F]
    return { filters, maps };
  };
  return s;
}

export function architecture(hp) {
  const c1 = S - 2;
  const p1 = Math.floor(c1 / 2);
  const c2 = p1 - 2;
  const p2 = Math.floor(c2 / 2);
  const f2 = hp.filters * 2;
  const flat = p2 * p2 * f2;
  const rows = [
    { name: `Input ${S}×${S}`, kind: 'input', inShape: [S, S, 1], outShape: [S, S, 1], params: 0,
      desc: { ko: `${S}×${S} 흑백 이미지입니다. 각 픽셀은 0(검정)~1(흰색) 사이 숫자입니다.`, en: `A ${S}×${S} grayscale image. Each pixel is a number from 0 (black) to 1 (white).` } },
    { name: `Conv 3×3 · ${hp.filters} filters`, kind: 'conv', inShape: [S, S, 1], outShape: [c1, c1, hp.filters], params: 9 * hp.filters + hp.filters,
      desc: { ko: `3×3 크기의 작은 돋보기(필터) ${hp.filters}개가 이미지를 훑으며 선, 모서리 같은 단순한 패턴을 찾습니다. 같은 필터를 모든 위치에 쓰기 때문에 숫자가 어디 있든 찾을 수 있습니다.`, en: `${hp.filters} small 3×3 magnifiers (filters) slide over the image looking for simple patterns like strokes and edges. The same filter is used everywhere, so it finds a pattern wherever it is.` } },
    { name: 'MaxPool 2×2', kind: 'pool', inShape: [c1, c1, hp.filters], outShape: [p1, p1, hp.filters], params: 0,
      desc: { ko: '2×2 칸마다 가장 큰 값만 남겨 크기를 절반으로 줄입니다. 위치가 조금 달라도 결과가 비슷해집니다.', en: 'Keeps only the largest value in each 2×2 block, halving the size. Small shifts in position matter less.' } },
    { name: `Conv 3×3 · ${f2} filters`, kind: 'conv', inShape: [p1, p1, hp.filters], outShape: [c2, c2, f2], params: 9 * hp.filters * f2 + f2,
      desc: { ko: '앞 층이 찾은 선과 모서리를 조합해 곡선, 고리처럼 더 복잡한 모양을 찾습니다.', en: 'Combines the strokes and edges found by the previous layer into more complex shapes like curves and loops.' } },
    { name: 'MaxPool 2×2', kind: 'pool', inShape: [c2, c2, f2], outShape: [p2, p2, f2], params: 0,
      desc: { ko: '다시 한 번 크기를 절반으로 줄입니다.', en: 'Halves the size once more.' } },
    { name: 'Flatten', kind: 'other', inShape: [p2, p2, f2], outShape: [flat], params: 0,
      desc: { ko: '3차원 특징 지도를 한 줄로 펼칩니다.', en: 'Unrolls the 3D feature maps into one long list.' } },
    { name: `Dense ${hp.dense}`, kind: 'dense', inShape: [flat], outShape: [hp.dense], params: flat * hp.dense + hp.dense,
      desc: { ko: '찾아낸 모양들을 종합해 어떤 숫자인지 판단할 근거를 만듭니다.', en: 'Combines the detected shapes into evidence for which digit it is.' } },
  ];
  if (hp.dropout) rows.push({ name: `Dropout ${hp.dropout}`, kind: 'other', inShape: [hp.dense], outShape: [hp.dense], params: 0,
    desc: { ko: '학습 중 일부 뉴런을 무작위로 꺼서 특정 뉴런에만 의존하지 않게 합니다(과적합 방지).', en: 'Randomly switches off some neurons during training so the model cannot rely on any single one (reduces overfitting).' } });
  rows.push({ name: 'Dense 10 · softmax', kind: 'output', inShape: [hp.dense], outShape: [10], params: hp.dense * 10 + 10,
    desc: { ko: '0~9 각 숫자일 확률 10개를 냅니다. 합은 1(100%)입니다.', en: 'Outputs 10 probabilities, one for each digit 0–9. They add up to 1 (100%).' } });
  return rows;
}
