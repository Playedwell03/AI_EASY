// Sound classifier: spectrogram image → small CNN → one of six sound kinds.
import { mulberry32 } from '../../core/rng.js';
import { splitIndices } from '../../core/split.js';
import { layersSession, evalCategorical, init } from '../../core/tfSession.js';
import { synthClip, features, KINDS } from '../../shared/audio.js';

export const FRAMES = 30;
export const BINS = 64;

export function makeData(hp) {
  const rand = mulberry32(hp.seed);
  const clips = [];
  const y = [];
  for (let i = 0; i < hp.n; i++) {
    const k = i % KINDS.length;
    clips.push(synthClip(KINDS[k], rand));
    y.push(k);
  }
  const x = clips.map(features);
  const idx = splitIndices(hp.n, [0.7, 0.15, 0.15], rand);
  const pick = (ids) => ({ x: ids.map((i) => x[i]), y: ids.map((i) => y[i]), clips: ids.map((i) => clips[i]) });
  const train = pick(idx.train);
  const val = pick(idx.val);
  const test = pick(idx.test);
  return { train, val, test, meta: { counts: { train: train.x.length, val: val.x.length, test: test.x.length } } };
}

export function specTensor(tf, specs) {
  const buf = new Float32Array(specs.length * FRAMES * BINS);
  specs.forEach((s, i) => s.forEach((row, t) => buf.set(row, (i * FRAMES + t) * BINS)));
  return tf.tensor4d(buf, [specs.length, FRAMES, BINS, 1]);
}

export function buildModel(tf, hp) {
  const m = tf.sequential();
  // Average neighbouring frequency bins first (64 → 32): half the work, same information for these sounds.
  m.add(tf.layers.averagePooling2d({ inputShape: [FRAMES, BINS, 1], poolSize: [1, 2] }));
  m.add(tf.layers.conv2d({ filters: hp.filters, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 0) }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.conv2d({ filters: hp.filters * 2, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 1) }));
  m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: hp.dense, activation: 'relu', kernelInitializer: init(tf, hp.seed, 2) }));
  m.add(tf.layers.dense({ units: KINDS.length, activation: 'softmax', kernelInitializer: init(tf, hp.seed, 3) }));
  m.compile({ optimizer: tf.train.adam(hp.lr), loss: 'categoricalCrossentropy' });
  return m;
}

export function createSession(tf, hp, data) {
  const toT = (part) => ({ x: specTensor(tf, part.x), y: tf.tidy(() => tf.oneHot(tf.tensor1d(part.y, 'int32'), KINDS.length).toFloat()) });
  const s = layersSession(tf, {
    model: buildModel(tf, hp),
    train: toT(data.train), val: toT(data.val), test: toT(data.test),
    batchSize: hp.batchSize, seed: hp.seed,
    evaluate: (m, x, y) => evalCategorical(tf, m, x, y),
  });
  s.predictSpecs = (specs) => tf.tidy(() => {
    const p = s.model.predict(specTensor(tf, specs)).arraySync();
    return p;
  });
  s.predictClips = (clips) => s.predictSpecs(clips.map(features));
  return s;
}

export function architecture(hp) {
  const half = BINS / 2;
  const c1 = [FRAMES - 2, half - 2];
  const p1 = c1.map((v) => Math.floor(v / 2));
  const c2 = p1.map((v) => v - 2);
  const p2 = c2.map((v) => Math.floor(v / 2));
  const f2 = hp.filters * 2;
  const flat = p2[0] * p2[1] * f2;
  return [
    { name: 'Sound · 0.5 s @ 8 kHz', kind: 'input', inShape: [4000], outShape: [4000], params: 0,
      desc: { ko: '0.5초 길이의 소리입니다. 1초에 8000번 잰 공기 압력 값 4000개로 이루어져 있습니다.', en: 'Half a second of sound: 4000 air-pressure samples, taken 8000 times per second.' } },
    { name: `Spectrogram · ${FRAMES}×${BINS}`, kind: 'other', inShape: [4000], outShape: [FRAMES, BINS, 1], params: 0,
      desc: { ko: '소리를 짧은 조각으로 나눠 조각마다 어떤 높이(주파수)의 소리가 얼마나 센지 계산합니다(푸리에 변환). 그러면 소리가 "가로=시간, 세로=높이"인 그림이 됩니다. 이 단계는 학습되지 않는 고정 계산입니다.', en: 'Cuts the sound into short slices and measures how strong each pitch (frequency) is in each slice (a Fourier transform). The sound becomes a picture: across = time, up = pitch. This step is fixed math, not learned.' } },
    { name: 'AvgPool 1×2 (pitch)', kind: 'pool', inShape: [FRAMES, BINS, 1], outShape: [FRAMES, half, 1], params: 0,
      desc: { ko: '이웃한 높이 칸 두 개를 평균 내어 그림을 세로로 절반 줄입니다. 계산이 절반으로 줄고, 이 소리들을 구별하는 데 필요한 정보는 남습니다.', en: 'Averages pairs of neighbouring pitch rows, halving the picture vertically. Half the work, and the information needed to tell these sounds apart remains.' } },
    { name: `Conv 3×3 · ${hp.filters}`, kind: 'conv', inShape: [FRAMES, half, 1], outShape: [...c1, hp.filters], params: 9 * hp.filters + hp.filters,
      desc: { ko: '이미지 CNN과 똑같이 필터가 그림을 훑으며 "올라가는 선", "짧은 점" 같은 무늬를 찾습니다. 소리 인식을 이미지 인식으로 바꾼 셈입니다.', en: 'Just like the image CNN, filters slide over the picture looking for patterns like "rising line" or "short blip". Sound recognition becomes image recognition.' } },
    { name: 'MaxPool 2×2', kind: 'pool', inShape: [...c1, hp.filters], outShape: [...p1, hp.filters], params: 0,
      desc: { ko: '크기를 절반으로 줄입니다.', en: 'Halves the size.' } },
    { name: `Conv 3×3 · ${f2}`, kind: 'conv', inShape: [...p1, hp.filters], outShape: [...c2, f2], params: 9 * hp.filters * f2 + f2,
      desc: { ko: '무늬를 조합해 "두 번 삑", "점점 높아짐" 같은 더 긴 모양을 봅니다.', en: 'Combines patterns into longer shapes like "two beeps" or "rising pitch".' } },
    { name: 'MaxPool 2×2 · Flatten', kind: 'pool', inShape: [...c2, f2], outShape: [flat], params: 0,
      desc: { ko: '한 번 더 줄이고 한 줄로 펼칩니다.', en: 'Shrinks once more and unrolls into one row.' } },
    { name: `Dense ${hp.dense}`, kind: 'dense', inShape: [flat], outShape: [hp.dense], params: flat * hp.dense + hp.dense,
      desc: { ko: '찾은 무늬를 종합합니다.', en: 'Combines the detected patterns.' } },
    { name: `Dense ${KINDS.length} · softmax`, kind: 'output', inShape: [hp.dense], outShape: [KINDS.length], params: hp.dense * KINDS.length + KINDS.length,
      desc: { ko: '여섯 가지 소리 종류 각각의 확률을 냅니다.', en: 'Gives a probability for each of the six sound kinds.' } },
  ];
}
