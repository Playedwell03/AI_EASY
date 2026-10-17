import { layersSession, evalBinary, init } from '../../core/tfSession.js';

export function buildModel(tf, hp) {
  const model = tf.sequential();
  for (let i = 0; i < hp.layers; i++) {
    model.add(tf.layers.dense({
      units: hp.units, activation: hp.activation, inputShape: i === 0 ? [2] : undefined,
      kernelInitializer: init(tf, hp.seed, i),
      kernelRegularizer: hp.l2 ? tf.regularizers.l2({ l2: hp.l2 }) : undefined,
    }));
  }
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: init(tf, hp.seed, 99) }));
  model.compile({ optimizer: tf.train.adam(hp.lr), loss: 'binaryCrossentropy' });
  return model;
}

export function createSession(tf, hp, data) {
  const toT = (part) => ({ x: tf.tensor2d(part.x, [part.x.length, 2]), y: tf.tensor2d(part.y, [part.y.length, 1]) });
  const s = layersSession(tf, {
    model: buildModel(tf, hp),
    train: toT(data.train), val: toT(data.val), test: toT(data.test),
    batchSize: hp.batchSize, seed: hp.seed,
    evaluate: (m, x, y) => evalBinary(tf, m, x, y),
  });
  // Probability of class 1 on an n×n grid over [-1,1]² (row-major, y from bottom).
  s.grid = (n) => {
    const pts = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) pts.push([-1 + (2 * i) / (n - 1), -1 + (2 * j) / (n - 1)]);
    return tf.tidy(() => s.model.predict(tf.tensor2d(pts)).dataSync().slice());
  };
  s.predictPoints = (pts) => tf.tidy(() => Array.from(s.model.predict(tf.tensor2d(pts, [pts.length, 2])).dataSync()));
  return s;
}

// Rows of { name, kind, inShape, outShape, params, desc } for the architecture view.
export function architecture(hp) {
  const rows = [{
    name: 'Input (x, y)', kind: 'input', inShape: [2], outShape: [2], params: 0,
    desc: { ko: '평면 위 점의 좌표 두 개(x, y)가 들어옵니다.', en: 'The two coordinates (x, y) of a point on the plane.' },
  }];
  let prev = 2;
  for (let i = 0; i < hp.layers; i++) {
    rows.push({
      name: `Dense ${i + 1} · ${hp.units} · ${hp.activation}`, kind: 'dense', inShape: [prev], outShape: [hp.units],
      params: prev * hp.units + hp.units,
      desc: {
        ko: `완전연결층입니다. 뉴런 ${hp.units}개가 각각 이전 층의 모든 값에 가중치를 곱해 더하고(+편향), ${hp.activation} 활성화 함수로 휘어 줍니다. 이 "휘어짐" 덕분에 직선이 아닌 경계도 그릴 수 있습니다.`,
        en: `A fully connected layer. Each of ${hp.units} neurons multiplies every value from the previous layer by a weight, adds them up (+ bias), and bends the result with ${hp.activation}. That bend is what lets the network draw non-straight boundaries.`,
      },
    });
    prev = hp.units;
  }
  rows.push({
    name: 'Output · sigmoid', kind: 'output', inShape: [prev], outShape: [1], params: prev + 1,
    desc: { ko: '시그모이드로 0~1 사이 값을 냅니다. "이 점이 주황(1) 클래스일 확률"입니다.', en: 'Sigmoid squashes the output into 0–1: the probability that the point is the orange class (1).' },
  });
  return rows;
}
