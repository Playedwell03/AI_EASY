import { layersSession, init } from '../../core/tfSession.js';
import { naiveErrors } from './data.js';

export function buildModel(tf, hp) {
  const m = tf.sequential();
  if (hp.arch === 'lstm') {
    m.add(tf.layers.lstm({ inputShape: [hp.window, 1], units: hp.units, kernelInitializer: init(tf, hp.seed, 0), recurrentInitializer: tf.initializers.orthogonal({ seed: hp.seed }) }));
  } else {
    m.add(tf.layers.conv1d({ inputShape: [hp.window, 1], filters: hp.units, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 0) }));
    m.add(tf.layers.conv1d({ filters: hp.units, kernelSize: 3, activation: 'relu', kernelInitializer: init(tf, hp.seed, 1) }));
    m.add(tf.layers.flatten());
  }
  m.add(tf.layers.dense({ units: 1, kernelInitializer: init(tf, hp.seed, 9) }));
  m.compile({ optimizer: tf.train.adam(hp.lr), loss: 'meanSquaredError' });
  return m;
}

export function createSession(tf, hp, data) {
  // Windows are fed relative to their last value and the model predicts the change, so a rising
  // trend that leaves the training range does not push inputs into unfamiliar territory.
  const rel = (xs) => tf.tensor3d(xs.map((w) => w.map((v) => [v - w[w.length - 1]])), [xs.length, hp.window, 1]);
  const toT = (part) => ({ x: rel(part.x), y: tf.tensor2d(part.y.map((v, i) => v - part.x[i][hp.window - 1]), [part.y.length, 1]) });
  const s = layersSession(tf, {
    model: buildModel(tf, hp),
    train: toT(data.train), val: toT(data.val), test: toT(data.test),
    batchSize: hp.batchSize, seed: hp.seed,
    async evaluate(m, x, y) {
      const [mse, mae] = tf.tidy(() => {
        const p = m.predict(x);
        return [tf.losses.meanSquaredError(y, p), tf.losses.absoluteDifference(y, p)];
      });
      const out = { loss: (await mse.data())[0], metrics: { mae: (await mae.data())[0] } };
      mse.dispose();
      mae.dispose();
      return out;
    },
  });
  const evalTest = s.evaluateTest;
  s.evaluateTest = async () => {
    const r = await evalTest();
    const naive = naiveErrors(data.test);
    return { ...r, mse: r.loss, naiveMae: naive.mae, naiveMse: naive.mse };
  };
  s.predictWindows = (xs) => tf.tidy(() => Array.from(s.model.predict(rel(xs)).dataSync()).map((d, i) => xs[i][hp.window - 1] + d));
  // Feed each prediction back in as the newest value, `steps` times.
  s.forecast = (start, steps) => {
    const w = start.slice();
    const out = [];
    for (let i = 0; i < steps; i++) {
      const p = s.predictWindows([w.slice(-hp.window)])[0];
      out.push(p);
      w.push(p);
    }
    return out;
  };
  return s;
}

export function architecture(hp) {
  const rows = [{ name: `Window · last ${hp.window} values`, kind: 'input', inShape: [hp.window, 1], outShape: [hp.window, 1], params: 0,
    desc: { ko: `최근 ${hp.window}개의 값을 순서대로 넣습니다. 이 창(window)을 한 칸씩 옮기며 많은 학습 예제를 만듭니다. 값은 "창의 마지막 값과의 차이"로 바꿔서 넣고, 모델은 다음 값이 지금보다 얼마나 변할지를 예측합니다. 그래서 추세 때문에 처음 보는 높이로 올라가도 당황하지 않습니다.`, en: `The most recent ${hp.window} values, in order. Sliding this window one step at a time creates many training examples. Each value is given as its difference from the window's last value, and the model predicts how much the next value will change — so a trend that climbs to unseen levels does not throw it off.` } }];
  if (hp.arch === 'lstm') {
    rows.push({ name: `LSTM · ${hp.units}`, kind: 'rnn', inShape: [hp.window, 1], outShape: [hp.units], params: 4 * (hp.units * (hp.units + 1) + hp.units),
      desc: { ko: 'LSTM은 값을 하나씩 순서대로 읽으며 "기억"을 갱신합니다. 기억할 것과 잊을 것을 고르는 문(gate)이 있어서 오래전 패턴도 유지할 수 있습니다.', en: 'An LSTM reads the values one by one, updating its memory. Gates decide what to keep and what to forget, so it can hold on to patterns from further back.' } });
  } else {
    rows.push({ name: `Conv1D 3 · ${hp.units}`, kind: 'conv', inShape: [hp.window, 1], outShape: [hp.window - 2, hp.units], params: 3 * hp.units + hp.units,
      desc: { ko: '시간 축을 따라 3칸짜리 필터가 미끄러지며 "올라가는 중", "꺾이는 중" 같은 짧은 모양을 찾습니다.', en: 'A 3-step filter slides along time and picks up short shapes like "rising" or "turning".' } });
    rows.push({ name: `Conv1D 3 · ${hp.units}`, kind: 'conv', inShape: [hp.window - 2, hp.units], outShape: [hp.window - 4, hp.units], params: 3 * hp.units * hp.units + hp.units,
      desc: { ko: '짧은 모양들을 조합해 더 긴 패턴을 봅니다.', en: 'Combines short shapes into longer patterns.' } });
    rows.push({ name: 'Flatten', kind: 'other', inShape: [hp.window - 4, hp.units], outShape: [(hp.window - 4) * hp.units], params: 0,
      desc: { ko: '한 줄로 펼칩니다.', en: 'Unrolls into one row.' } });
  }
  const inDim = hp.arch === 'lstm' ? hp.units : (hp.window - 4) * hp.units;
  rows.push({ name: 'Dense 1 · next value', kind: 'output', inShape: [inDim], outShape: [1], params: inDim + 1,
    desc: { ko: '다음 값이 지금 값에서 얼마나 변할지 예측합니다. 여기에 마지막 값을 더하면 예측값이 됩니다.', en: 'Predicts how much the next value changes from the current one. Adding back the last value gives the forecast.' } });
  return rows;
}
