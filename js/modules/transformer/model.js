// A tiny GPT-style character model: causal self-attention, trained to predict every next character.
import { mulberry32, shuffle } from '../../core/rng.js';
import { makeTextData, encode } from '../../shared/corpus.js';
import { createEncoder } from '../../shared/attention.js';
import { addGeneration } from '../rnn/model.js';

export function makeData(hp) {
  return makeTextData({ lang: hp.corpus, chars: hp.chars, seqLen: hp.seqLen, seed: hp.seed, fraction: hp.corpusFraction, stride: Math.max(1, hp.seqLen / 2) });
}

export function createSession(tf, hp, data) {
  const V = data.vocab.itos.length;
  const T = hp.seqLen;
  const D = hp.dModel;
  const enc = createEncoder(tf, { vocab: V, dModel: D, heads: hp.heads, ff: D * 2, blocks: hp.blocks, maxLen: T, seed: hp.seed });
  const Wout = tf.variable(tf.randomNormal([D, V], 0, 1 / Math.sqrt(D), 'float32', hp.seed * 3 + 11));
  const bout = tf.variable(tf.zeros([V]));
  const vars = [...enc.variables, Wout, bout];
  const optimizer = tf.train.adam(hp.lr);
  // Targets for every position: the window shifted by one, ending with the next character.
  const tensors = (part) => ({
    x: tf.tensor2d(part.x, [part.x.length, T], 'int32'),
    y: tf.tensor2d(part.x.map((w, i) => [...w.slice(1), part.y[i]]), [part.x.length, T], 'int32'),
  });
  const S = { train: tensors(data.train), val: tensors(data.val), test: tensors(data.test) };
  const rand = mulberry32(hp.seed * 17 + 3);

  const logits = (ids) => {
    const { hidden } = enc.apply(ids, { causal: true });
    const [B] = ids.shape;
    return hidden.reshape([B * T, D]).matMul(Wout).add(bout).reshape([B, T, V]);
  };
  const lossOf = (x, y) => tf.losses.softmaxCrossEntropy(tf.oneHot(y, V), logits(x));

  async function evaluate(split) {
    const { x, y } = S[split];
    const t = tf.tidy(() => {
      const lg = logits(x);
      const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(y, V), lg);
      const acc = lg.argMax(-1).equal(y).mean();
      return tf.stack([loss, acc]);
    });
    const [loss, acc] = await t.data();
    t.dispose();
    return { loss, acc };
  }

  let disposed = false;
  const s = {
    async epoch() {
      const n = S.train.x.shape[0];
      const order = shuffle(Array.from({ length: n }, (_, i) => i), rand);
      for (let b = 0; b < n; b += hp.batchSize) {
        const ids = tf.tensor1d(order.slice(b, b + hp.batchSize), 'int32');
        const cost = optimizer.minimize(() => lossOf(tf.gather(S.train.x, ids), tf.gather(S.train.y, ids)), true, vars);
        ids.dispose();
        cost?.dispose();
      }
      const tr = await evaluate('train');
      const v = await evaluate('val');
      return { trainLoss: tr.loss, valLoss: v.loss, metrics: { acc: v.acc, perplexity: Math.exp(v.loss), train_acc: tr.acc } };
    },
    async evaluateTest() {
      const r = await evaluate('test');
      return { loss: r.loss, acc: r.acc, perplexity: Math.exp(r.loss) };
    },
    // Attention weights for the last T characters of text: [blocks][heads][t][t] plus the characters.
    attentionMaps(text) {
      const ids = encode(text, data.vocab.stoi).slice(-T);
      if (!ids.length) return [];
      return tf.tidy(() => enc.apply(tf.tensor2d([ids], [1, ids.length], 'int32'), { causal: true }).weights.map((w) => w.arraySync()[0]));
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      optimizer.dispose();
      enc.dispose();
      Wout.dispose();
      bout.dispose();
      Object.values(S).forEach((p) => { p.x.dispose(); p.y.dispose(); });
    },
  };
  return addGeneration(s, data, T, (w) => tf.tidy(() => {
    const p = tf.softmax(logits(tf.tensor2d([w], [1, T], 'int32')).slice([0, T - 1, 0], [1, 1, V]).reshape([V]));
    return Array.from(p.dataSync());
  }));
}

export function architecture(hp, data) {
  const V = data?.vocab.itos.length ?? 60;
  const D = hp.dModel;
  const T = hp.seqLen;
  const block = 4 * D * D + 2 * D * D * 2 + 2 * D + D * 2 + 4 * D;
  const rows = [
    { name: `Input · ${T} characters`, kind: 'input', inShape: [T], outShape: [T], params: 0,
      desc: { ko: `${T}글자의 번호가 한꺼번에 들어옵니다. RNN과 달리 순서대로 읽지 않고 모든 글자를 동시에 처리합니다.`, en: `${T} character IDs arrive all at once. Unlike an RNN, it does not read them in order — it processes every position in parallel.` } },
    { name: `Token + position embedding · ${D}`, kind: 'embed', inShape: [T], outShape: [T, D], params: V * D + T * D,
      desc: { ko: '글자 벡터에 "몇 번째 자리인지"를 나타내는 위치 벡터를 더합니다. 동시에 처리하기 때문에, 이것이 없으면 순서를 알 수 없습니다.', en: 'Each character vector gets a position vector added, saying where it sits. Since everything is processed at once, without this the model could not tell the order.' } },
  ];
  for (let i = 0; i < hp.blocks; i++) {
    rows.push({ name: `Block ${i + 1} · masked self-attention (${hp.heads} heads)`, kind: 'attn', inShape: [T, D], outShape: [T, D], params: 4 * D * D + 2 * D,
      desc: { ko: `각 글자가 "앞에 있는 글자들 중 어디를 참고할지"를 점수로 매겨(어텐션) 정보를 모읍니다. 헤드 ${hp.heads}개가 서로 다른 관점(예: 바로 앞 글자, 문장 앞의 목적어)으로 동시에 봅니다. 미래 글자는 가려져(마스크) 볼 수 없습니다.`, en: `Each character scores which earlier characters to draw on (attention) and gathers their information. ${hp.heads} heads look from different angles at once (say, the previous character, or the object earlier in the sentence). Future characters are masked out.` } });
    rows.push({ name: `Block ${i + 1} · feed-forward ${D * 2}`, kind: 'dense', inShape: [T, D], outShape: [T, D], params: block - (4 * D * D + 2 * D),
      desc: { ko: '모은 정보를 글자마다 따로 한 번 더 가공합니다. 앞뒤로 잔차 연결과 층 정규화가 있어 깊게 쌓아도 학습이 안정적입니다.', en: 'Processes the gathered information once more, separately for each position. Residual connections and layer normalization keep training stable as blocks stack up.' } });
  }
  rows.push({ name: `Output · softmax over ${V}`, kind: 'output', inShape: [T, D], outShape: [T, V], params: D * V + V,
    desc: { ko: '모든 자리에서 동시에 "다음 글자"를 예측합니다. 그래서 문장 하나로 여러 개의 학습 신호를 얻습니다.', en: 'Predicts the next character at every position at once, so a single window gives many training signals.' } });
  return rows;
}
