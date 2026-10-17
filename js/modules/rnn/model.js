import { layersSession, evalCategorical, init } from '../../core/tfSession.js';
import { makeTextData, sampleIndex, encode } from '../../shared/corpus.js';

export function makeData(hp) {
  return makeTextData({ lang: hp.corpus, chars: hp.chars, seqLen: hp.seqLen, seed: hp.seed, fraction: hp.corpusFraction, stride: 2 });
}

export function buildModel(tf, hp, V) {
  const m = tf.sequential();
  m.add(tf.layers.embedding({ inputDim: V, outputDim: hp.embed, inputLength: hp.seqLen, embeddingsInitializer: tf.initializers.randomUniform({ minval: -0.05, maxval: 0.05, seed: hp.seed }) }));
  const rnnOpts = { units: hp.units, kernelInitializer: init(tf, hp.seed, 1), recurrentInitializer: tf.initializers.orthogonal({ seed: hp.seed }) };
  m.add(hp.cell === 'gru' ? tf.layers.gru(rnnOpts) : tf.layers.lstm(rnnOpts));
  m.add(tf.layers.dense({ units: V, activation: 'softmax', kernelInitializer: init(tf, hp.seed, 2) }));
  m.compile({ optimizer: tf.train.adam(hp.lr), loss: 'categoricalCrossentropy' });
  return m;
}

// Shared by the RNN and Transformer modules: next-token helpers on top of a probs(ids) function.
export function addGeneration(s, data, seqLen, probsFor) {
  const { stoi, itos } = data.vocab;
  const pad = stoi[' '] ?? 0;
  const window = (ids) => {
    const w = ids.slice(-seqLen);
    while (w.length < seqLen) w.unshift(pad);
    return w;
  };
  s.nextProbs = (text) => probsFor(window(encode(text, stoi)));
  s.generate = (prompt, n, temperature, rand) => {
    const ids = encode(prompt, stoi);
    let out = '';
    for (let i = 0; i < n; i++) {
      const k = sampleIndex(probsFor(window(ids)), temperature, rand);
      ids.push(k);
      out += itos[k];
    }
    return out;
  };
  s.genLog = [];
  return s;
}

export function createSession(tf, hp, data) {
  const V = data.vocab.itos.length;
  const toT = (part) => ({
    x: tf.tensor2d(part.x, [part.x.length, hp.seqLen]),
    y: tf.tidy(() => tf.oneHot(tf.tensor1d(part.y, 'int32'), V).toFloat()),
  });
  const s = layersSession(tf, {
    model: buildModel(tf, hp, V),
    train: toT(data.train), val: toT(data.val), test: toT(data.test),
    batchSize: hp.batchSize, seed: hp.seed,
    async evaluate(m, x, y) {
      const r = await evalCategorical(tf, m, x, y);
      return { loss: r.loss, metrics: { ...r.metrics, perplexity: Math.exp(r.loss) } };
    },
  });
  return addGeneration(s, data, hp.seqLen, (ids) => tf.tidy(() => Array.from(s.model.predict(tf.tensor2d([ids], [1, hp.seqLen])).dataSync())));
}

export function architecture(hp, data) {
  const V = data?.vocab.itos.length ?? 60;
  const gates = hp.cell === 'gru' ? 3 : 4;
  const cell = hp.cell === 'gru' ? 'GRU' : 'LSTM';
  return [
    { name: `Input · ${hp.seqLen} characters`, kind: 'input', inShape: [hp.seqLen], outShape: [hp.seqLen], params: 0,
      desc: { ko: `직전 ${hp.seqLen}글자의 번호(토큰 ID)가 들어옵니다. 각 글자는 어휘 목록에서의 위치 번호로 바뀝니다.`, en: `The IDs (token numbers) of the previous ${hp.seqLen} characters. Each character becomes its position in the vocabulary.` } },
    { name: `Embedding · ${hp.embed}`, kind: 'embed', inShape: [hp.seqLen], outShape: [hp.seqLen, hp.embed], params: V * hp.embed,
      desc: { ko: `글자 번호를 ${hp.embed}개의 숫자로 이루어진 벡터로 바꿉니다. 처음엔 무작위지만, 학습하면서 비슷하게 쓰이는 글자끼리 비슷한 벡터를 갖게 됩니다.`, en: `Turns each character ID into a vector of ${hp.embed} numbers. They start random; during training, characters used in similar ways get similar vectors.` } },
    { name: `${cell} · ${hp.units}`, kind: 'rnn', inShape: [hp.seqLen, hp.embed], outShape: [hp.units], params: gates * (hp.units * (hp.units + hp.embed) + hp.units),
      desc: hp.cell === 'gru'
        ? { ko: 'GRU는 글자를 하나씩 읽으며 기억(은닉 상태)을 갱신합니다. 업데이트 문과 리셋 문 두 개로 무엇을 기억할지 고릅니다. LSTM보다 단순하고 빠릅니다.', en: 'A GRU reads one character at a time and updates its memory (hidden state). Two gates — update and reset — choose what to keep. Simpler and faster than an LSTM.' }
        : { ko: 'LSTM은 글자를 하나씩 순서대로 읽으며 기억을 갱신합니다. 입력·망각·출력 문(gate)이 있어서 "문장 앞부분의 목적어" 같은 정보를 뒤까지 들고 갈 수 있습니다.', en: 'An LSTM reads characters one by one, updating its memory. Input, forget, and output gates let it carry information — like the object earlier in a sentence — further along.' } },
    { name: `Dense ${V} · softmax`, kind: 'output', inShape: [hp.units], outShape: [V], params: hp.units * V + V,
      desc: { ko: '어휘의 모든 글자 각각에 대해 "다음 글자일 확률"을 냅니다. 여기서 하나를 뽑아 이어 붙이면 문장이 생성됩니다.', en: 'Gives every character in the vocabulary a probability of being next. Picking one and appending it, again and again, generates text.' } },
  ];
}
