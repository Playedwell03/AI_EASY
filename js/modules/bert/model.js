// A tiny BERT: bidirectional encoder trained to fill in masked characters (masked language modeling).
import { mulberry32 } from '../../core/rng.js';
import { splitIndices } from '../../core/split.js';
import { makeCorpus, buildVocab, encode } from '../../shared/corpus.js';
import { createEncoder, vari } from '../../shared/attention.js';

/**
 * Pick ~maskProb of the real (non-pad) tokens as targets. Of those: 80% → [MASK], 10% → random token,
 * 10% → unchanged. labels holds the original id at targets and -1 elsewhere. At least one target.
 */
export function maskTokens(ids, rand, { maskProb, maskId, padId, vocabSize }) {
  const input = ids.slice();
  const labels = ids.map(() => -1);
  const real = ids.map((id, i) => (id === padId ? -1 : i)).filter((i) => i >= 0);
  let chosen = real.filter(() => rand() < maskProb);
  if (!chosen.length && real.length) chosen = [real[Math.floor(rand() * real.length)]];
  for (const i of chosen) {
    labels[i] = ids[i];
    const r = rand();
    if (r < 0.8) input[i] = maskId;
    else if (r < 0.9) {
      let k = Math.floor(rand() * vocabSize);
      while (k === maskId || k === padId) k = Math.floor(rand() * vocabSize);
      input[i] = k;
    }
  }
  return { input, labels };
}

export function makeData(hp) {
  const sentences = makeCorpus(hp.corpus, hp.chars, 42);
  const { itos } = buildVocab(sentences.join(' '));
  itos.push('[PAD]', '[MASK]');
  const stoi = Object.fromEntries(itos.map((c, i) => [c, i]));
  const padId = stoi['[PAD]'];
  const maskId = stoi['[MASK]'];
  const longest = Math.max(...sentences.map((s) => [...s].length));
  const maxLen = Math.max(hp.maxLen, Math.ceil(longest / 8) * 8);
  const toIds = (s) => {
    const e = encode(s, stoi).slice(0, maxLen);
    while (e.length < maxLen) e.push(padId);
    return e;
  };
  const rand = mulberry32(hp.seed);
  const idx = splitIndices(sentences.length, [0.8, 0.1, 0.1], rand);
  const keep = Math.max(2, Math.round(idx.train.length * hp.corpusFraction));
  const opts = { maskProb: hp.maskProb, maskId, padId, vocabSize: itos.length };
  const part = (ids, fixedMasks) => {
    const text = ids.map((i) => sentences[i]);
    const out = { text, ids: text.map(toIds) };
    if (fixedMasks) {
      const r = mulberry32(hp.seed * 7 + text.length);
      const m = out.ids.map((x) => maskTokens(x, r, opts));
      out.input = m.map((x) => x.input);
      out.labels = m.map((x) => x.labels);
    }
    return out;
  };
  const train = part(idx.train.slice(0, keep), false);
  const val = part(idx.val, true);
  const test = part(idx.test, true);
  return {
    train, val, test, maxLen, vocab: { itos, stoi, padId, maskId }, maskOpts: opts,
    sentences: { train: train.text, val: val.text, test: test.text },
    meta: { counts: { train: train.ids.length, val: val.ids.length, test: test.ids.length } },
  };
}

export function createSession(tf, hp, data) {
  const { itos, stoi, padId, maskId } = data.vocab;
  const V = itos.length;
  const T = data.maxLen;
  const D = hp.dModel;
  const enc = createEncoder(tf, { vocab: V, dModel: D, heads: hp.heads, ff: D * 2, blocks: hp.blocks, maxLen: T, seed: hp.seed });
  const Wout = vari(tf, tf.randomNormal([D, V], 0, 1 / Math.sqrt(D), 'float32', hp.seed * 3 + 7));
  const bout = vari(tf, tf.zeros([V]));
  const vars = [...enc.variables, Wout, bout];
  const optimizer = tf.train.adam(hp.lr);
  const rand = mulberry32(hp.seed * 19 + 5);
  const special = new Set([padId, maskId]);

  const forward = (input) => {
    const padMask = input.notEqual(padId).toFloat();
    const { hidden, weights } = enc.apply(input, { causal: false, padMask });
    const [B] = input.shape;
    return { logits: hidden.reshape([B * T, D]).matMul(Wout).add(bout).reshape([B, T, V]), hidden, padMask, weights };
  };
  // Cross-entropy averaged over target positions only (labels -1 are ignored).
  const maskedLoss = (logits, labels) => {
    const w = labels.greaterEqual(0).toFloat();
    const lab = labels.maximum(tf.scalar(0, 'int32'));
    const picked = tf.logSoftmax(logits).mul(tf.oneHot(lab, V)).sum(-1);
    return picked.mul(w).sum().neg().div(w.sum().maximum(1));
  };
  const fixed = (p) => ({ input: tf.tensor2d(p.input, [p.input.length, T], 'int32'), labels: tf.tensor2d(p.labels, [p.labels.length, T], 'int32') });
  const S = { val: fixed(data.val), test: fixed(data.test) };

  async function evaluate({ input, labels }) {
    const t = tf.tidy(() => {
      const { logits } = forward(input);
      const loss = maskedLoss(logits, labels);
      const w = labels.greaterEqual(0).toFloat();
      const hit1 = logits.argMax(-1).equal(labels).toFloat().mul(w).sum();
      // In the top 5 ⇔ fewer than 5 characters score higher than the right one (cheaper than topk on WebGL).
      const lab = labels.maximum(tf.scalar(0, 'int32'));
      const right = logits.mul(tf.oneHot(lab, V)).sum(-1, true);
      const hit5 = logits.greater(right).toFloat().sum(-1).less(5).toFloat().mul(w).sum();
      return tf.stack([loss, hit1.div(w.sum()), hit5.div(w.sum())]);
    });
    const [loss, acc, top5] = await t.data();
    t.dispose();
    return { loss, acc, top5 };
  }

  let disposed = false;
  return {
    async epoch() {
      // Fresh masks every epoch (dynamic masking), so the model cannot memorize which slots are hidden.
      const masked = data.train.ids.map((ids) => maskTokens(ids, rand, data.maskOpts));
      const n = masked.length;
      const X = tf.tensor2d(masked.map((m) => m.input), [n, T], 'int32');
      const Y = tf.tensor2d(masked.map((m) => m.labels), [n, T], 'int32');
      for (let b = 0; b < n; b += hp.batchSize) {
        const size = Math.min(hp.batchSize, n - b);
        const cost = optimizer.minimize(() => {
          const { logits } = forward(X.slice([b, 0], [size, T]));
          return maskedLoss(logits, Y.slice([b, 0], [size, T]));
        }, true, vars);
        cost?.dispose();
        await tf.nextFrame(); // keep the page responsive (and spread out first-time GPU compiles)
      }
      const tr = await evaluate({ input: X, labels: Y });
      X.dispose();
      Y.dispose();
      const v = await evaluate(S.val);
      return { trainLoss: tr.loss, valLoss: v.loss, metrics: { acc: v.acc, top5: v.top5, train_acc: tr.acc } };
    },
    async evaluateTest() {
      const r = await evaluate(S.test);
      return { loss: r.loss, acc: r.acc, top5: r.top5 };
    },
    // Hide the characters at `positions` and return the top-5 guesses for each.
    fillMask(text, positions) {
      const chars = [...text].slice(0, T);
      const ids = chars.map((c) => stoi[c] ?? maskId);
      positions.forEach((p) => { if (p < ids.length) ids[p] = maskId; });
      while (ids.length < T) ids.push(padId);
      return tf.tidy(() => {
        const { logits } = forward(tf.tensor2d([ids], [1, T], 'int32'));
        const probs = tf.softmax(logits).arraySync()[0];
        return positions.map((p) => ({
          pos: p,
          original: chars[p],
          probs: probs[p],
          top: probs[p].map((q, k) => ({ c: itos[k], p: q, k })).filter((x) => !special.has(x.k)).sort((a, b) => b.p - a.p).slice(0, 5),
        }));
      });
    },
    // Average of the encoder's output vectors over the real characters: one vector per sentence.
    sentenceVector(text) {
      const ids = encode(text, stoi).slice(0, T);
      const len = ids.length;
      while (ids.length < T) ids.push(padId);
      return tf.tidy(() => Array.from(forward(tf.tensor2d([ids], [1, T], 'int32')).hidden.slice([0, 0, 0], [1, len, D]).mean(1).dataSync()));
    },
    attentionMaps(text) {
      const ids = [...text].slice(0, T).map((c) => stoi[c] ?? maskId);
      if (!ids.length) return [];
      return tf.tidy(() => enc.apply(tf.tensor2d([ids], [1, ids.length], 'int32')).weights.map((w) => w.arraySync()[0]));
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      optimizer.dispose();
      enc.dispose();
      Wout.dispose();
      bout.dispose();
      Object.values(S).forEach((p) => { p.input.dispose(); p.labels.dispose(); });
    },
  };
}

export function architecture(hp, data) {
  const V = data?.vocab.itos.length ?? 60;
  const T = data?.maxLen ?? hp.maxLen;
  const D = hp.dModel;
  const rows = [
    { name: `Input · sentence with [MASK]`, kind: 'input', inShape: [T], outShape: [T], params: 0,
      desc: { ko: `문장 하나(최대 ${T}글자)에서 15% 정도의 글자를 [MASK]로 가린 것이 들어옵니다. 짧은 문장은 [PAD]로 길이를 맞춥니다.`, en: `One sentence (up to ${T} characters) with about 15% of its characters hidden as [MASK]. Shorter sentences are padded with [PAD].` } },
    { name: `Token + position embedding · ${D}`, kind: 'embed', inShape: [T], outShape: [T, D], params: V * D + T * D,
      desc: { ko: '글자 벡터와 위치 벡터를 더합니다. GPT식 모델과 같습니다.', en: 'Character vectors plus position vectors — the same as a GPT-style model.' } },
  ];
  for (let i = 0; i < hp.blocks; i++) {
    rows.push({ name: `Block ${i + 1} · bidirectional self-attention (${hp.heads} heads)`, kind: 'attn', inShape: [T, D], outShape: [T, D], params: 4 * D * D + 2 * D,
      desc: { ko: 'GPT와 가장 큰 차이: 미래를 가리지 않습니다. 각 글자가 문장 앞뒤 모든 글자를 참고하므로, 빈칸의 양쪽 문맥을 함께 써서 맞힐 수 있습니다. [PAD] 자리는 참고하지 않도록 가립니다.', en: 'The key difference from GPT: nothing is hidden from the future. Every character can look at the whole sentence, before and after, so a blank is filled using context on both sides. [PAD] slots are masked out.' } });
    rows.push({ name: `Block ${i + 1} · feed-forward ${D * 2}`, kind: 'dense', inShape: [T, D], outShape: [T, D], params: 2 * D * D * 2 + 3 * D,
      desc: { ko: '글자마다 정보를 한 번 더 가공합니다.', en: 'Processes each position once more.' } });
  }
  rows.push({ name: `MLM head · softmax over ${V}`, kind: 'output', inShape: [T, D], outShape: [T, V], params: D * V + V,
    desc: { ko: '가려진 자리마다 원래 어떤 글자였는지 확률을 냅니다. 손실은 가려진 자리에서만 계산합니다.', en: 'For every hidden slot, gives a probability for each possible original character. The loss is computed only at hidden slots.' } });
  return rows;
}
