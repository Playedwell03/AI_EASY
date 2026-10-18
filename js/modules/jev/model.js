// A Jev-style decision model: a state (form fields) + typed questions → calibrated answers.
// Tokens: 5 field tokens + 3 question tokens ([Q yes/no], [Q choice], [Q score]) → Transformer encoder →
// each question token's output goes to its own typed head.
import { mulberry32, shuffle } from '../../core/rng.js';
import { ece, reliabilityBins } from '../../core/metrics.js';
import { initBlockParams, transformerBlock, layerNorm, disposeParams, vari } from '../../shared/attention.js';
import { FIELDS, makeData, baseRateLoss } from './data.js';

export { makeData, baseRateLoss };

const NUM = FIELDS.filter((f) => f.type === 'num');
export const TOKENS = [...FIELDS.map((f) => f.label), { ko: '[Q 예/아니오]', en: '[Q yes/no]' }, { ko: '[Q 선택]', en: '[Q choice]' }, { ko: '[Q 점수]', en: '[Q score]' }];

const norm = (f, v) => ((v - f.min) / (f.max - f.min)) * 2 - 1;

export function createSession(tf, hp, data) {
  const D = hp.dModel;
  const s0 = hp.seed * 31;
  const rnd = (shape, std, k) => vari(tf, tf.randomNormal(shape, 0, std, 'float32', s0 + k));
  const P = {
    fieldEmb: rnd([FIELDS.length, D], 0.1, 1),
    valW: rnd([NUM.length, D], 0.5, 2),
    valB: rnd([NUM.length, D], 0.1, 3),
    purposeEmb: rnd([FIELDS[4].options.length, D], 0.5, 4),
    qEmb: rnd([3, D], 0.5, 5),
    lnG: vari(tf, tf.ones([D])),
    lnB: vari(tf, tf.zeros([D])),
    Wy: rnd([D, 1], 1 / Math.sqrt(D), 6),
    by: vari(tf, tf.zeros([1])),
    Wc: rnd([D, 3], 1 / Math.sqrt(D), 7),
    bc: vari(tf, tf.zeros([3])),
    Ws: rnd([D, 1], 1 / Math.sqrt(D), 8),
    bs: vari(tf, tf.zeros([1])),
  };
  const blocks = Array.from({ length: hp.blocks }, (_, i) => initBlockParams(tf, D, hp.heads, D * 2, hp.seed * 17 + i));
  const vars = [...Object.values(P), ...blocks.flatMap((b) => Object.values(b))];
  const optimizer = tf.train.adam(hp.lr);
  const rand = mulberry32(hp.seed * 3 + 1);

  const toInputs = (fields) => ({
    num: tf.tensor2d(fields.map((f) => NUM.map((d) => norm(d, f[d.key]))), [fields.length, NUM.length]),
    purpose: tf.tensor1d(fields.map((f) => f.purpose), 'int32'),
  });
  const toTargets = (answers) => ({
    yes: tf.tensor2d(answers.map((a) => [a.approve]), [answers.length, 1]),
    route: tf.tidy(() => tf.oneHot(tf.tensor1d(answers.map((a) => a.route), 'int32'), 3).toFloat()),
    risk: tf.tensor2d(answers.map((a) => [a.risk]), [answers.length, 1]),
  });
  const S = Object.fromEntries(['train', 'val', 'test'].map((k) => [k, { ...toInputs(data[k].fields), ...toTargets(data[k].answers) }]));

  // Returns { yesLogit [B,1], routeLogits [B,3], score [B,1], weights }.
  function forward(num, purpose) {
    const B = num.shape[0];
    const numTok = num.expandDims(2).mul(P.valW).add(P.valB); // [B,4,D]
    const purTok = tf.gather(P.purposeEmb, purpose).expandDims(1); // [B,1,D]
    const fieldTok = tf.concat([numTok, purTok], 1).add(P.fieldEmb); // [B,5,D]
    const qTok = P.qEmb.expandDims(0).tile([B, 1, 1]); // [B,3,D]
    let x = tf.concat([fieldTok, qTok], 1); // [B,8,D]
    const weights = [];
    for (const b of blocks) {
      const r = transformerBlock(tf, b, x, { heads: hp.heads });
      x = r.out;
      weights.push(r.weights);
    }
    x = layerNorm(tf, x, P.lnG, P.lnB);
    const q = (i) => x.slice([0, 5 + i, 0], [B, 1, D]).reshape([B, D]);
    return {
      yesLogit: q(0).matMul(P.Wy).add(P.by),
      routeLogits: q(1).matMul(P.Wc).add(P.bc),
      score: q(2).matMul(P.Ws).add(P.bs).sigmoid(),
      weights,
    };
  }

  const lossOf = (o, t) => tf.losses.sigmoidCrossEntropy(t.yes, o.yesLogit)
    .add(tf.losses.softmaxCrossEntropy(t.route, o.routeLogits))
    .add(tf.losses.meanSquaredError(t.risk, o.score));

  async function evaluate(split) {
    const t = S[split];
    const r = tf.tidy(() => {
      const o = forward(t.num, t.purpose);
      return [lossOf(o, t), o.yesLogit.sigmoid(), o.routeLogits.argMax(-1), o.score];
    });
    const loss = (await r[0].data())[0];
    const pYes = Array.from(await r[1].data());
    const route = Array.from(await r[2].data());
    const score = Array.from(await r[3].data());
    tf.dispose(r);
    const ans = data[split].answers;
    const yes = ans.map((a) => a.approve === 1);
    const acc = pYes.filter((p, i) => (p >= 0.5) === yes[i]).length / ans.length;
    const routeAcc = route.filter((k, i) => k === ans[i].route).length / ans.length;
    const scoreMae = score.reduce((s, v, i) => s + Math.abs(v - ans[i].risk), 0) / ans.length;
    return { loss, acc, routeAcc, scoreMae, ece: ece(pYes, yes), pYes, yes };
  }

  let disposed = false;
  let lastVal = null;
  return {
    async epoch({ cancelled } = {}) {
      const n = data.train.fields.length;
      const order = shuffle(Array.from({ length: n }, (_, i) => i), rand);
      for (let b = 0; b < n; b += hp.batchSize) {
        if (cancelled?.()) return null;
        const ids = tf.tensor1d(order.slice(b, b + hp.batchSize), 'int32');
        const cost = optimizer.minimize(() => {
          const g = (x) => tf.gather(x, ids);
          const t = { yes: g(S.train.yes), route: g(S.train.route), risk: g(S.train.risk) };
          return lossOf(forward(g(S.train.num), g(S.train.purpose)), t);
        }, true, vars);
        ids.dispose();
        cost?.dispose();
        await tf.nextFrame();
      }
      const tr = await evaluate('train');
      const v = await evaluate('val');
      lastVal = v;
      return { trainLoss: tr.loss, valLoss: v.loss, metrics: { acc: v.acc, routeAcc: v.routeAcc, scoreMae: v.scoreMae, ece: v.ece, train_acc: tr.acc } };
    },
    async evaluateTest() {
      const r = await evaluate('test');
      return { loss: r.loss, acc: r.acc, routeAcc: r.routeAcc, scoreMae: r.scoreMae, ece: r.ece };
    },
    // Reliability bins of the yes/no answer on validation data (latest epoch).
    calibration(nBins = 10) {
      return lastVal ? reliabilityBins(lastVal.pYes, lastVal.yes, nBins) : [];
    },
    async calibrationOn(split, nBins = 10) {
      const r = await evaluate(split);
      return { bins: reliabilityBins(r.pYes, r.yes, nBins), ece: r.ece, pYes: r.pYes, yes: r.yes };
    },
    predict(fields) {
      return tf.tidy(() => {
        const inp = toInputs(fields);
        const o = forward(inp.num, inp.purpose);
        const yes = o.yesLogit.sigmoid().dataSync();
        const route = tf.softmax(o.routeLogits).arraySync();
        const score = o.score.dataSync();
        return fields.map((_, i) => ({ approve: yes[i], route: route[i], risk: score[i] }));
      });
    },
    attention(fields) {
      return tf.tidy(() => {
        const inp = toInputs([fields]);
        return forward(inp.num, inp.purpose).weights.map((w) => w.arraySync()[0]);
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      optimizer.dispose();
      Object.values(P).forEach((v) => v.dispose());
      blocks.forEach(disposeParams);
      Object.values(S).forEach((p) => Object.values(p).forEach((t) => t.dispose()));
    },
  };
}

export function architecture(hp) {
  const D = hp.dModel;
  const rows = [
    { name: 'State · 5 fields', kind: 'input', inShape: [5], outShape: [5], params: 0,
      desc: { ko: '신청서 한 장(상태): 소득, 부채, 근속 연수, 연체 횟수, 대출 목적.', en: 'One application form (the state): income, debt, years employed, missed payments, purpose.' } },
    { name: `Field tokens · ${D}`, kind: 'embed', inShape: [5], outShape: [5, D], params: 5 * D + 2 * 4 * D + 4 * D,
      desc: { ko: '항목마다 "어떤 항목인지" 벡터와 "값" 벡터를 더해 토큰 하나를 만듭니다. 숫자는 크기에 비례한 벡터로, 목적(범주)은 범주별 벡터로 바꿉니다. 글자 대신 표의 칸이 토큰이 되는 셈입니다.', en: 'Each field becomes one token: a "which field" vector plus a "value" vector — numbers scale a learned vector, the category picks its own vector. Table cells play the role that characters played before.' } },
    { name: `Typed question tokens · 3`, kind: 'embed', inShape: [3], outShape: [3, D], params: 3 * D,
      desc: { ko: '질문 세 개를 토큰으로 붙입니다: [예/아니오] 승인할까?, [선택] 어느 창구로?, [점수] 위험도는? 질문의 "형식"이 정해져 있어서 답의 형식도 정해집니다.', en: 'Three questions are appended as tokens: [yes/no] approve?, [choice] which queue?, [score] how risky? Because the question type is fixed, so is the answer\'s format.' } },
  ];
  for (let i = 0; i < hp.blocks; i++) {
    rows.push({ name: `Block ${i + 1} · self-attention (${hp.heads} heads) + FFN`, kind: 'attn', inShape: [8, D], outShape: [8, D], params: 4 * D * D + 2 * D * D * 2 + 7 * D,
      desc: { ko: '질문 토큰이 항목 토큰들을 어텐션으로 살펴 답에 필요한 정보를 모읍니다. 예를 들어 [예/아니오] 질문은 연체 횟수와 소득을 많이 볼 수 있습니다(테스트 탭에서 확인).', en: 'Question tokens attend over the field tokens to gather what they need — the yes/no question may look mostly at missed payments and income (see the Test tab).' } });
  }
  rows.push(
    { name: 'Head · yes/no → sigmoid', kind: 'output', inShape: [D], outShape: [1], params: D + 1,
      desc: { ko: '[예/아니오] 질문 토큰의 출력 → 승인 확률(0~1). 이 확률이 "믿을 만한지"가 이 모듈의 핵심입니다.', en: 'Output of the yes/no token → probability of approval (0–1). Whether this probability can be trusted is the heart of this module.' } },
    { name: 'Head · choice → softmax(3)', kind: 'output', inShape: [D], outShape: [3], params: D * 3 + 3,
      desc: { ko: '[선택] 질문 토큰의 출력 → 일반 처리 / 심사역 검토 / 거절 안내 세 창구의 확률.', en: 'Output of the choice token → probabilities for standard / manual review / decline queues.' } },
    { name: 'Head · score → sigmoid', kind: 'output', inShape: [D], outShape: [1], params: D + 1,
      desc: { ko: '[점수] 질문 토큰의 출력 → 위험도 점수(0~1).', en: 'Output of the score token → a risk score (0–1).' } },
  );
  return rows;
}
