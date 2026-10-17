// Minimal Transformer pieces written with plain tf ops so attention weights can be shown.
// Shapes: x is [batch, time, dModel]; attention weights are [batch, heads, time, time].

function normal(tf, shape, std, seed) {
  return tf.variable(tf.randomNormal(shape, 0, std, 'float32', seed));
}

export function initBlockParams(tf, dModel, heads, ff, seed) {
  const s = seed * 1009;
  const std = 1 / Math.sqrt(dModel);
  return {
    Wq: normal(tf, [dModel, dModel], std, s + 1),
    Wk: normal(tf, [dModel, dModel], std, s + 2),
    Wv: normal(tf, [dModel, dModel], std, s + 3),
    Wo: normal(tf, [dModel, dModel], std / Math.sqrt(2), s + 4),
    ln1g: tf.variable(tf.ones([dModel])),
    ln1b: tf.variable(tf.zeros([dModel])),
    ln2g: tf.variable(tf.ones([dModel])),
    ln2b: tf.variable(tf.zeros([dModel])),
    W1: normal(tf, [dModel, ff], std, s + 5),
    b1: tf.variable(tf.zeros([ff])),
    W2: normal(tf, [ff, dModel], 1 / Math.sqrt(ff) / Math.sqrt(2), s + 6),
    b2: tf.variable(tf.zeros([dModel])),
  };
}

export function disposeParams(p) {
  for (const v of Object.values(p)) v.dispose();
}

export function layerNorm(tf, x, g, b) {
  const { mean, variance } = tf.moments(x, -1, true);
  return x.sub(mean).div(variance.add(1e-5).sqrt()).mul(g).add(b);
}

// [B,T,D] @ [D,E] → [B,T,E]
function dense(x, W) {
  const [B, T, D] = x.shape;
  return x.reshape([B * T, D]).matMul(W).reshape([B, T, W.shape[1]]);
}

/**
 * opts: { heads, causal, padMask } — padMask is [B, T] with 1 for real tokens, 0 for padding.
 * Returns { out: [B,T,D], weights: [B,H,T,T] }.
 */
export function multiHeadAttention(tf, p, x, { heads, causal = false, padMask = null }) {
  const [B, T, D] = x.shape;
  const dh = D / heads;
  const split = (t) => t.reshape([B, T, heads, dh]).transpose([0, 2, 1, 3]); // [B,H,T,dh]
  const q = split(dense(x, p.Wq));
  const k = split(dense(x, p.Wk));
  const v = split(dense(x, p.Wv));
  let scores = tf.matMul(q, k, false, true).div(Math.sqrt(dh)); // [B,H,T,T]
  if (causal) {
    // -1e9 above the diagonal: position i may only look at positions ≤ i.
    const mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
    scores = scores.add(tf.scalar(1).sub(mask).mul(-1e9));
  }
  if (padMask) scores = scores.add(tf.scalar(1).sub(padMask).reshape([B, 1, 1, T]).mul(-1e9));
  const weights = tf.softmax(scores);
  const ctx = tf.matMul(weights, v).transpose([0, 2, 1, 3]).reshape([B, T, D]);
  return { out: dense(ctx, p.Wo), weights };
}

// Pre-LayerNorm block: x + Attention(LN(x)), then x + FeedForward(LN(x)).
export function transformerBlock(tf, p, x, opts) {
  const a = multiHeadAttention(tf, p, layerNorm(tf, x, p.ln1g, p.ln1b), opts);
  const h = x.add(a.out);
  const f = dense(dense(layerNorm(tf, h, p.ln2g, p.ln2b), p.W1).add(p.b1).relu(), p.W2).add(p.b2);
  return { out: h.add(f), weights: a.weights };
}

/**
 * A small token encoder/decoder: token + learned position embeddings → blocks → final LayerNorm.
 * Used by the Transformer LM (causal), BERT (bidirectional) and Jev-style (bidirectional) modules.
 */
export function createEncoder(tf, { vocab, dModel, heads, ff, blocks, maxLen, seed }) {
  const params = {
    tok: normal(tf, [vocab, dModel], 0.1, seed * 7 + 1),
    pos: normal(tf, [maxLen, dModel], 0.1, seed * 7 + 2),
    lnfg: tf.variable(tf.ones([dModel])),
    lnfb: tf.variable(tf.zeros([dModel])),
  };
  const layers = Array.from({ length: blocks }, (_, i) => initBlockParams(tf, dModel, heads, ff, seed * 13 + i));
  return {
    params,
    layers,
    variables: [...Object.values(params), ...layers.flatMap((l) => Object.values(l))],
    // ids: int tensor [B,T]. Returns { hidden: [B,T,D], weights: [blocks] of [B,H,T,T] }.
    apply(ids, { causal = false, padMask = null } = {}) {
      const T = ids.shape[1];
      let x = tf.gather(params.tok, ids).add(params.pos.slice([0, 0], [T, dModel]));
      const weights = [];
      for (const l of layers) {
        const r = transformerBlock(tf, l, x, { heads, causal, padMask });
        x = r.out;
        weights.push(r.weights);
      }
      return { hidden: layerNorm(tf, x, params.lnfg, params.lnfb), weights };
    },
    dispose() {
      Object.values(params).forEach((v) => v.dispose());
      layers.forEach(disposeParams);
    },
  };
}
