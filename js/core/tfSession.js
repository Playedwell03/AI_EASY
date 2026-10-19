// A training Session for ordinary tf.LayersModel classifiers/regressors.
// Each epoch shuffles the training set with a seeded RNG so runs are reproducible.
import { mulberry32, shuffle } from './rng.js';

/**
 * opts: {
 *   model: compiled tf.LayersModel,
 *   train: { x: Tensor, y: Tensor }, val: { x, y }, test: { x, y },
 *   batchSize, seed,
 *   evaluate(model, x, y) -> Promise<{ loss: number, metrics: object }>   // returns plain numbers, disposes its own tensors
 * }
 */
export function layersSession(tf, { model, train, val, test, batchSize, seed, evaluate }) {
  const rand = mulberry32(seed * 7919 + 13);
  const n = train.x.shape[0];
  // Measuring training loss on a capped subset keeps per-epoch evaluation cheap.
  const EVAL_CAP = 600;
  const trainEval = n > EVAL_CAP
    ? { x: train.x.slice(0, EVAL_CAP), y: train.y.slice(0, EVAL_CAP) }
    : train;
  let disposed = false;

  return {
    model,
    async epoch({ cancelled } = {}) {
      const order = tf.tensor1d(shuffle(Array.from({ length: n }, (_, i) => i), rand), 'int32');
      const xs = tf.gather(train.x, order);
      const ys = tf.gather(train.y, order);
      order.dispose();
      let trainLoss;
      try {
        const h = await model.fit(xs, ys, {
          epochs: 1, batchSize, shuffle: false, verbose: 0,
          callbacks: { onBatchEnd: () => { if (cancelled?.()) model.stopTraining = true; } },
        });
        trainLoss = h.history.loss[0];
      } finally {
        xs.dispose();
        ys.dispose();
      }
      if (cancelled?.()) return null;
      const v = await evaluate(model, val.x, val.y);
      const tr = await evaluate(model, trainEval.x, trainEval.y);
      // Report the training loss measured after the epoch so it is comparable with validation loss.
      return {
        trainLoss: Number.isFinite(trainLoss) ? tr.loss : trainLoss,
        valLoss: v.loss,
        metrics: { ...prefix(tr.metrics, 'train_'), ...v.metrics },
      };
    },
    async evaluateTest() {
      const r = await evaluate(model, test.x, test.y);
      return { loss: r.loss, ...r.metrics };
    },
    predict(x) {
      return model.predict(x);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      model.optimizer?.dispose();
      model.dispose();
      if (trainEval !== train) { trainEval.x.dispose(); trainEval.y.dispose(); }
      for (const part of [train, val, test]) {
        part.x.dispose();
        part.y.dispose();
      }
    },
  };
}

function prefix(obj, p) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[p + k] = v;
  return out;
}

// Evaluate helpers returning plain numbers.
export async function evalBinary(tf, model, x, y) {
  const [loss, acc] = tf.tidy(() => {
    const p = model.predict(x);
    const l = tf.metrics.binaryCrossentropy(y, p).mean();
    const a = tf.metrics.binaryAccuracy(y, p).mean();
    return [l, a];
  });
  const out = { loss: (await loss.data())[0], metrics: { acc: (await acc.data())[0] } };
  loss.dispose();
  acc.dispose();
  return out;
}

export async function evalCategorical(tf, model, x, y) {
  const [loss, acc] = tf.tidy(() => {
    const p = model.predict(x);
    const l = tf.metrics.categoricalCrossentropy(y, p).mean();
    const a = tf.metrics.categoricalAccuracy(y, p).mean();
    return [l, a];
  });
  const out = { loss: (await loss.data())[0], metrics: { acc: (await acc.data())[0] } };
  loss.dispose();
  acc.dispose();
  return out;
}

// Seeded Glorot initializer for layer i of a model.
export function init(tf, seed, i) {
  return tf.initializers.glorotUniform({ seed: seed * 101 + i });
}
