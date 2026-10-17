// Shared helpers for module tests: run a module end-to-end in Node with the tfjs CPU backend.
import * as tf from '@tensorflow/tfjs';
import { diagnoseCurves } from '../js/core/diagnose.js';

await tf.setBackend('cpu');
export { tf };

// Training-quality tests take minutes; run them with SLOW=1 (npm run test:slow).
export const slow = { skip: process.env.SLOW === '1' ? false : 'training test — run with SLOW=1' };

// Node's 'cpu' backend is the same engine a browser without WebGL uses, so tests run the
// module's cpuDefaults — exactly what such a browser would run.
export async function trainModule(def, overrides = {}) {
  const hp = { ...def.defaults, ...def.cpuDefaults, ...overrides };
  const data = def.makeData(hp);
  const session = def.createSession(tf, hp, data);
  const history = [];
  const t0 = Date.now();
  for (let i = 0; i < hp.epochs; i++) {
    const r = await session.epoch();
    history.push(r);
    if (!Number.isFinite(r.trainLoss) || !Number.isFinite(r.valLoss)) break;
  }
  const seconds = (Date.now() - t0) / 1000;
  const diagnosis = diagnoseCurves(history, { baselineLoss: def.baselineLoss(hp, data) });
  return { hp, data, session, history, diagnosis, seconds };
}
