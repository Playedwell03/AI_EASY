// 2D toy datasets for binary classification. Coordinates are clipped to [-1, 1].
import { randn } from '../../core/rng.js';
import { splitIndices } from '../../core/split.js';
import { mulberry32 } from '../../core/rng.js';

const clip = (v) => Math.max(-1, Math.min(1, v));

export function makePoints(kind, n, noise, rand) {
  const x = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    let px;
    let py;
    if (kind === 'circle') {
      const r = label ? 0.55 + rand() * 0.4 : rand() * 0.4;
      const a = rand() * Math.PI * 2;
      px = r * Math.cos(a);
      py = r * Math.sin(a);
    } else if (kind === 'xor') {
      px = (rand() * 0.85 + 0.1) * (rand() < 0.5 ? -1 : 1);
      py = (rand() * 0.85 + 0.1) * (rand() < 0.5 ? -1 : 1);
      if ((px * py > 0 ? 0 : 1) !== label) py = -py;
    } else if (kind === 'spiral') {
      const t = rand();
      const r = 0.1 + t * 0.85;
      const a = t * 2.6 * Math.PI + label * Math.PI;
      px = r * Math.cos(a);
      py = r * Math.sin(a);
    } else {
      // moons
      const a = rand() * Math.PI;
      px = label ? 0.5 - Math.cos(a) * 0.55 : -0.25 + Math.cos(a) * 0.55;
      py = label ? 0.2 - Math.sin(a) * 0.5 : -0.15 + Math.sin(a) * 0.5;
    }
    x.push([clip(px + noise * randn(rand) * 0.5), clip(py + noise * randn(rand) * 0.5)]);
    y.push(label);
  }
  return { x, y };
}

export function makeData(hp) {
  const rand = mulberry32(hp.seed);
  const all = makePoints(hp.dataset, hp.n, hp.noise, rand);
  const idx = splitIndices(hp.n, [0.7, 0.15, 0.15], rand);
  const pick = (ids) => ({ x: ids.map((i) => all.x[i]), y: ids.map((i) => all.y[i]) });
  const train = pick(idx.train);
  const val = pick(idx.val);
  const test = pick(idx.test);
  return { train, val, test, meta: { counts: { train: train.x.length, val: val.x.length, test: test.x.length } } };
}
