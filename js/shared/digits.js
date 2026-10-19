// Synthetic "handwritten" digits: stroke templates + random pose/thickness/wobble, rasterized
// with anti-aliasing. Pure JS so it runs in Node tests and the browser alike.
import { randn, mulberry32 } from '../core/rng.js';
import { splitIndices } from '../core/split.js';

export const DIGIT_SIZE = 20;

const arc = (cx, cy, rx, ry, a0, a1, steps = 14) =>
  Array.from({ length: steps + 1 }, (_, i) => {
    const a = a0 + ((a1 - a0) * i) / steps;
    return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)];
  });

// Each digit: list of polylines in a unit box (x right, y down).
const TEMPLATES = [
  [arc(0.5, 0.5, 0.27, 0.38, 0, Math.PI * 2, 20)],
  [[[0.36, 0.26], [0.52, 0.12], [0.52, 0.88]]],
  [[[0.26, 0.3], [0.36, 0.15], [0.55, 0.12], [0.72, 0.24], [0.7, 0.42], [0.26, 0.88], [0.78, 0.88]]],
  [[[0.26, 0.18], [0.5, 0.12], [0.72, 0.22], [0.68, 0.4], [0.45, 0.5]], [[0.45, 0.5], [0.72, 0.6], [0.74, 0.78], [0.5, 0.9], [0.25, 0.82]]],
  [[[0.62, 0.88], [0.62, 0.12], [0.22, 0.64], [0.8, 0.64]]],
  [[[0.72, 0.12], [0.31, 0.12], [0.28, 0.46], [0.55, 0.42], [0.73, 0.58], [0.7, 0.8], [0.5, 0.9], [0.25, 0.82]]],
  [[[0.68, 0.14], [0.45, 0.2], [0.3, 0.45], [0.28, 0.72], [0.45, 0.9], [0.68, 0.82], [0.72, 0.62], [0.55, 0.5], [0.3, 0.6]]],
  [[[0.22, 0.12], [0.78, 0.12], [0.42, 0.88]]],
  [arc(0.5, 0.3, 0.18, 0.17, 0, Math.PI * 2, 16), arc(0.5, 0.68, 0.22, 0.21, 0, Math.PI * 2, 16)],
  [[[0.7, 0.4], [0.5, 0.5], [0.3, 0.4], [0.32, 0.2], [0.5, 0.12], [0.7, 0.2], [0.7, 0.45], [0.62, 0.88]]],
];

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy || 1e-9;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  const qx = ax + t * dx - px;
  const qy = ay + t * dy - py;
  return Math.sqrt(qx * qx + qy * qy);
}

export function renderDigit(d, rand, size = DIGIT_SIZE, { speckle: useSpeckle = true, variation: v = 1 } = {}) {
  const rot = (rand() - 0.5) * 0.6 * v; // about ±17° at v=1
  const scale = 0.95 + (rand() - 0.5) * 0.3 * v;
  const shear = (rand() - 0.5) * 0.4 * v;
  const tx = (rand() - 0.5) * 4 * v;
  const ty = (rand() - 0.5) * 4 * v;
  const thick = 1.7 + (rand() - 0.5) * 1.0 * v;
  const wobble = 0.04 * v;
  const speckle = useSpeckle ? rand() * 0.25 : 0;
  const cos = Math.cos(rot);
  const sin = Math.sin(rot);
  const box = size * 0.8 * scale;
  const strokes = TEMPLATES[d].map((line) => line.map(([x, y]) => {
    const u = x - 0.5 + randn(rand) * wobble;
    const v = y - 0.5 + randn(rand) * wobble;
    const su = u + shear * v;
    return [size / 2 + tx + (su * cos - v * sin) * box, size / 2 + ty + (su * sin + v * cos) * box];
  }));
  const img = new Float32Array(size * size);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      let best = Infinity;
      for (const line of strokes) {
        for (let i = 1; i < line.length; i++) {
          const dist = segDist(px + 0.5, py + 0.5, line[i - 1][0], line[i - 1][1], line[i][0], line[i][1]);
          if (dist < best) best = dist;
        }
      }
      const ink = Math.max(0, Math.min(1, thick / 2 + 0.5 - best));
      const noise = useSpeckle && rand() < 0.06 ? rand() * speckle * 2 : 0;
      img[py * size + px] = Math.min(1, ink + noise);
    }
  }
  return img;
}

// n digits with labels cycling 0..9 (balanced).
export function makeDigits(n, rand, size = DIGIT_SIZE, opts) {
  const x = [];
  const y = [];
  for (let i = 0; i < n; i++) {
    const d = i % 10;
    x.push(renderDigit(d, rand, size, opts));
    y.push(d);
  }
  return { x, y };
}

// Turn a user's drawing (an RGBA ImageData-like {data,width,height}, ink = alpha) into a
// size×size image framed like the training digits: crop to the ink, fit, and center.
export function frameDrawing(src, size = DIGIT_SIZE) {
  const { data, width, height } = src;
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4 + 3] > 20) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  const out = new Float32Array(size * size);
  if (x1 < 0) return out;
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  const target = size * 0.62; // training digits are ~0.62·size tall
  const s = target / Math.max(bw, bh);
  const ox = size / 2 - (bw * s) / 2;
  const oy = size / 2 - (bh * s) / 2;
  // Area-average each output pixel from the source.
  for (let oyi = 0; oyi < size; oyi++) {
    for (let oxi = 0; oxi < size; oxi++) {
      const sx0 = x0 + (oxi - ox) / s;
      const sy0 = y0 + (oyi - oy) / s;
      const step = 1 / s;
      let sum = 0;
      let cnt = 0;
      for (let yy = sy0; yy < sy0 + step; yy += Math.max(1, step / 4)) {
        for (let xx = sx0; xx < sx0 + step; xx += Math.max(1, step / 4)) {
          const ix = Math.floor(xx);
          const iy = Math.floor(yy);
          cnt++;
          if (ix >= 0 && iy >= 0 && ix < width && iy < height) sum += data[(iy * width + ix) * 4 + 3] / 255;
        }
      }
      out[oyi * size + oxi] = cnt ? Math.min(1, (sum / cnt) * 1.3) : 0;
    }
  }
  return out;
}

// n digits split 70/15/15 into { train, val, test, meta.counts }, reproducible from seed.
export function makeDigitSplits(n, seed, opts) {
  const rand = mulberry32(seed);
  const all = makeDigits(n, rand, DIGIT_SIZE, opts);
  const idx = splitIndices(n, [0.7, 0.15, 0.15], rand);
  const pick = (ids) => ({ x: ids.map((i) => all.x[i]), y: ids.map((i) => all.y[i]) });
  const train = pick(idx.train);
  const val = pick(idx.val);
  const test = pick(idx.test);
  return { train, val, test, meta: { counts: { train: train.x.length, val: val.x.length, test: test.x.length } } };
}
