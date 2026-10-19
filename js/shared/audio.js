// Synthetic sound clips and a pure-JS short-time Fourier transform (spectrogram).
import { randn } from '../core/rng.js';

export const SR = 8000;
export const KINDS = ['tone_low', 'tone_high', 'chirp_up', 'chirp_down', 'two_beeps', 'noise'];

// A 0.5 s clip of the given kind with random pitch, timing, loudness and background noise.
export function synthClip(kind, rand, { noise } = {}) {
  const n = SR / 2;
  const out = new Float32Array(n);
  const amp = 0.35 + rand() * 0.45;
  const start = Math.floor(rand() * 0.12 * SR);
  const len = Math.floor((0.25 + rand() * 0.12) * SR);
  const bg = noise ?? 0.02 + rand() * 0.06;
  let phase = 0;
  const f0 = { tone_low: 200 + rand() * 150, tone_high: 900 + rand() * 500, chirp_up: 300 + rand() * 150, chirp_down: 1400 + rand() * 300, two_beeps: 650 + rand() * 350 }[kind];
  const f1 = kind === 'chirp_up' ? 1300 + rand() * 300 : kind === 'chirp_down' ? 300 + rand() * 150 : f0;
  for (let i = 0; i < n; i++) {
    let v = bg * randn(rand);
    const k = i - start;
    if (k >= 0 && k < len) {
      const t = k / len;
      const env = Math.min(1, t * 20, (1 - t) * 20); // quick fade in/out
      if (kind === 'noise') {
        v += amp * env * randn(rand) * 0.6;
      } else {
        const f = f0 + (f1 - f0) * t;
        phase += (2 * Math.PI * f) / SR;
        const gate = kind === 'two_beeps' ? (t < 0.4 || t > 0.6 ? 1 : 0) : 1;
        v += amp * env * gate * Math.sin(phase);
      }
    }
    out[i] = Math.max(-1, Math.min(1, v));
  }
  return out;
}

// In-place iterative radix-2 FFT on re/im arrays (length a power of two).
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const ang = (-2 * Math.PI) / size;
    for (let s = 0; s < n; s += size) {
      for (let k = 0; k < size / 2; k++) {
        const wr = Math.cos(ang * k);
        const wi = Math.sin(ang * k);
        const a = s + k;
        const b = a + size / 2;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// Magnitude spectrogram: [frames][win/2 + 1], Hann-windowed.
export function stft(signal, { win = 256, hop = 128 } = {}) {
  const frames = Math.floor((signal.length - win) / hop) + 1;
  const hann = Array.from({ length: win }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (win - 1)));
  const out = [];
  for (let f = 0; f < frames; f++) {
    const re = new Float64Array(win);
    const im = new Float64Array(win);
    for (let i = 0; i < win; i++) re[i] = signal[f * hop + i] * hann[i];
    fft(re, im);
    const mag = new Array(win / 2 + 1);
    for (let k = 0; k <= win / 2; k++) mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    out.push(mag);
  }
  return out;
}

// Model input: log spectrogram, frequency bins averaged in pairs (129 → 64), scaled to [0, 1] per clip.
export function features(signal) {
  const spec = stft(signal);
  const rows = spec.map((f) => Array.from({ length: 64 }, (_, k) => Math.log1p(10 * (f[2 * k] + f[2 * k + 1]) / 2)));
  const max = Math.max(1e-6, ...rows.flat());
  return rows.map((r) => r.map((v) => v / max));
}

// Resample a Float32Array from one rate to another (linear interpolation), e.g. microphone → 8 kHz.
export function resample(input, fromRate, toRate, length) {
  const out = new Float32Array(length);
  const ratio = fromRate / toRate;
  for (let i = 0; i < length; i++) {
    const x = i * ratio;
    const j = Math.floor(x);
    const a = input[j] ?? 0;
    const b = input[j + 1] ?? a;
    out[i] = a + (b - a) * (x - j);
  }
  return out;
}
