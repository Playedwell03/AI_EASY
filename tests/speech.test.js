import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import { mulberry32 } from '../js/core/rng.js';
import { synthClip, stft, KINDS, SR, features } from '../js/shared/audio.js';
import speech from '../js/modules/speech/index.js';

test('synthClip makes 0.5 s clips in [-1, 1] for every kind, reproducibly', () => {
  for (const kind of KINDS) {
    const a = synthClip(kind, mulberry32(4));
    assert.equal(a.length, SR / 2);
    assert.ok(a.every((v) => v >= -1 && v <= 1), kind);
    assert.deepEqual(a, synthClip(kind, mulberry32(4)));
  }
});

test('stft of a 440 Hz sine peaks at the matching bin', () => {
  const sig = Float32Array.from({ length: 4000 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / SR));
  const spec = stft(sig, { win: 256, hop: 128 });
  assert.equal(spec.length, 30);
  assert.equal(spec[0].length, 129);
  const frame = spec[10];
  const peak = frame.indexOf(Math.max(...frame));
  assert.ok(Math.abs(peak - Math.round((440 / SR) * 256)) <= 1, `peak bin ${peak}`);
});

test('chirps rise and fall in frequency', () => {
  const peakTrack = (kind) => {
    const spec = stft(synthClip(kind, mulberry32(9), { noise: 0 }), { win: 256, hop: 128 });
    const loud = spec.map((f) => ({ e: f.reduce((a, b) => a + b, 0), p: f.indexOf(Math.max(...f)) })).filter((f, _, all) => f.e > 0.5 * Math.max(...all.map((x) => x.e)));
    return [loud[0].p, loud.at(-1).p];
  };
  const [upStart, upEnd] = peakTrack('chirp_up');
  const [downStart, downEnd] = peakTrack('chirp_down');
  assert.ok(upEnd > upStart, `up ${upStart}→${upEnd}`);
  assert.ok(downEnd < downStart, `down ${downStart}→${downEnd}`);
});

test('features are a [frames][64] grid scaled to [0, 1]', () => {
  const f = features(synthClip('tone_low', mulberry32(1)));
  assert.equal(f.length, 30);
  assert.equal(f[0].length, 64);
  const all = f.flat();
  assert.ok(Math.max(...all) <= 1 && Math.min(...all) >= 0);
});

test('makeData gives balanced classes and split counts', () => {
  const d = speech.makeData({ ...speech.defaults, n: 120 });
  assert.equal(d.meta.counts.train + d.meta.counts.val + d.meta.counts.test, 120);
  assert.equal(d.train.x[0].length, 30);
});

test('default sound classifier reaches 90% test accuracy', slow, async () => {
  const r = await trainModule(speech);
  const res = await r.session.evaluateTest();
  r.session.dispose();
  assert.ok(res.acc >= 0.9, `acc ${res.acc}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard)`);
});

for (const p of speech.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(speech, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
