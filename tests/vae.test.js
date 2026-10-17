import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tf, trainModule, slow } from './helpers.js';
import vae from '../js/modules/vae/index.js';

test('makeData gives flattened digit images and a positive mean-image baseline', () => {
  const d = vae.makeData({ ...vae.defaults, n: 200 });
  assert.equal(d.train.x[0].length, 400);
  const b = vae.baselineLoss(vae.defaults, d);
  assert.ok(b > 0.05 && b < 1, `baseline ${b}`);
});

test('default VAE reconstructs well below the mean-image baseline', slow, async () => {
  const r = await trainModule(vae);
  const res = await r.session.evaluateTest();
  const baseline = vae.baselineLoss(r.hp, r.data);
  assert.ok(res.recon < 0.6 * baseline, `recon ${res.recon} vs baseline ${baseline}`);
  assert.equal(r.diagnosis.verdict, 'good');
  assert.ok(r.seconds < 120, `took ${r.seconds}s (regression guard; the 1-minute target is checked in the browser)`);
  const z = r.session.encode(r.data.test.x.slice(0, 5));
  assert.equal(z.length, 5);
  assert.equal(z[0].length, 2);
  const imgs = r.session.decode([[0, 0], [1, -1]]);
  assert.equal(imgs.length, 2);
  assert.equal(imgs[0].length, 400);
  r.session.dispose();
});

test('plain autoencoder mode trains without KL', slow, async () => {
  const r = await trainModule(vae, { variational: false, epochs: 5 });
  assert.equal(r.history.at(-1).metrics.kl, 0);
  r.session.dispose();
});

for (const p of vae.breakPresets) {
  test(`preset "${p.id}" is diagnosed as ${p.expect}`, slow, async () => {
    const before = tf.memory().numTensors;
    const r = await trainModule(vae, p.hp);
    r.session.dispose();
    assert.equal(r.diagnosis.verdict, p.expect);
    assert.equal(tf.memory().numTensors, before, 'session leaks tensors');
  });
}
