import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRunner } from '../js/core/runner.js';

function fakeSession({ nanAt } = {}) {
  const s = { epochs: 0, disposed: 0 };
  s.epoch = async () => {
    s.epochs++;
    const loss = nanAt === s.epochs ? NaN : 1 / s.epochs;
    return { trainLoss: loss, valLoss: loss, metrics: { acc: 0.5 } };
  };
  s.evaluateTest = async () => ({ acc: 0.5 });
  s.dispose = () => { s.disposed++; };
  return s;
}

const nextFrame = () => new Promise((r) => setImmediate(r));

test('starting twice runs only one loop', async () => {
  const sess = fakeSession();
  const r = createRunner({ createSession: () => sess, epochs: 5, nextFrame });
  await Promise.all([r.start(), r.start()]);
  assert.equal(sess.epochs, 5);
  assert.equal(r.history.length, 5);
  assert.equal(r.state, 'done');
});

test('reset disposes the session and clears history', async () => {
  const sess = fakeSession();
  const r = createRunner({ createSession: () => sess, epochs: 3, nextFrame });
  await r.start();
  r.reset();
  assert.equal(sess.disposed, 1);
  assert.equal(r.history.length, 0);
  assert.equal(r.state, 'idle');
  assert.equal(r.session, null);
});

test('non-finite loss stops training as diverged', async () => {
  const sess = fakeSession({ nanAt: 3 });
  const r = createRunner({ createSession: () => sess, epochs: 10, nextFrame });
  await r.start();
  assert.equal(r.state, 'diverged');
  assert.equal(r.history.length, 3);
  assert.equal(sess.epochs, 3);
});

test('pause then start continues the same history', async () => {
  const sess = fakeSession();
  const r = createRunner({ createSession: () => sess, epochs: 6, nextFrame });
  r.on('epoch', () => { if (r.history.length === 2) r.pause(); });
  await r.start();
  assert.equal(r.state, 'paused');
  assert.equal(r.history.length, 2);
  await r.start();
  assert.equal(r.history.length, 6);
  assert.equal(r.state, 'done');
});

test('reset while running stops the loop and never touches the old session again', async () => {
  const sess = fakeSession();
  const r = createRunner({ createSession: () => sess, epochs: 50, nextFrame });
  r.on('epoch', () => { if (r.history.length === 2) r.reset(); });
  await r.start();
  assert.equal(sess.epochs, 2);
  assert.equal(sess.disposed, 1);
  assert.equal(r.state, 'idle');
});

test('state events fire on transitions', async () => {
  const r = createRunner({ createSession: () => fakeSession(), epochs: 2, nextFrame });
  const states = [];
  r.on('state', (s) => states.push(s));
  await r.start();
  assert.deepEqual(states, ['running', 'done']);
});

test('start right after a mid-epoch reset begins a fresh session', async () => {
  const sessions = [];
  let release;
  const slow = () => {
    const s = fakeSession();
    const inner = s.epoch;
    s.epoch = async () => { if (sessions.length === 1) await new Promise((r) => { release = r; }); return inner(); };
    sessions.push(s);
    return s;
  };
  const r = createRunner({ createSession: slow, epochs: 3, nextFrame });
  const first = r.start();
  await nextFrame();
  r.reset(); // first session is mid-epoch
  const second = r.start();
  release();
  await Promise.all([first, second]);
  assert.equal(sessions.length, 2);
  assert.equal(sessions[0].disposed, 1);
  assert.equal(sessions[1].epochs, 3);
  assert.equal(r.history.length, 3);
  assert.equal(r.state, 'done');
});

test('an epoch that throws ends in error state with the error kept', async () => {
  const sess = fakeSession();
  sess.epoch = async () => { throw new Error('boom'); };
  const r = createRunner({ createSession: () => sess, epochs: 3, nextFrame });
  await r.start();
  assert.equal(r.state, 'error');
  assert.equal(r.error.message, 'boom');
  r.reset();
  assert.equal(r.error, null);
});
