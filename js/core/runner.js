// Runs a module's training session epoch by epoch, yielding to the UI between epochs.
// Guarantees a single loop at a time; reset() cancels and disposes cleanly even mid-epoch.

export function createRunner({ createSession, epochs, nextFrame }) {
  let session = null;
  let history = [];
  let state = 'idle';
  let loop = null; // the in-flight loop promise, if any
  let gen = 0; // bumped by reset() so a running loop knows it was cancelled
  let pauseRequested = false;
  let error = null;
  const listeners = { epoch: new Set(), state: new Set() };

  function setState(s) {
    if (s === state) return;
    state = s;
    for (const fn of [...listeners.state]) fn(s); // copy: listeners may re-subscribe while handling
  }

  async function run(myGen, sess) {
    try {
      while (history.length < epochs) {
        // Sessions check this between batches, so a reset stops the old loop within one batch.
        const res = await sess.epoch({ cancelled: () => myGen !== gen });
        if (myGen !== gen) return; // reset happened during the epoch
        history.push({ epoch: history.length + 1, ...res });
        const bad = !Number.isFinite(res.trainLoss) || !Number.isFinite(res.valLoss);
        if (bad) setState('diverged');
        for (const fn of [...listeners.epoch]) fn(history[history.length - 1]);
        if (myGen !== gen) return; // reset from an epoch listener
        if (bad) return;
        if (pauseRequested) { setState('paused'); return; }
        await nextFrame();
        if (myGen !== gen) return;
        if (pauseRequested) { setState('paused'); return; }
      }
      setState('done');
    } catch (e) {
      if (myGen !== gen) return;
      error = e;
      console.error(e);
      setState('error');
    } finally {
      // A cancelled loop owns the disposal of the session it was using,
      // and must not clear the handle of a newer loop started after reset().
      if (myGen !== gen) sess.dispose();
      else loop = null;
    }
  }

  return {
    start() {
      if (loop) return loop;
      if (state === 'done' || state === 'diverged' || state === 'error') return Promise.resolve();
      if (!session) session = createSession();
      pauseRequested = false;
      setState('running');
      loop = run(gen, session);
      return loop;
    },
    pause() {
      if (state === 'running') pauseRequested = true;
    },
    reset() {
      gen++;
      pauseRequested = false;
      if (session && !loop) session.dispose();
      loop = null;
      session = null;
      history = [];
      error = null;
      setState('idle');
    },
    on(evt, fn) {
      listeners[evt].add(fn);
      return () => listeners[evt].delete(fn);
    },
    get history() { return history; },
    get state() { return state; },
    get session() { return session; },
    get error() { return error; },
    get epochs() { return epochs; },
  };
}
