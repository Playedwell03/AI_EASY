import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dataNeedsRebuild } from '../js/core/hp.js';

test('data is rebuilt when any data key differs from what the data was built from', () => {
  const keys = ['seed', 'n'];
  assert.equal(dataNeedsRebuild(keys, { seed: 1, n: 200 }, { seed: 1, n: 200, lr: 1 }), false);
  // preset (n=200) → defaults (n=2000): must rebuild even though the patch itself is empty
  assert.equal(dataNeedsRebuild(keys, { seed: 1, n: 200 }, { seed: 1, n: 2000, lr: 0.003 }), true);
});

test('modules without dataKeys always rebuild', () => {
  assert.equal(dataNeedsRebuild(undefined, { a: 1 }, { a: 1 }), true);
});
