// Pure decision logic used inside the "Use" tab apps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toPlane } from '../js/modules/mlp/uses.js';

test('mlp toPlane maps a feature range onto [-1, 1] and clamps', () => {
  assert.equal(toPlane(0, 0, 20), -1);
  assert.equal(toPlane(20, 0, 20), 1);
  assert.equal(toPlane(10, 0, 20), 0);
  assert.equal(toPlane(30, 0, 20), 1);
});
