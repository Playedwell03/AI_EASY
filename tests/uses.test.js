import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MODULES } from '../js/registry.js';
import { TABS } from '../js/router.js';
import { COMMON } from '../js/i18n.js';

const both = (pair, where) => assert.ok(pair && pair.ko && pair.en, `${where} needs ko and en`);

test('the Use tab exists after Test', () => {
  assert.deepEqual(TABS, ['arch', 'data', 'train', 'diagnose', 'test', 'use']);
  for (const k of ['tab.use', 'tabdesc.use', 'use.modelStep', 'use.needTrain', 'use.realWorld', 'use.pick']) {
    assert.ok(COMMON.ko[k] && COMMON.en[k], k);
  }
});

for (const m of MODULES) {
  test(`${m.id}: at least 3 use cases, each with exactly one model step and full text`, () => {
    assert.ok(Array.isArray(m.useCases) && m.useCases.length >= 3, `${m.id} has ${m.useCases?.length ?? 0} use cases`);
    const ids = new Set();
    for (const u of m.useCases) {
      assert.ok(!ids.has(u.id), `duplicate id ${u.id}`);
      ids.add(u.id);
      both(u.title, `${m.id}/${u.id}.title`);
      both(u.intro, `${m.id}/${u.id}.intro`);
      both(u.realWorld, `${m.id}/${u.id}.realWorld`);
      assert.equal(typeof u.render, 'function');
      assert.equal(u.pipeline.filter((s) => s.kind === 'model').length, 1, `${m.id}/${u.id} model steps`);
      for (const s of u.pipeline) {
        assert.ok(['input', 'pre', 'model', 'post', 'output'].includes(s.kind), s.kind);
        both(s.label, `${m.id}/${u.id} step label`);
        both(s.desc, `${m.id}/${u.id} step desc`);
      }
    }
  });
}
