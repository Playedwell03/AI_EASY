import { test } from 'node:test';
import assert from 'node:assert/strict';
import { t, setLang, getLang, onLangChange, COMMON } from '../js/i18n.js';
import { MODULES } from '../js/registry.js';

test('defaults to Korean and switches to English', () => {
  setLang('ko');
  assert.equal(getLang(), 'ko');
  assert.equal(t('tab.train'), '학습');
  setLang('en');
  assert.equal(t('tab.train'), 'Train');
  setLang('ko');
});

test('substitutes variables', () => {
  const s = t('f.valRoseAfter', { epoch: 3, pct: 12 });
  assert.ok(s.includes('3') && s.includes('12'));
});

test('missing key returns the key itself', () => {
  assert.equal(t('no.such.key'), 'no.such.key');
});

test('module dictionaries are used when passed', () => {
  const dict = { ko: { hi: '안녕 {name}' }, en: { hi: 'Hi {name}' } };
  setLang('en');
  assert.equal(t('hi', { name: 'A' }, dict), 'Hi A');
  setLang('ko');
});

test('onLangChange notifies and unsubscribes', () => {
  let calls = 0;
  const off = onLangChange(() => calls++);
  setLang('en');
  off();
  setLang('ko');
  assert.equal(calls, 1);
});

function sameKeys(name, dict) {
  const ko = Object.keys(dict.ko).sort();
  const en = Object.keys(dict.en).sort();
  assert.deepEqual(en, ko, `${name}: ko/en keys differ`);
}

test('every dictionary has identical ko/en keys', () => {
  sameKeys('COMMON', COMMON);
  for (const m of MODULES) {
    sameKeys(m.id, m.text);
    for (const k of ['title', 'summary']) assert.ok(m[k].ko && m[k].en, `${m.id}.${k}`);
  }
});

test('every diagnosis key the engine can emit has text', () => {
  const keys = [
    'f.lossNaN', 'f.lossExploded', 'f.worseThanBaseline', 'f.tooFewEpochs', 'f.valRoseAfter', 'f.trainKeptFalling',
    'f.nearBaseline', 'f.beatsBaseline', 'f.gapSmall', 'f.gapLarge', 'f.stillImproving',
    'adv.lowerLr', 'adv.trainMore', 'adv.earlyStop', 'adv.regularize', 'adv.moreData',
    'adv.biggerModel', 'adv.raiseLr',
    'verdict.good', 'verdict.insufficient', 'verdict.diverged', 'verdict.underfitting', 'verdict.overfitting',
  ];
  for (const k of keys) assert.ok(COMMON.ko[k] && COMMON.en[k], k);
});
