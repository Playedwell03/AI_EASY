// Module 4 — Transformer (GPT-style) character language model with visible attention.
import { h } from '../../ui/dom.js';
import { attentionExplorer, TF_TEXT } from '../../ui/attentionView.js';
import { useCases, USE_TEXT } from './uses.js';
import { renderTextData, genLogViz, lmDiagnose, lmTest, LM_TEXT, LM_CONTROLS, PROMPTS } from '../rnn/index.js';
import { makeData, createSession, architecture } from './model.js';

export default {
  id: 'transformer',
  title: { ko: '언어 모델 (Transformer)', en: 'Language model (Transformer)' },
  summary: {
    ko: 'GPT와 같은 구조의 아주 작은 모델입니다. 모든 글자를 동시에 보고, 어텐션으로 "어디를 참고할지" 고르는 모습을 지도로 확인합니다.',
    en: 'A tiny model with the same structure as GPT. It sees every character at once and uses attention to choose what to draw on — which you can see as a map.',
  },
  archHint: 'Embedding + Position → [Masked Attention → FFN] × N → Softmax',
  synthetic: true,
  defaults: { seed: 1, corpus: 'ko', chars: 5000, corpusFraction: 1, seqLen: 32, dModel: 64, heads: 4, blocks: 2, lr: 0.003, epochs: 12, batchSize: 32 },
  cpuDefaults: { dModel: 32, heads: 2, seqLen: 16, blocks: 1, epochs: 12, chars: 2500 },
  dataKeys: ['seed', 'corpus', 'chars', 'corpusFraction', 'seqLen'],
  controls: [
    { key: 'blocks', type: 'select', label: 'c.blocks', where: 'arch', options: [1, 2, 3].map((v) => ({ value: v, label: String(v) })) },
    { key: 'heads', type: 'select', label: 'c.heads', where: 'arch', options: [1, 2, 4].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dModel', type: 'select', label: 'c.dModel', where: 'arch', options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    { key: 'seqLen', type: 'select', label: 'c.seqLen', where: 'arch', options: [16, 24, 32].map((v) => ({ value: v, label: String(v) })) },
    ...LM_CONTROLS,
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 120, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { corpusFraction: 0.1, epochs: 80 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 3 }, expect: 'diverged' },
  ],
  metrics: [{ key: 'perplexity', label: 'metric.perplexity', format: 'num' }],
  testMetrics: [
    { key: 'perplexity', label: 'metric.perplexity', format: 'num' },
    { key: 'acc', label: 'm.nextAcc', format: 'pct' },
  ],
  baselineLoss: (hp, data) => Math.log(data.vocab.itos.length),
  architecture,
  makeData,
  createSession,
  renderData: renderTextData,

  renderTrainViz(el, ctx) {
    const genEl = h('div');
    const attnEl = h('div');
    el.append(h('div', { class: 'grid-2' }, genEl, h('div', null, h('h3', { class: 'small' }, ctx.t('tf.attnTitle')), attnEl)));
    const gen = genLogViz(genEl, ctx);
    const sample = ctx.data.sentences.val[0].slice(0, ctx.hp.seqLen);
    const explorer = attentionExplorer(attnEl, ctx, (text) => ({ maps: ctx.session ? ctx.session.attentionMaps(text) : [], chars: [...text].slice(-ctx.hp.seqLen) }), { help: ctx.t('tf.attnTrain') });
    return { update() { gen.update(); if (ctx.session) explorer.update(sample); } };
  },

  diagnosePanel: lmDiagnose,
  useCases,

  renderTest(el, ctx) {
    lmTest(el, ctx);
    const card = h('div', { class: 'card' }, h('h3', null, ctx.t('tf.attnTitle')));
    el.append(card);
    const input = h('input', { class: 'text-input', value: `${PROMPTS[ctx.hp.corpus]}${ctx.hp.corpus === 'ko' ? '공원에서 공을' : 'throws a ball'}` });
    card.append(h('div', { class: 'control', style: { marginBottom: '10px' } }, input));
    const explorer = attentionExplorer(card, ctx, (text) => ({ maps: ctx.session.attentionMaps(text), chars: [...text].slice(-ctx.hp.seqLen) }), { help: ctx.t('tf.attnTest') });
    input.addEventListener('input', () => explorer.update(input.value));
    explorer.update(input.value);
  },

  text: {
    ko: {
      ...LM_TEXT.ko,
      ...TF_TEXT.ko,
      ...USE_TEXT.ko,
      'c.blocks': '블록 수',
      'c.heads': '어텐션 헤드 수',
      'c.dModel': '벡터 크기 (d_model)',
      'c.seqLen': '한 번에 보는 글자 수',
      'p.overfit': '문장 10%만으로 80에폭 → 과적합',
      'p.diverge': '학습률 3 → 발산',
      'm.nextAcc': '다음 글자 정확도',
    },
    en: {
      ...LM_TEXT.en,
      ...TF_TEXT.en,
      ...USE_TEXT.en,
      'c.blocks': 'Blocks',
      'c.heads': 'Attention heads',
      'c.dModel': 'Vector size (d_model)',
      'c.seqLen': 'Characters seen at once',
      'p.overfit': '10% of sentences for 80 epochs → overfitting',
      'p.diverge': 'Learning rate 3 → diverges',
      'm.nextAcc': 'Next-character accuracy',
    },
  },
};
