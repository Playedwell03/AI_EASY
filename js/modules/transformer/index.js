// Module 4 — Transformer (GPT-style) character language model with visible attention.
import { h } from '../../ui/dom.js';
import { heatmap } from '../../ui/chart.js';
import { renderTextData, genLogViz, lmDiagnose, lmTest, LM_TEXT, LM_CONTROLS, PROMPTS } from '../rnn/index.js';
import { makeData, createSession, architecture } from './model.js';

const show = (c) => (c === ' ' ? '␣' : c);

// Heatmap of one layer/head for some text, with selectors. Returns { update(text) }.
export function attentionExplorer(el, ctx, getMaps, { help } = {}) {
  const layerSel = h('select', { class: 'text-input', style: { width: 'auto' } });
  const headSel = h('select', { class: 'text-input', style: { width: 'auto' } });
  const canvas = h('canvas');
  let maps = [];
  let chars = [];
  const draw = () => {
    const m = maps[Number(layerSel.value) || 0]?.[Number(headSel.value) || 0];
    if (!m) return;
    heatmap(canvas, m, { rowLabels: chars, colLabels: chars, height: Math.min(420, 30 + chars.length * 16) });
  };
  layerSel.addEventListener('change', draw);
  headSel.addEventListener('change', draw);
  el.append(
    help && h('p', { class: 'help' }, help),
    h('div', { class: 'row', style: { marginBottom: '8px' } },
      h('label', { class: 'small muted' }, ctx.t('tf.layer')), layerSel,
      h('label', { class: 'small muted' }, ctx.t('tf.head')), headSel),
    h('div', { class: 'chart-wrap', style: { maxWidth: '520px' } }, canvas),
    h('p', { class: 'small muted', style: { marginTop: '6px' } }, ctx.t('tf.howToRead')));
  return {
    update(text) {
      const r = getMaps(text);
      maps = r.maps;
      chars = r.chars.map(show);
      if (!maps.length) return;
      if (layerSel.options.length !== maps.length) {
        layerSel.replaceChildren(...maps.map((_, i) => h('option', { value: i }, String(i + 1))));
        headSel.replaceChildren(...maps[0].map((_, i) => h('option', { value: i }, String(i + 1))));
      }
      draw();
    },
  };
}

export const TF_TEXT = {
  ko: {
    'tf.layer': '블록',
    'tf.head': '헤드',
    'tf.howToRead': '행: 지금 글자 · 열: 참고하는 글자. 칸이 진할수록 그 글자를 많이 참고합니다.',
    'tf.attnTrain': '아래 문장에 대한 어텐션입니다. 학습 전에는 고르게 퍼져 있다가, 학습할수록 특정 글자(바로 앞 글자, 띄어쓰기, 목적어 등)에 집중하는 무늬가 생깁니다.',
    'tf.attnTest': '입력한 문장의 마지막 부분에 대한 어텐션입니다. 대각선 위쪽이 비어 있는 것은 미래 글자를 볼 수 없게 가렸기(마스크) 때문입니다.',
    'tf.attnTitle': '어텐션 지도',
  },
  en: {
    'tf.layer': 'Block',
    'tf.head': 'Head',
    'tf.howToRead': 'Rows: current character · columns: characters it draws on. Darker cells mean more attention.',
    'tf.attnTrain': 'Attention for the text below. Before training it is spread evenly; as training goes on, patterns appear that focus on particular characters (the previous one, spaces, the object).',
    'tf.attnTest': 'Attention for the end of the text you typed. The empty area above the diagonal is the mask: it cannot see future characters.',
    'tf.attnTitle': 'Attention map',
  },
};

export default {
  id: 'transformer',
  title: { ko: '언어 모델 (Transformer)', en: 'Language model (Transformer)' },
  summary: {
    ko: 'GPT와 같은 구조의 아주 작은 모델입니다. 모든 글자를 동시에 보고, 어텐션으로 "어디를 참고할지" 고르는 모습을 지도로 확인합니다.',
    en: 'A tiny model with the same structure as GPT. It sees every character at once and uses attention to choose what to draw on — which you can see as a map.',
  },
  archHint: 'Embedding + Position → [Masked Attention → FFN] × N → Softmax',
  synthetic: true,
  defaults: { seed: 1, corpus: 'ko', chars: 5000, corpusFraction: 1, seqLen: 32, dModel: 64, heads: 4, blocks: 2, lr: 0.003, epochs: 30, batchSize: 32 },
  cpuDefaults: { dModel: 32, seqLen: 24, blocks: 1, epochs: 25 },
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
