// Module 3 — Character-level language model with an RNN (LSTM / GRU).
import { h, fmt, probBars } from '../../ui/dom.js';
import { mulberry32 } from '../../core/rng.js';
import { makeData, createSession, architecture } from './model.js';
import { PROMPTS } from '../../shared/corpus.js';
import { useCases, USE_TEXT } from './uses.js';

export { PROMPTS };
const show = (c) => (c === ' ' ? '␣' : c);

// Pieces shared with the Transformer and BERT modules.
export function renderTextData(el, data, ctx) {
  const { itos } = data.vocab;
  const ex = data.train.x[5];
  el.append(
    h('p', { class: 'help' }, ctx.t('lm.dataIntro')),
    h('div', { class: 'gen-text' }, data.sentences.train.slice(0, 6).join('\n')),
    h('h3', { class: 'small', style: { marginTop: '16px' } }, ctx.t('lm.vocab', { n: itos.length })),
    h('div', { class: 'row', style: { gap: '4px' } }, itos.map((c) => h('span', { class: 'badge mono' }, show(c)))),
    ex && h('div', null,
      h('h3', { class: 'small', style: { marginTop: '16px' } }, ctx.t('lm.example')),
      h('div', { class: 'gen-text' }, `${ex.map((i) => itos[i]).join('')}  →  ${show(itos[data.train.y[5]])}`),
      h('p', { class: 'help', style: { marginTop: '6px' } }, ctx.t('lm.exampleHelp'))));
}

export function genLogViz(el, ctx) {
  const log = h('div', { class: 'gen-log' });
  const prompt = PROMPTS[ctx.hp.corpus];
  el.append(h('p', { class: 'help' }, ctx.t('lm.vizIntro', { prompt: prompt.trim() })), log);
  return {
    update() {
      const s = ctx.session;
      if (!s) { log.replaceChildren(); return; }
      const n = ctx.runner.history.length;
      if (n && s.genLog.at(-1)?.epoch !== n) {
        s.genLog.push({ epoch: n, text: prompt + s.generate(prompt, 50, 0.5, mulberry32(ctx.hp.seed + n)) });
      }
      log.replaceChildren(...[...s.genLog].reverse().map((g) =>
        h('div', null, h('div', { class: 'ep' }, ctx.t('train.epochOf', { n: g.epoch, total: ctx.runner.epochs })), h('div', { class: 'gen-text' }, g.text))));
    },
  };
}

export async function lmDiagnose(el, ctx, history) {
  const last = history.at(-1).metrics;
  const V = ctx.data.vocab.itos.length;
  const findings = [{ level: 'info', key: 'lm.pplMeaning', vars: { p: fmt(last.perplexity), v: V } }];
  if (last.perplexity < 0.5 * V) findings.push({ level: 'good', key: 'lm.pplGood' });
  // Check the pattern the corpus is built on: does the object predict its verb?
  const s = ctx.session;
  const probe = ctx.hp.corpus === 'ko' ? ['오늘 친구는 공원에서 우유를 ', '마'] : ['the cat drinks some ', 'm'];
  const p = s.nextProbs(probe[0]);
  const k = ctx.data.vocab.stoi[probe[1]];
  findings.push({ level: p[k] > 0.5 ? 'good' : 'warn', key: 'lm.patternProbe', vars: { prompt: probe[0], c: probe[1], p: `${(p[k] * 100).toFixed(0)}%` } });
  return findings;
}

export function lmTest(el, ctx) {
  const s = ctx.session;
  const { itos } = ctx.data.vocab;
  const input = h('input', { class: 'text-input', value: PROMPTS[ctx.hp.corpus] });
  const tempVal = h('span', { class: 'val' }, '0.5');
  const temp = h('input', { type: 'range', min: 0, max: 1.5, step: 0.1, value: 0.5 });
  const out = h('div', { class: 'gen-text' }, '…');
  const bars = h('div');
  let calls = 0;
  const refresh = () => {
    const probs = s.nextProbs(input.value);
    const top = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
    bars.replaceChildren(probBars(top.map(([, i]) => show(itos[i])), top.map(([p]) => p)));
  };
  const gen = () => {
    out.textContent = input.value + s.generate(input.value, 80, Number(temp.value), mulberry32(ctx.hp.seed * 97 + calls++));
  };
  temp.addEventListener('input', () => { tempVal.textContent = temp.value; });
  input.addEventListener('input', refresh);
  el.append(h('div', { class: 'card' },
    h('h3', null, ctx.t('test.try')),
    h('p', { class: 'help' }, ctx.t('lm.tryHelp')),
    h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        h('div', { class: 'control' }, h('label', null, ctx.t('lm.prompt')), input),
        h('div', { class: 'control' }, h('label', null, h('span', null, ctx.t('lm.temperature')), tempVal), temp),
        h('p', { class: 'help', style: { margin: 0 } }, ctx.t('lm.tempHelp')),
        h('div', null, h('button', { class: 'btn btn-primary', type: 'button', onclick: gen }, ctx.t('lm.generate'))),
        out),
      h('div', null, h('h3', { class: 'small' }, ctx.t('lm.nextChar')), bars))));
  refresh();
  gen();
}

// Text keys shared by the language modules (merged into each module's dictionary).
export const LM_TEXT = {
  ko: {
    'c.corpus': '연습 문장 언어',
    'c.fraction': '학습에 쓸 문장 비율',
    'lm.dataIntro': '규칙에 따라 만든 짧은 문장들입니다. "우유를" 다음엔 "마신다", "책을" 다음엔 "읽는다"처럼 목적어와 동사가 짝을 이룹니다. 모델은 글자 단위로, 앞 글자들을 보고 다음 글자를 맞히는 연습을 합니다.',
    'lm.vocab': '어휘 (글자 {n}개)',
    'lm.example': '학습 예제 하나',
    'lm.exampleHelp': '화살표 왼쪽의 글자들을 보고 오른쪽 글자 하나를 맞혀야 합니다. 문장 전체를 한 칸씩 밀면서 이런 예제를 수천 개 만듭니다.',
    'lm.vizIntro': '매 에폭이 끝날 때마다 "{prompt}"로 시작하는 문장을 생성해 봅니다. 처음엔 엉터리지만 점점 단어와 문장 구조가 나타납니다.',
    'lm.pplMeaning': '퍼플렉서티 {p}: 다음 글자를 고를 때 평균적으로 약 {p}개 후보 사이에서 고민한다는 뜻입니다. 아무것도 모르면 {v}개(어휘 전체) 사이에서 고민합니다.',
    'lm.pplGood': '퍼플렉서티가 어휘 크기의 절반보다 훨씬 작습니다. 글자 사이의 규칙을 배웠습니다.',
    'lm.patternProbe': '패턴 확인: "{prompt}" 다음에 "{c}"가 올 확률을 {p}로 예측했습니다. 목적어에 맞는 동사를 기억하고 있는지 보여 줍니다.',
    'lm.tryHelp': '시작 문장을 입력하고 생성 버튼을 눌러 보세요. 입력하는 동안 오른쪽에 "다음 글자" 후보가 실시간으로 바뀝니다.',
    'lm.prompt': '시작 문장',
    'lm.temperature': '온도(Temperature)',
    'lm.tempHelp': '온도가 0에 가까우면 항상 가장 확률 높은 글자를 고르고(반복적), 높을수록 낮은 확률의 글자도 뽑아 다양하지만 엉뚱해집니다.',
    'lm.generate': '생성하기',
    'lm.nextChar': '다음 글자 후보 Top 5',
    'metric.perplexity': '퍼플렉서티(Perplexity)',
  },
  en: {
    'c.corpus': 'Practice sentence language',
    'c.fraction': 'Share of sentences used for training',
    'lm.dataIntro': 'Short sentences made from rules. Objects and verbs come in pairs, like "drinks" → "some milk" and "reads" → "a book". The model works character by character, practicing guessing the next character from the ones before it.',
    'lm.vocab': 'Vocabulary ({n} characters)',
    'lm.example': 'One training example',
    'lm.exampleHelp': 'From the characters left of the arrow, it must guess the single character on the right. Sliding along the text creates thousands of these examples.',
    'lm.vizIntro': 'After every epoch it generates a sentence starting with "{prompt}". Gibberish at first, then words and sentence structure appear.',
    'lm.pplMeaning': 'Perplexity {p}: when choosing the next character it is, on average, torn between about {p} candidates. Knowing nothing, it would be torn between all {v} (the whole vocabulary).',
    'lm.pplGood': 'Perplexity is far below half the vocabulary size. It learned the rules between characters.',
    'lm.patternProbe': 'Pattern check: after "{prompt}" it gives "{c}" a {p} chance. That shows whether it remembers which words go together.',
    'lm.tryHelp': 'Type a starting text and press Generate. While you type, the next-character candidates on the right update live.',
    'lm.prompt': 'Starting text',
    'lm.temperature': 'Temperature',
    'lm.tempHelp': 'Near 0 it always picks the most likely character (repetitive). Higher values also pick unlikely characters: more varied, but odder.',
    'lm.generate': 'Generate',
    'lm.nextChar': 'Top 5 next characters',
    'metric.perplexity': 'Perplexity',
  },
};

export const LM_CONTROLS = [
  { key: 'corpus', type: 'select', label: 'c.corpus', where: 'data', options: [{ value: 'ko', label: { ko: '한국어', en: 'Korean' } }, { value: 'en', label: { ko: '영어', en: 'English' } }] },
  { key: 'corpusFraction', type: 'select', label: 'c.fraction', where: 'data', options: [0.05, 0.1, 0.25, 0.5, 1].map((v) => ({ value: v, label: `${v * 100}%` })) },
];

export default {
  id: 'rnn',
  title: { ko: '언어 모델 (RNN)', en: 'Language model (RNN)' },
  summary: {
    ko: '글자를 하나씩 읽으며 기억을 이어 가는 순환 신경망으로 다음 글자를 예측하고, 문장을 생성해 봅니다.',
    en: 'A recurrent network reads one character at a time, carrying a memory forward, to predict the next character and generate text.',
  },
  archHint: 'Embedding → LSTM | GRU → Softmax(vocab)',
  synthetic: true,
  // Recurrent steps are many tiny ops: the CPU backend beats WebGL here (measured ~8s vs ~4s per epoch).
  preferBackend: 'cpu',
  defaults: { seed: 1, corpus: 'ko', chars: 2000, corpusFraction: 1, seqLen: 12, embed: 24, cell: 'gru', units: 32, lr: 0.01, epochs: 10, batchSize: 32 },
  dataKeys: ['seed', 'corpus', 'chars', 'corpusFraction', 'seqLen'],
  controls: [
    { key: 'cell', type: 'select', label: 'c.cell', where: 'arch', options: [{ value: 'gru', label: 'GRU' }, { value: 'lstm', label: 'LSTM' }] },
    { key: 'units', type: 'select', label: 'c.units', where: 'arch', options: [16, 32, 64, 128].map((v) => ({ value: v, label: String(v) })) },
    { key: 'seqLen', type: 'select', label: 'c.seqLen', where: 'arch', options: [8, 12, 16, 24].map((v) => ({ value: v, label: String(v) })) },
    ...LM_CONTROLS,
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 80, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 16, max: 256, step: 16, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { corpusFraction: 0.1, units: 64, epochs: 40 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 3 }, expect: 'diverged' },
  ],
  metrics: [
    { key: 'perplexity', label: 'metric.perplexity', format: 'num' },
  ],
  testMetrics: [
    { key: 'perplexity', label: 'metric.perplexity', format: 'num' },
    { key: 'acc', label: 'm.nextAcc', format: 'pct' },
  ],
  baselineLoss: (hp, data) => Math.log(data.vocab.itos.length),
  architecture,
  makeData,
  createSession,
  renderData: renderTextData,
  renderTrainViz: genLogViz,
  diagnosePanel: lmDiagnose,
  renderTest: lmTest,
  useCases,

  text: {
    ko: {
      ...LM_TEXT.ko,
      ...USE_TEXT.ko,
      'c.cell': '순환 셀 종류',
      'c.units': '은닉 상태 크기',
      'c.seqLen': '한 번에 보는 글자 수',
      'p.overfit': '문장 10%만으로 40에폭 → 과적합',
      'p.diverge': '학습률 3 → 발산',
      'm.nextAcc': '다음 글자 정확도',
    },
    en: {
      ...LM_TEXT.en,
      ...USE_TEXT.en,
      'c.cell': 'Recurrent cell',
      'c.units': 'Hidden state size',
      'c.seqLen': 'Characters seen at once',
      'p.overfit': '10% of sentences for 40 epochs → overfitting',
      'p.diverge': 'Learning rate 3 → diverges',
      'm.nextAcc': 'Next-character accuracy',
    },
  },
};
