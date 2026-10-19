// Module 5 — BERT-style masked language model (fill in the blanks, both directions).
import { h, pct, probBars } from '../../ui/dom.js';
import { mulberry32 } from '../../core/rng.js';
import { attentionExplorer, TF_TEXT } from '../../ui/attentionView.js';
import { LM_CONTROLS } from '../rnn/index.js';
import { makeData, createSession, architecture, maskTokens } from './model.js';
import { useCases, USE_TEXT } from './uses.js';

const show = (c) => (c === ' ' ? '␣' : c);

// One sentence drawn as character boxes; masked slots highlighted.
function maskedRow(itos, input, labels) {
  return h('div', { class: 'row', style: { gap: '3px' } }, input.map((id, i) => {
    if (itos[id] === '[PAD]') return null;
    const target = labels[i] !== -1;
    const tok = itos[id];
    return h('span', {
      class: 'mono',
      title: target ? `→ ${itos[labels[i]]}` : '',
      style: { padding: '3px 5px', borderRadius: '5px', fontSize: '13px', background: target ? 'var(--accent-soft)' : 'var(--surface-2)', border: target ? '1px solid var(--accent)' : '1px solid transparent' },
    }, tok === '[MASK]' ? '▢' : show(tok));
  }));
}

export default {
  id: 'bert',
  title: { ko: '빈칸 맞히기 모델 (BERT)', en: 'Fill-in-the-blank model (BERT)' },
  summary: {
    ko: '문장 중간을 가리고 앞뒤 문맥을 모두 써서 맞히는 양방향 모델입니다. 검색·분류에 널리 쓰이는 BERT의 축소판입니다.',
    en: 'A bidirectional model that hides parts of a sentence and fills them in using context on both sides — a small version of BERT, widely used for search and classification.',
  },
  archHint: 'Embedding → [Bidirectional Attention → FFN] × N → MLM head',
  synthetic: true,
  defaults: { seed: 1, corpus: 'ko', chars: 9000, corpusFraction: 1, maxLen: 32, maskProb: 0.15, dModel: 64, heads: 4, blocks: 2, lr: 0.003, epochs: 18, batchSize: 32 },
  cpuDefaults: { chars: 3500, dModel: 32, heads: 2, blocks: 1, epochs: 16, maskProb: 0.25 },
  dataKeys: ['seed', 'corpus', 'chars', 'corpusFraction', 'maxLen', 'maskProb'],
  controls: [
    { key: 'blocks', type: 'select', label: 'c.blocks', where: 'arch', options: [1, 2, 3].map((v) => ({ value: v, label: String(v) })) },
    { key: 'heads', type: 'select', label: 'c.heads', where: 'arch', options: [1, 2, 4].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dModel', type: 'select', label: 'c.dModel', where: 'arch', options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    ...LM_CONTROLS,
    { key: 'maskProb', type: 'select', label: 'c.maskProb', where: 'data', options: [0.1, 0.15, 0.25, 0.4].map((v) => ({ value: v, label: `${v * 100}%` })) },
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 150, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { corpusFraction: 0.05, epochs: 120 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 3 }, expect: 'diverged' },
  ],
  metrics: [
    { key: 'acc', label: 'm.top1', format: 'pct' },
    { key: 'top5', label: 'm.top5', format: 'pct' },
  ],
  testMetrics: [
    { key: 'acc', label: 'm.top1', format: 'pct', sub: 'm.chance' },
    { key: 'top5', label: 'm.top5', format: 'pct' },
  ],
  baselineLoss: (hp, data) => Math.log(data.vocab.itos.length - 2),
  architecture,
  makeData,
  createSession,
  useCases,

  renderData(el, data, ctx) {
    const { itos } = data.vocab;
    const rand = mulberry32(ctx.hp.seed + 3);
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      h('div', { class: 'stack', style: { gap: '6px' } }, data.train.ids.slice(0, 5).map((ids) => {
        const m = maskTokens(ids, rand, data.maskOpts);
        return maskedRow(itos, m.input, m.labels);
      })),
      h('p', { class: 'help', style: { marginTop: '10px' } }, ctx.t('m.dataRule')),
      h('p', { class: 'small muted' }, ctx.t('m.vocab', { n: itos.length - 2 })));
  },

  renderTrainViz(el, ctx) {
    const sentence = ctx.data.sentences.val[0];
    const pos = [...sentence].map((c, i) => (c !== ' ' && c !== '.' ? i : -1)).filter((i) => i >= 0);
    const target = pos[Math.floor(pos.length * 0.6)];
    const demo = h('div');
    const attnEl = h('div');
    el.append(h('div', { class: 'grid-2' },
      h('div', null, h('h3', { class: 'small' }, ctx.t('m.demoTitle')), h('p', { class: 'help' }, ctx.t('m.demoHelp')), demo),
      h('div', null, h('h3', { class: 'small' }, ctx.t('tf.attnTitle')), attnEl)));
    const explorer = attentionExplorer(attnEl, ctx, (text) => ({ maps: ctx.session ? ctx.session.attentionMaps(text) : [], chars: [...text] }), { help: ctx.t('m.attnHelp') });
    return {
      update() {
        const s = ctx.session;
        if (!s) { demo.replaceChildren(); return; }
        const [f] = s.fillMask(sentence, [target]);
        const chars = [...sentence];
        demo.replaceChildren(
          h('div', { class: 'gen-text' }, chars.map((c, i) => (i === target ? h('b', { class: 'tag-warn' }, '▢') : c))),
          h('div', { style: { marginTop: '8px' } }, probBars(f.top.map((x) => show(x.c)), f.top.map((x) => x.p), { highlight: f.top.findIndex((x) => x.c === chars[target]) })),
          h('p', { class: 'small muted' }, ctx.t('m.answer', { c: chars[target] })));
        explorer.update(sentence);
      },
    };
  },

  async diagnosePanel(el, ctx, history) {
    const last = history.at(-1).metrics;
    const V = ctx.data.vocab.itos.length - 2;
    const findings = [{ level: last.acc >= 0.35 ? 'good' : 'warn', key: 'm.top1Finding', vars: { a: pct(last.acc), c: pct(1 / V) } }];
    findings.push({ level: 'info', key: 'm.top5Finding', vars: { a: pct(last.top5) } });
    el.append(h('div', { class: 'card' }, h('h3', null, ctx.t('m.vsGpt')), h('p', { class: 'help', style: { margin: 0 } }, ctx.t('m.vsGptBody'))));
    return findings;
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const input = h('input', { class: 'text-input', value: ctx.data.sentences.test[0] });
    const row = h('div', { class: 'row', style: { gap: '3px' } });
    const out = h('div', { class: 'stack' });
    const masked = new Set();
    const draw = () => {
      const chars = [...input.value];
      row.replaceChildren(...chars.map((c, i) => h('button', {
        type: 'button', class: 'mono',
        style: { padding: '4px 6px', border: '1px solid var(--border)', borderRadius: '5px', cursor: 'pointer', font: 'inherit', fontSize: '14px', background: masked.has(i) ? 'var(--accent)' : 'var(--surface)', color: masked.has(i) ? '#fff' : 'var(--text)' },
        onclick: () => { if (masked.has(i)) masked.delete(i); else masked.add(i); draw(); },
      }, masked.has(i) ? '▢' : show(c))));
      const pos = [...masked].filter((i) => i < chars.length).sort((a, b) => a - b);
      out.replaceChildren(...(pos.length ? s.fillMask(input.value, pos).map((f) => h('div', null,
        h('b', { class: 'small' }, ctx.t('m.slotAnswer', { n: f.pos + 1, c: show(f.original) })),
        probBars(f.top.map((x) => show(x.c)), f.top.map((x) => x.p), { highlight: f.top.findIndex((x) => x.c === f.original) }))) : [h('p', { class: 'muted small' }, ctx.t('m.clickToMask'))]));
    };
    input.addEventListener('input', () => { masked.clear(); draw(); });
    el.append(h('div', { class: 'card' }, h('h3', null, ctx.t('test.try')), h('p', { class: 'help' }, ctx.t('m.tryHelp')), input, h('div', { style: { margin: '10px 0' } }, row), out));
    const first = [...input.value].findIndex((c, i) => i > 4 && c !== ' ');
    masked.add(first);
    draw();
  },

  text: {
    ko: {
      ...TF_TEXT.ko,
      ...USE_TEXT.ko,
      'c.corpus': '연습 문장 언어',
      'c.fraction': '학습에 쓸 문장 비율',
      'c.blocks': '블록 수',
      'c.heads': '어텐션 헤드 수',
      'c.dModel': '벡터 크기 (d_model)',
      'c.maskProb': '가리는 글자 비율',
      'p.overfit': '문장 5%만으로 120에폭 → 과적합',
      'p.diverge': '학습률 3 → 발산',
      'm.top1': '빈칸 정답률 (1순위)',
      'm.top5': '빈칸 정답률 (5순위 안)',
      'm.chance': '무작위로 찍으면 약 2%',
      'm.dataIntro': '파란 테두리 칸이 이번에 맞혀야 하는 글자입니다. ▢는 [MASK]로 가려진 글자입니다. 마우스를 올리면 정답이 보입니다. 같은 문장도 에폭마다 다른 곳이 가려집니다.',
      'm.dataRule': '왜 가린 글자 중 일부는 ▢가 아니라 다른 글자나 원래 글자 그대로일까요? 실제 사용할 때는 [MASK]가 없기 때문에, "보이는 글자도 의심하라"고 가르치는 장치입니다(80%는 [MASK], 10%는 아무 글자, 10%는 그대로).',
      'm.vocab': '어휘: 글자 {n}개 + 특수 토큰 [PAD], [MASK]',
      'm.demoTitle': '빈칸 맞히기 연습 (검증 문장)',
      'm.demoHelp': '매 에폭마다 같은 빈칸을 맞혀 봅니다. 초록 막대가 정답입니다.',
      'm.answer': '정답: {c}',
      'm.attnHelp': '대각선 위쪽도 채워져 있습니다. GPT식 모델과 달리 뒤쪽 글자도 참고하기 때문입니다.',
      'm.top1Finding': '가려진 글자의 1순위 정답률이 {a}입니다(무작위로 찍으면 {c}).',
      'm.top5Finding': '정답이 상위 5개 후보 안에 드는 비율은 {a}입니다.',
      'm.vsGpt': 'GPT식 모델과 무엇이 다른가요?',
      'm.vsGptBody': 'GPT식(4번 모듈)은 왼쪽 글자만 보고 다음 글자를 맞히므로 글을 "생성"하는 데 강합니다. BERT식은 양쪽을 다 보고 빈칸을 맞히므로 문장을 "이해"해서 분류·검색·교정하는 데 강합니다. 대신 BERT식은 글을 처음부터 이어 쓰는 데는 맞지 않습니다.',
      'm.tryHelp': '문장의 글자를 눌러 가려 보세요(여러 개 가능). 모델이 앞뒤 문맥으로 원래 글자를 맞힙니다.',
      'm.clickToMask': '위에서 가릴 글자를 눌러 주세요.',
      'm.slotAnswer': '{n}번째 글자 (정답: {c})',
    },
    en: {
      ...TF_TEXT.en,
      ...USE_TEXT.en,
      'c.corpus': 'Practice sentence language',
      'c.fraction': 'Share of sentences used for training',
      'c.blocks': 'Blocks',
      'c.heads': 'Attention heads',
      'c.dModel': 'Vector size (d_model)',
      'c.maskProb': 'Share of characters hidden',
      'p.overfit': '5% of sentences for 120 epochs → overfitting',
      'p.diverge': 'Learning rate 3 → diverges',
      'm.top1': 'Blank accuracy (top 1)',
      'm.top5': 'Blank accuracy (in top 5)',
      'm.chance': 'Random guessing: about 2%',
      'm.dataIntro': 'Boxes with a blue border are the characters to guess this time; ▢ is a character hidden by [MASK]. Hover to see the answer. Each epoch hides different spots in the same sentence.',
      'm.dataRule': 'Why are some targets shown as another character, or unchanged, instead of ▢? Real inputs never contain [MASK], so this teaches the model to doubt visible characters too (80% [MASK], 10% random, 10% unchanged).',
      'm.vocab': 'Vocabulary: {n} characters + special tokens [PAD], [MASK]',
      'm.demoTitle': 'Fill-in practice (validation sentence)',
      'm.demoHelp': 'The same blank is tried after every epoch. The green bar is the right answer.',
      'm.answer': 'Answer: {c}',
      'm.attnHelp': 'Cells above the diagonal are filled too: unlike a GPT-style model, it also looks at later characters.',
      'm.top1Finding': 'Top-1 accuracy on hidden characters is {a} (random guessing: {c}).',
      'm.top5Finding': 'The right answer is among the top 5 candidates {a} of the time.',
      'm.vsGpt': 'How is this different from the GPT-style model?',
      'm.vsGptBody': 'The GPT-style model (module 4) only sees characters to the left and predicts the next one, so it is good at generating text. The BERT style sees both sides and fills blanks, so it is good at understanding text for classification, search and correction — but not at writing from scratch.',
      'm.tryHelp': 'Click characters to hide them (several is fine). The model uses context on both sides to guess them.',
      'm.clickToMask': 'Click a character above to hide it.',
      'm.slotAnswer': 'Character {n} (answer: {c})',
    },
  },
};
