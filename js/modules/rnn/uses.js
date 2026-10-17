// "Use" tab apps for character language models (shared by the RNN and Transformer modules).
import { h, pct } from '../../ui/dom.js';
import { mulberry32 } from '../../core/rng.js';
import { PROMPTS } from '../../shared/corpus.js';
import { step, lt } from '../../ui/useView.js';

export function topK(probs, k) {
  return probs.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p).slice(0, k);
}

// Probability the model gave each character, given the text before it (null for the first).
export function charProbs(text, nextProbs, stoi) {
  const chars = [...text];
  return chars.map((c, i) => {
    if (i === 0) return null;
    const k = stoi[c];
    return k == null ? 0 : nextProbs(chars.slice(0, i).join(''))[k];
  });
}

export function textPerplexity(text, nextProbs, stoi) {
  const ps = charProbs(text, nextProbs, stoi).slice(1);
  if (!ps.length) return Infinity;
  const nll = ps.reduce((a, p) => a - Math.log(Math.max(p, 1e-9)), 0) / ps.length;
  return Math.exp(nll);
}

const show = (c) => (c === ' ' ? '␣' : c);

const keyboard = {
  id: 'keyboard',
  icon: '⌨️',
  title: lt('키보드 자동완성', 'Keyboard suggestions'),
  intro: lt('글자를 칠 때마다 다음에 올 글자 세 개를 추천합니다. 추천을 눌러 빠르게 입력해 보세요.', 'Suggests three next characters as you type. Tap a suggestion to type faster.'),
  realWorld: lt('스마트폰 키보드는 단어 단위로, 사용자의 입력 습관까지 배워 추천합니다. 휴대폰 안에서 돌도록 작고 빠른 모델을 씁니다.', 'Phone keyboards suggest whole words and adapt to your habits, using small, fast models that run on the phone.'),
  pipeline: [
    step('input', '타이핑', 'Typing', '사용자가 글자를 입력합니다.', 'The user types.'),
    step('pre', '최근 글자 자르기', 'Recent characters', '입력한 글 끝부분을 모델이 보는 길이만큼 잘라 번호로 바꿉니다.', 'Takes the end of the text (the model\'s context length) and turns it into IDs.'),
    step('model', '다음 글자 확률', 'Next-character odds', '학습한 언어 모델이 모든 글자에 대해 다음에 올 확률을 냅니다.', 'Your language model gives every character a probability of coming next.'),
    step('post', '상위 3개 고르기', 'Pick top 3', '가장 확률이 높은 세 글자를 고릅니다.', 'Keeps the three most likely characters.'),
    step('output', '추천 버튼', 'Suggestion chips', '키보드 위에 추천 버튼으로 보여 줍니다.', 'Shows them as chips above the keyboard.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const { itos } = ctx.data.vocab;
    const input = h('textarea', { class: 'text-input', rows: 3, style: { fontFamily: 'var(--mono)' } }, PROMPTS[ctx.hp.corpus]);
    const chips = h('div', { class: 'row', style: { gap: '8px' } });
    const saved = h('div', { class: 'small muted' });
    let taps = 0;
    let typed = 0;
    const refresh = () => {
      flow.run();
      chips.replaceChildren(...topK(s.nextProbs(input.value), 3).map(({ i, p }) => h('button', {
        class: 'chip', type: 'button',
        onclick: () => { input.value += itos[i]; taps++; refresh(); input.focus(); },
      }, h('b', { class: 'mono' }, show(itos[i])), h('span', { class: 'small muted' }, pct(p)))));
      saved.textContent = ctx.t('use.lm.saved', { taps, typed });
    };
    input.addEventListener('input', () => { typed++; refresh(); });
    el.append(h('p', { class: 'help' }, ctx.t('use.lm.keyboardHelp')), input, chips, saved);
    refresh();
  },
};

const story = {
  id: 'story',
  icon: '📖',
  title: lt('이야기 생성기', 'Story generator'),
  intro: lt('첫 문장만 주면 모델이 이야기를 이어서 씁니다.', 'Give the first words and the model keeps the story going.'),
  realWorld: lt('글쓰기 도우미 서비스는 글자가 아니라 단어 조각(토큰) 단위로, 수십억 개 문장을 배운 큰 모델을 씁니다. 원리는 같습니다: 다음 것을 예측하고 이어 붙이기.', 'Writing assistants use huge models trained on billions of sentences, working on word pieces (tokens) instead of characters. Same principle: predict the next piece and append it.'),
  pipeline: [
    step('input', '첫 문장', 'Opening', '이야기의 시작을 입력합니다.', 'Type how the story begins.'),
    step('pre', '번호로 바꾸기', 'To IDs', '글자를 번호로 바꿉니다.', 'Characters become IDs.'),
    step('model', '다음 글자 확률', 'Next-character odds', '학습한 모델이 다음 글자 확률을 냅니다.', 'Your model gives next-character probabilities.'),
    step('post', '하나 뽑아 이어 붙이기', 'Sample & append', '온도에 따라 한 글자를 뽑아 붙이고, 이 과정을 100번 반복합니다.', 'Samples one character (by temperature), appends it, and repeats 100 times.'),
    step('output', '완성된 이야기', 'Finished story', '생성된 글을 보여 줍니다.', 'Shows the generated text.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const input = h('input', { class: 'text-input', value: PROMPTS[ctx.hp.corpus] });
    const out = h('div', { class: 'gen-text', style: { minHeight: '90px' } });
    let temp = 0.6;
    let n = 0;
    const tv = h('span', { class: 'val' }, '0.6');
    const write = () => {
      flow.run();
      const text = s.generate(input.value, 100, temp, mulberry32(ctx.hp.seed * 31 + n++));
      out.replaceChildren(h('b', null, input.value), text);
    };
    el.append(
      h('div', { class: 'grid-2' },
        h('div', { class: 'control' }, h('label', null, ctx.t('lm.prompt')), input),
        h('div', { class: 'control' }, h('label', null, h('span', null, ctx.t('lm.temperature')), tv),
          h('input', { type: 'range', min: 0, max: 1.5, step: 0.1, value: temp, oninput: (e) => { temp = Number(e.target.value); tv.textContent = e.target.value; } }))),
      h('div', null, h('button', { class: 'btn btn-primary', type: 'button', onclick: write }, ctx.t('use.lm.write'))),
      out);
    write();
  },
};

const typo = {
  id: 'typo',
  icon: '🖍️',
  title: lt('오타 의심 표시', 'Typo highlighter'),
  intro: lt('문장의 글자마다 "여기에 이 글자가 올 만한가"를 따져, 어색한 글자를 칠해 줍니다.', 'Checks how likely each character is in its place and highlights the odd ones.'),
  realWorld: lt('맞춤법 검사기는 사전·문법 규칙과 언어 모델을 함께 씁니다. 언어 모델만으로는 "드문 단어"와 "오타"를 헷갈릴 수 있습니다.', 'Spell checkers combine dictionaries and grammar rules with language models — a model alone can confuse rare words with typos.'),
  pipeline: [
    step('input', '문장 입력', 'Text', '검사할 문장을 입력합니다.', 'Type the text to check.'),
    step('pre', '한 글자씩 앞부분 자르기', 'Prefix per character', '각 글자에 대해 "그 앞까지의 글"을 만듭니다.', 'For each character, takes the text before it.'),
    step('model', '그 글자의 확률', 'Probability of that character', '학습한 모델이 앞부분을 보고 실제 그 글자가 올 확률을 매깁니다.', 'Your model scores how likely that exact character was after the text before it.'),
    step('post', '낮은 확률 찾기', 'Find low odds', '확률이 기준보다 낮은 글자를 오타 의심으로 봅니다.', 'Characters below the threshold are suspected typos.'),
    step('output', '칠해서 보여 주기', 'Highlight', '의심 글자를 빨갛게 칠합니다.', 'Suspicious characters are shown in red.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const { stoi } = ctx.data.vocab;
    const sample = ctx.hp.corpus === 'ko' ? '오늘 고양이는 공원에서 우유를 마신다. 아이는 학교에서 책울 읽는다.' : 'the cat drinks some milk at home. a child reads a bokk at school.';
    const input = h('input', { class: 'text-input', value: sample });
    let th = 0.05;
    const out = h('div', { class: 'gen-text', style: { lineHeight: 2 } });
    const tv = h('span', { class: 'val' }, pct(th));
    const check = () => {
      flow.run();
      const ps = charProbs(input.value, s.nextProbs, stoi);
      const chars = [...input.value];
      out.replaceChildren(...chars.map((c, i) => {
        const p = ps[i];
        const bad = p != null && p < th;
        return h('span', { title: p == null ? '' : pct(p), style: bad ? { background: 'var(--bad-soft)', color: 'var(--bad)', borderRadius: '3px', fontWeight: 700 } : null }, c);
      }));
      const n = ps.filter((p) => p != null && p < th).length;
      summary.textContent = ctx.t('use.lm.typoSummary', { n });
    };
    const summary = h('div', { class: 'small muted' });
    input.addEventListener('input', check);
    el.append(
      h('p', { class: 'help' }, ctx.t('use.lm.typoHelp')),
      input,
      h('div', { class: 'control', style: { maxWidth: '320px' } }, h('label', null, h('span', null, ctx.t('use.lm.threshold')), tv),
        h('input', { type: 'range', min: 0.005, max: 0.3, step: 0.005, value: th, oninput: (e) => { th = Number(e.target.value); tv.textContent = pct(th); check(); } })),
      out, summary);
    check();
  },
};

export const useCases = [keyboard, story, typo];

export const USE_TEXT = {
  ko: {
    'use.lm.keyboardHelp': '아래 칸에 이어서 타이핑하거나, 추천 버튼을 눌러 보세요. 예: "우유를 " 다음에는 무엇을 추천할까요?',
    'use.lm.saved': '추천으로 입력한 글자 {taps}개 · 직접 친 글자 {typed}개',
    'use.lm.write': '이어 쓰기',
    'use.lm.typoHelp': '예시 문장에는 오타가 하나 숨어 있습니다. 빨갛게 칠해진 글자를 찾아보세요. 글자에 마우스를 올리면 확률이 보입니다.',
    'use.lm.threshold': '의심 기준 (이 확률보다 낮으면)',
    'use.lm.typoSummary': '의심 글자 {n}개',
  },
  en: {
    'use.lm.keyboardHelp': 'Keep typing in the box or tap a suggestion. For example, what does it suggest after "drinks "?',
    'use.lm.saved': '{taps} characters from suggestions · {typed} typed by hand',
    'use.lm.write': 'Continue the story',
    'use.lm.typoHelp': 'The sample hides a typo. Look for characters in red. Hover a character to see its probability.',
    'use.lm.threshold': 'Suspicion threshold (below this probability)',
    'use.lm.typoSummary': '{n} suspicious character(s)',
  },
};
