// "Use" tab apps for the Transformer language model.
import { h, fmt, pct, probBars } from '../../ui/dom.js';
import { mulberry32, shuffle } from '../../core/rng.js';
import { PROMPTS, makeCorpus } from '../../shared/corpus.js';
import { step, lt } from '../../ui/useView.js';
import { topK, textPerplexity } from '../rnn/uses.js';

export function rankByScore(candidates, score) {
  return candidates.map((text) => ({ text, score: score(text) })).sort((a, b) => a.score - b.score);
}

const show = (c) => (c === ' ' ? '␣' : c);

// Greedy continuation until a sentence end, capped at n characters.
function complete(s, text, n) {
  let out = '';
  for (let i = 0; i < n; i++) {
    const c = s.generate(text + out, 1, 0, mulberry32(1));
    out += c;
    if (c === '.') break;
  }
  return out;
}

const search = {
  id: 'search',
  icon: '🔎',
  title: lt('검색어 자동완성', 'Search autocomplete'),
  intro: lt('검색창에 치는 동안 완성된 문장 후보를 보여 주고, 모델이 어느 글자를 근거로 삼았는지도 보여 줍니다.', 'Shows full-sentence suggestions while you type — and which characters the model leaned on.'),
  realWorld: lt('검색 서비스는 실제 사람들의 검색 기록으로 학습하고, 부적절한 추천을 거르는 필터를 덧붙입니다.', 'Search engines learn from real query logs and add filters that block inappropriate suggestions.'),
  pipeline: [
    step('input', '검색어 입력', 'Typing a query', '사용자가 검색창에 입력합니다.', 'The user types into the search box.'),
    step('pre', '글자 번호로', 'To IDs', '입력을 글자 번호로 바꾸고 위치 정보를 붙입니다.', 'Turns the input into IDs plus position information.'),
    step('model', '어텐션 + 다음 글자', 'Attention + next char', '학습한 Transformer가 앞 글자들을 어텐션으로 참고해 다음 글자 확률을 냅니다.', 'Your Transformer attends over the earlier characters and gives next-character probabilities.'),
    step('post', '후보 3개 완성', 'Complete 3 candidates', '가장 그럴듯한 다음 글자 3개에서 출발해 마침표까지 이어 씁니다.', 'Starts from the three likeliest next characters and continues each to the period.'),
    step('output', '추천 목록 + 근거', 'Suggestions + evidence', '추천 문장과 함께, 모델이 주목한 글자를 막대로 보여 줍니다.', 'Shows suggestions plus bars for the characters the model attended to.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const { itos } = ctx.data.vocab;
    const input = h('input', { class: 'text-input', value: ctx.hp.corpus === 'ko' ? '오늘 친구는 바다에서 노래를 ' : 'my friend sings ' });
    const list = h('div', { class: 'stack', style: { gap: '6px' } });
    const why = h('div');
    const refresh = () => {
      flow.run();
      const q = input.value;
      const sugg = topK(s.nextProbs(q), 3).map(({ i, p }) => ({ text: itos[i] + complete(s, q + itos[i], 30), p }));
      list.replaceChildren(...sugg.map((sg) => h('div', { class: 'mail-row' }, h('span', null, '🔎'), h('span', null, h('span', { class: 'muted' }, q), h('b', null, sg.text)), h('span', { class: 'mono small' }, pct(sg.p)))));
      // Attention of the last typed character in the last block, averaged over heads.
      const maps = s.attentionMaps(q);
      if (!maps.length) { why.replaceChildren(); return; }
      const lastBlock = maps[maps.length - 1];
      const t = lastBlock[0].length;
      const avg = Array.from({ length: t }, (_, j) => lastBlock.reduce((a, head) => a + head[t - 1][j], 0) / lastBlock.length);
      const chars = [...q].slice(-t);
      const top = avg.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 6).sort((a, b) => a.j - b.j);
      why.replaceChildren(h('b', { class: 'small' }, ctx.t('use.tf.why')), probBars(top.map(({ j }) => `${show(chars[j])} (${j - t + 1})`), top.map(({ w }) => w)));
    };
    input.addEventListener('input', refresh);
    el.append(h('div', { class: 'control' }, h('label', null, ctx.t('use.tf.query')), input), h('div', { class: 'grid-2' }, list, why));
    refresh();
  },
};

const rerank = {
  id: 'rerank',
  icon: '🎙️',
  title: lt('음성 인식 후보 고르기', 'Speech-recognition rescoring'),
  intro: lt('소리를 듣고 만든 비슷한 문장 후보 중에서 가장 자연스러운 것을 고릅니다.', 'Picks the most natural sentence among similar-sounding candidates.'),
  realWorld: lt('음성 인식기는 소리만으로는 헷갈리는 후보를 여러 개 만든 뒤, 언어 모델 점수로 최종 문장을 고릅니다. 실제로는 단어 단위의 큰 언어 모델을 씁니다.', 'Speech recognizers produce several acoustically similar candidates and let a language model pick the final sentence — in practice a large word-level model.'),
  pipeline: [
    step('input', '소리 인식 후보', 'Acoustic candidates', '음성 인식기가 비슷하게 들리는 문장 후보 몇 개를 만들었다고 가정합니다.', 'Assume the recognizer produced a few similar-sounding candidates.'),
    step('pre', '후보 정리', 'Prepare candidates', '후보마다 글자 번호로 바꿉니다.', 'Each candidate becomes a list of IDs.'),
    step('model', '자연스러움 점수', 'Naturalness score', '학습한 Transformer가 문장의 글자 하나하나를 얼마나 예상했는지로 퍼플렉서티를 계산합니다.', 'Your Transformer computes perplexity from how well it expected each character.'),
    step('post', '점수순 정렬', 'Sort by score', '퍼플렉서티가 낮을수록(덜 놀랄수록) 자연스러운 문장입니다.', 'Lower perplexity (less surprise) means a more natural sentence.'),
    step('output', '최종 자막', 'Final caption', '1등 문장을 자막으로 씁니다.', 'The winner becomes the caption.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const { stoi } = ctx.data.vocab;
    const sentences = makeCorpus(ctx.hp.corpus, 4000, 777);
    let k = 0;
    const box = h('div', { class: 'stack', style: { gap: '6px' } });
    const run = () => {
      flow.run();
      const rand = mulberry32(ctx.hp.seed * 5 + k);
      const truth = sentences[k % sentences.length];
      const words = truth.replace(/\.$/, '').split(' ');
      const swapped = [...words];
      const a = Math.floor(rand() * words.length);
      const b = (a + 1 + Math.floor(rand() * (words.length - 1))) % words.length;
      [swapped[a], swapped[b]] = [swapped[b], swapped[a]];
      const other = sentences[(k + 3) % sentences.length].replace(/\.$/, '').split(' ');
      const mixed = [...words.slice(0, -1), other[other.length - 1]];
      const chars = [...truth];
      const ci = 2 + Math.floor(rand() * (chars.length - 4));
      chars[ci] = chars[(ci + 5) % chars.length];
      const cands = shuffle([truth, `${swapped.join(' ')}.`, `${mixed.join(' ')}.`, chars.join('')], rand);
      const ranked = rankByScore([...new Set(cands)], (c) => textPerplexity(c, s.nextProbs, stoi));
      box.replaceChildren(...ranked.map((r, i) => h('div', { class: 'mail-row' },
        h('span', null, i === 0 ? '🥇' : `${i + 1}`),
        h('span', null, r.text, r.text === truth ? h('span', { class: 'small tag-good' }, `  ← ${ctx.t('use.tf.original')}`) : ''),
        h('span', { class: 'mono small' }, `ppl ${fmt(r.score)}`))));
    };
    el.append(h('p', { class: 'help' }, ctx.t('use.tf.rerankHelp')), h('div', null, h('button', { class: 'btn', type: 'button', onclick: () => { k++; run(); } }, ctx.t('use.tf.nextAudio'))), box);
    run();
  },
};

const chat = {
  id: 'chat',
  icon: '💬',
  title: lt('대화 응답 이어 쓰기', 'Reply writer'),
  intro: lt('메신저에서 답장을 쓰기 시작하면 나머지를 모델이 이어 씁니다.', 'Start a reply in a messenger and the model finishes it.'),
  realWorld: lt('대화형 AI는 같은 "다음 것 예측"을 수천억 단어로 학습한 거대한 Transformer와, 사람 피드백으로 다듬는 과정(RLHF)을 거쳐 만들어집니다.', 'Chat assistants are giant Transformers doing the same next-token prediction over hundreds of billions of words, then refined with human feedback (RLHF).'),
  pipeline: [
    step('input', '답장 시작', 'Start of reply', '사용자가 답장 앞부분을 씁니다.', 'The user types the beginning of a reply.'),
    step('pre', '번호로 바꾸기', 'To IDs', '글자 번호로 바꿉니다.', 'Characters become IDs.'),
    step('model', '다음 글자 예측', 'Predict next', '학습한 Transformer가 다음 글자 확률을 냅니다.', 'Your Transformer gives next-character probabilities.'),
    step('post', '한 글자씩 뽑기', 'Sample one by one', '온도 0.5로 한 글자를 뽑아 붙이기를 문장이 끝날 때까지 반복합니다.', 'Samples a character at temperature 0.5 and appends it, until the sentence ends.'),
    step('output', '말풍선에 표시', 'Show in bubble', '한 글자씩 타이핑되듯 보여 줍니다.', 'Types it out character by character.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const thread = h('div', { class: 'stack', style: { gap: '8px' } });
    const input = h('input', { class: 'text-input', value: PROMPTS[ctx.hp.corpus] });
    let n = 0;
    let timer = null;
    const bubble = (text, mine) => h('div', { style: { justifySelf: mine ? 'end' : 'start', maxWidth: '80%', padding: '8px 12px', borderRadius: '14px', background: mine ? 'var(--accent)' : 'var(--surface-2)', color: mine ? '#fff' : 'var(--text)' } }, text);
    thread.append(bubble(ctx.t('use.tf.friendMsg'), false));
    const send = () => {
      flow.run();
      clearInterval(timer);
      const start = input.value;
      let rest = '';
      const rand = mulberry32(ctx.hp.seed * 13 + n++);
      for (let i = 0; i < 60; i++) {
        const c = s.generate(start + rest, 1, 0.5, rand);
        rest += c;
        if (c === '.') break;
      }
      const b = bubble('', true);
      const typed = h('span', { style: { opacity: 0.85 } });
      b.append(start, typed);
      thread.append(b);
      let i = 0;
      timer = setInterval(() => { typed.textContent = rest.slice(0, ++i); if (i >= rest.length) clearInterval(timer); }, 35);
    };
    el.append(h('div', { style: { background: 'var(--bg)', borderRadius: '12px', padding: '12px' } }, thread),
      h('div', { class: 'row' }, h('div', { style: { flex: 1 } }, input), h('button', { class: 'btn btn-primary', type: 'button', onclick: send }, ctx.t('use.tf.send'))));
    return { destroy() { clearInterval(timer); } };
  },
};

export const useCases = [search, rerank, chat];

export const USE_TEXT = {
  ko: {
    'use.tf.query': '검색어',
    'use.tf.why': '모델이 주목한 글자 (마지막 글자 기준)',
    'use.tf.rerankHelp': '음성 인식기가 들은 소리로 비슷한 후보 네 개를 만들었습니다. 그중 하나가 실제로 말한 문장입니다. 내 모델이 고른 1등이 원래 문장과 같은지 보세요.',
    'use.tf.nextAudio': '다른 녹음',
    'use.tf.original': '실제로 말한 문장',
    'use.tf.friendMsg': '오늘 뭐 했어?',
    'use.tf.send': '이어 쓰고 보내기',
  },
  en: {
    'use.tf.query': 'Query',
    'use.tf.why': 'Characters the model attended to (for the last one)',
    'use.tf.rerankHelp': 'The recognizer turned a recording into four similar candidates; one is what was actually said. See whether your model\'s top pick matches it.',
    'use.tf.nextAudio': 'Another recording',
    'use.tf.original': 'what was actually said',
    'use.tf.friendMsg': 'What did you do today?',
    'use.tf.send': 'Finish & send',
  },
};
