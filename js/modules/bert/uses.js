// "Use" tab apps for the masked language model.
import { h, pct } from '../../ui/dom.js';
import { step, lt } from '../../ui/useView.js';

export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

// Replace a character only when it looks unlikely AND the model's best guess is far more likely.
export const shouldFix = (origProb, best, { low = 0.05, ratio = 5 } = {}) => origProb < low && best.p > ratio * origProb && best.p > 0.2;

export function nearest(query, items, k) {
  return items.map((it) => ({ ...it, score: cosine(query, it.v) })).sort((a, b) => b.score - a.score).slice(0, k);
}

const restore = {
  id: 'restore',
  icon: '🩹',
  title: lt('지워진 글자 복원', 'Restore damaged text'),
  intro: lt('얼룩져서 지워진 글자(?)를 앞뒤 문맥으로 되살립니다.', 'Brings back characters lost to smudges (?) using the context on both sides.'),
  realWorld: lt('오래된 문서·비석 복원 연구와 OCR(문자 인식) 후처리에서 실제로 마스크 언어 모델로 빈칸을 채웁니다.', 'Masked language models are really used to fill gaps in old documents and to clean up OCR output.'),
  pipeline: [
    step('input', '손상된 문장', 'Damaged text', '알아볼 수 없는 글자가 ?로 표시된 문장입니다.', 'A sentence where unreadable characters are marked ?.'),
    step('pre', '? → [MASK]', '? → [MASK]', '? 자리를 [MASK] 토큰으로 바꿉니다.', 'Each ? becomes a [MASK] token.'),
    step('model', '양방향 빈칸 맞히기', 'Bidirectional fill-in', '학습한 BERT가 앞뒤 글자를 모두 보고 [MASK] 자리의 원래 글자 확률을 냅니다.', 'Your BERT looks at characters on both sides and scores the original character for each [MASK].'),
    step('post', '가장 그럴듯한 글자', 'Most likely character', '자리마다 1등 글자를 고릅니다.', 'Picks the top character for each slot.'),
    step('output', '복원된 문장', 'Restored text', '복원 결과와 후보를 보여 줍니다.', 'Shows the restored sentence and candidates.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const examples = ctx.hp.corpus === 'ko'
      ? ['오늘 고양?는 공원에서 우?를 마신다.', '아침에 ?생님은 학교에서 책을 ?는다.', '주말에 아이는 바다에서 노래를 부?다.']
      : ['the c?t drinks some m?lk at home today.', 'my friend reads a b??k at school.', 'a child sings a s?ng by the sea.'];
    const input = h('input', { class: 'text-input', value: examples[0] });
    const out = h('div', { class: 'big-result', style: { fontSize: '18px' } });
    const cands = h('div', { class: 'row', style: { gap: '12px', alignItems: 'flex-start' } });
    const run = () => {
      flow.run();
      const chars = [...input.value];
      const pos = chars.map((c, i) => (c === '?' ? i : -1)).filter((i) => i >= 0);
      if (!pos.length) { out.textContent = ctx.t('use.bert.noQ'); cands.replaceChildren(); return; }
      const fills = s.fillMask(input.value, pos);
      const fixed = [...chars];
      fills.forEach((f) => { fixed[f.pos] = f.top[0].c; });
      out.replaceChildren(...fixed.map((c, i) => (pos.includes(i) ? h('span', { class: 'tag-good', style: { textDecoration: 'underline' } }, c) : c)));
      cands.replaceChildren(...fills.map((f, j) => h('div', { class: 'stat', style: { minWidth: '120px' } },
        h('div', { class: 'label' }, ctx.t('use.bert.slot', { n: j + 1 })),
        ...f.top.slice(0, 3).map((x) => h('div', { class: 'mono small' }, `${x.c === ' ' ? '␣' : x.c}  ${pct(x.p)}`)))));
    };
    input.addEventListener('input', run);
    el.append(h('p', { class: 'help' }, ctx.t('use.bert.restoreHelp')),
      h('div', { class: 'row', style: { gap: '6px' } }, examples.map((ex) => h('button', { class: 'chip', type: 'button', onclick: () => { input.value = ex; run(); } }, ex))),
      input, out, cands);
    run();
  },
};

const fixer = {
  id: 'fixer',
  icon: '✏️',
  title: lt('오타 교정기', 'Typo fixer'),
  intro: lt('글자를 하나씩 가려 보며 "원래 이 글자가 맞나?"를 확인하고, 더 그럴듯한 글자를 제안합니다.', 'Hides each character in turn, checks whether it belongs there, and suggests a likelier one.'),
  realWorld: lt('맞춤법 교정 서비스는 이런 모델에 사전, 띄어쓰기 규칙, 사용자 피드백을 더해 고칠지 말지를 정합니다.', 'Spelling services combine models like this with dictionaries, spacing rules and user feedback to decide what to fix.'),
  pipeline: [
    step('input', '문장 입력', 'Text', '검사할 문장을 입력합니다.', 'Type the text to check.'),
    step('pre', '한 글자씩 가리기', 'Hide one at a time', '각 글자를 차례로 [MASK]로 가린 문장을 만듭니다.', 'Makes a copy with each character hidden in turn.'),
    step('model', '원래 글자 맞히기', 'Guess the original', '학습한 BERT가 가린 자리에 올 글자 확률을 냅니다. 실제 글자의 확률이 낮으면 의심합니다.', 'Your BERT scores each hidden slot; a low score for the actual character is suspicious.'),
    step('post', '고칠지 결정', 'Decide to fix', '실제 글자 확률이 5% 미만이고, 1등 후보가 5배 넘게 그럴듯할 때만 고칩니다.', 'Fixes only if the actual character is under 5% and the top guess is over 5× more likely.'),
    step('output', '교정 결과', 'Corrected text', '고친 글자를 표시합니다.', 'Shows the corrected characters.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const input = h('input', { class: 'text-input', value: ctx.hp.corpus === 'ko' ? '저녁에 강아지는 숲에서 사과를 머는다.' : 'the dog eats an applr in the park.' });
    const out = h('div', { class: 'gen-text', style: { lineHeight: 2 } });
    const list = h('div', { class: 'small' });
    const run = () => {
      flow.run();
      const chars = [...input.value];
      const fills = chars.map((c, i) => (c === ' ' ? null : s.fillMask(input.value, [i])[0]));
      const fixes = [];
      out.replaceChildren(...chars.map((c, i) => {
        const f = fills[i];
        if (!f) return c;
        const k = ctx.data.vocab.stoi[c];
        const origP = k == null ? 0 : f.probs[k];
        const best = f.top[0];
        if (best.c !== c && shouldFix(origP, best)) {
          fixes.push({ i, from: c, to: best.c, p: best.p, o: origP });
          return h('span', { title: `${c} ${pct(origP)} → ${best.c} ${pct(best.p)}` }, h('s', { class: 'tag-bad' }, c), h('b', { class: 'tag-good' }, best.c));
        }
        return c;
      }));
      list.replaceChildren(fixes.length ? ctx.t('use.bert.fixes', { list: fixes.map((f) => `${f.from}→${f.to} (${pct(f.o)}→${pct(f.p)})`).join(', ') }) : ctx.t('use.bert.noFix'));
    };
    el.append(h('p', { class: 'help' }, ctx.t('use.bert.fixHelp')), h('div', { class: 'row' }, h('div', { style: { flex: 1 } }, input), h('button', { class: 'btn btn-primary', type: 'button', onclick: run }, ctx.t('use.bert.check'))), out, list);
    run();
  },
};

const similar = {
  id: 'similar',
  icon: '🧭',
  title: lt('비슷한 문장 찾기', 'Find similar sentences'),
  intro: lt('문장을 벡터(숫자 목록)로 바꿔서, 뜻이 비슷한 문장을 가까운 순서로 찾아 줍니다.', 'Turns sentences into vectors and finds the closest ones in meaning.'),
  realWorld: lt('FAQ 챗봇, 사내 문서 검색, 중복 질문 찾기는 문장 벡터 검색(임베딩 검색)을 씁니다. 실제로는 문장 의미를 따로 더 학습한 모델을 씁니다.', 'FAQ bots, document search and duplicate-question detection use sentence-vector (embedding) search, usually with models trained further specifically for sentence meaning.'),
  pipeline: [
    step('input', '찾을 문장', 'Query sentence', '찾고 싶은 내용을 문장으로 입력합니다.', 'Type a sentence describing what you want.'),
    step('pre', '번호로 바꾸기', 'To IDs', '글자를 번호로 바꿉니다.', 'Characters become IDs.'),
    step('model', '문장 벡터 만들기', 'Sentence vector', '학습한 BERT의 출력 벡터들을 평균 내어 문장 하나를 숫자 목록 하나로 만듭니다.', 'Averages your BERT\'s output vectors into one vector per sentence.'),
    step('post', '코사인 유사도', 'Cosine similarity', '미리 계산해 둔 문장 벡터들과 방향이 얼마나 같은지 비교합니다.', 'Compares its direction with the pre-computed vectors of the other sentences.'),
    step('output', '가까운 문장 5개', 'Top 5 matches', '가장 비슷한 문장을 보여 줍니다.', 'Shows the closest sentences.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const pool = [...new Set([...ctx.data.sentences.val, ...ctx.data.sentences.test])].slice(0, 80);
    const items = pool.map((text) => ({ text, v: s.sentenceVector(text) }));
    const input = h('input', { class: 'text-input', value: ctx.hp.corpus === 'ko' ? '강아지가 우유를 마신다' : 'the dog drinks milk' });
    const list = h('div', { class: 'stack', style: { gap: '6px' } });
    const run = () => {
      flow.run();
      list.replaceChildren(...nearest(s.sentenceVector(input.value), items, 5).map((r, i) => h('div', { class: 'mail-row' }, h('span', null, `${i + 1}`), h('span', null, r.text), h('span', { class: 'mono small' }, r.score.toFixed(3)))));
    };
    input.addEventListener('change', run);
    el.append(h('p', { class: 'help' }, ctx.t('use.bert.simHelp', { n: items.length })),
      h('div', { class: 'row' }, h('div', { style: { flex: 1 } }, input), h('button', { class: 'btn btn-primary', type: 'button', onclick: run }, ctx.t('use.bert.search'))), list);
    run();
  },
};

export const useCases = [restore, fixer, similar];

export const USE_TEXT = {
  ko: {
    'use.bert.noQ': '문장에 ?를 넣어 지워진 글자를 표시해 주세요.',
    'use.bert.slot': '{n}번째 ? 후보',
    'use.bert.restoreHelp': '예시를 고르거나 직접 입력하세요. 지워진 글자 자리는 ?로 표시합니다.',
    'use.bert.fixHelp': '예시 문장에는 오타가 하나 있습니다. 교정기가 찾아내는지 보세요. 직접 오타를 넣어 봐도 됩니다.',
    'use.bert.check': '검사',
    'use.bert.fixes': '고친 곳: {list}',
    'use.bert.noFix': '고칠 곳을 찾지 못했습니다.',
    'use.bert.simHelp': '검사·시험용 문장 {n}개 중에서 입력한 문장과 가장 가까운 것을 찾습니다. 같은 글자가 많지 않아도 비슷한 뜻(같은 목적어·동사)이면 가깝게 나오는지 보세요.',
    'use.bert.search': '찾기',
  },
  en: {
    'use.bert.noQ': 'Put ? where characters are missing.',
    'use.bert.slot': 'Candidates for ? #{n}',
    'use.bert.restoreHelp': 'Pick an example or type your own. Mark missing characters with ?.',
    'use.bert.fixHelp': 'The example has one typo. See whether the fixer finds it — or add your own typo.',
    'use.bert.check': 'Check',
    'use.bert.fixes': 'Fixed: {list}',
    'use.bert.noFix': 'Nothing to fix found.',
    'use.bert.simHelp': 'Searches {n} validation/test sentences for the ones closest to yours. See whether similar meaning (same object and verb) comes out close even without many shared characters.',
    'use.bert.search': 'Search',
  },
};
