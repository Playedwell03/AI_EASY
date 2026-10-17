// "Use" tab apps for the digit CNN: read handwritten numbers inside small apps.
import { h, pct } from '../../ui/dom.js';
import { frameDrawing } from '../../shared/digits.js';
import { step, lt } from '../../ui/useView.js';
import { drawPad, imgCell } from '../../ui/pad.js';

// preds: [{ digit, prob }] (null = empty box). Boxes under minConf are flagged for a human to check.
export function readNumber(preds, minConf = 0.6) {
  if (!preds.length) return { text: '', value: null, unsure: [] };
  const text = preds.map((p) => (p ? String(p.digit) : '?')).join('');
  const unsure = preds.map((p, i) => (!p || p.prob < minConf ? i : -1)).filter((i) => i >= 0);
  return { text, value: /^\d+$/.test(text) ? Number(text) : null, unsure };
}

export function calc(a, op, b) {
  if (op === '+') return a + b;
  if (op === '-') return a - b;
  if (op === '×') return a * b;
  if (op === '÷') return b === 0 ? null : a / b;
  return null;
}

// A row of n drawing boxes. Each box shows what the model read and how sure it is.
function digitRow(ctx, n, flow, onChange) {
  const preds = new Array(n).fill(null);
  const boxes = Array.from({ length: n }, (_, i) => {
    const label = h('div', { class: 'mono small', style: { textAlign: 'center', minHeight: '20px' } }, '·');
    const seen = h('div', { style: { width: '40px', margin: '0 auto' } });
    const pad = drawPad((img) => {
      flow.pulse(0);
      setTimeout(() => flow.pulse(1), 120);
      const framed = frameDrawing(img);
      seen.replaceChildren(imgCell(framed, ''));
      setTimeout(() => flow.pulse(2), 240);
      const probs = ctx.session.predictImages([framed])[0];
      const digit = probs.indexOf(Math.max(...probs));
      preds[i] = { digit, prob: probs[digit] };
      label.replaceChildren(h('b', { class: probs[digit] < 0.6 ? 'tag-warn' : '' }, String(digit)), ` ${pct(probs[digit])}`);
      setTimeout(() => { flow.pulse(3); flow.pulse(4); }, 360);
      onChange(preds);
    }, { size: 96, brush: 8 });
    const clear = () => { pad.clear(); preds[i] = null; label.textContent = '·'; seen.replaceChildren(); onChange(preds); };
    return h('div', { class: 'stack', style: { gap: '4px', justifyItems: 'center' } }, pad.canvas, label, seen,
      h('button', { class: 'btn btn-sm', type: 'button', onclick: clear }, ctx.t('m.clear')));
  });
  return { el: h('div', { class: 'row', style: { gap: '10px', alignItems: 'flex-start' } }, boxes), preds };
}

const PIPE = (what) => [
  step('input', '손글씨 입력', 'Handwriting', `${what.ko} 칸에 손으로 씁니다.`, `Handwriting goes into the ${what.en} boxes.`),
  step('pre', '자르고 맞추기', 'Crop & center', '글씨를 잘라 가운데 맞추고 20×20 흑백 이미지로 줄입니다(칸마다 아래 작은 그림).', 'Each box is cropped, centered and shrunk to a 20×20 grayscale image (the small picture under it).'),
  step('model', '숫자 인식', 'Recognize digit', '학습한 CNN이 칸마다 0~9 각각의 확률을 냅니다.', 'Your trained CNN gives each box a probability for every digit 0–9.'),
  step('post', '확신도 확인', 'Check confidence', '가장 높은 확률의 숫자를 고르고, 60%보다 낮으면 "확인 필요"로 표시합니다.', 'Picks the most likely digit; under 60% it is marked "please check".'),
  step('output', '앱 동작', 'App acts', '읽은 숫자로 앱이 할 일을 합니다.', 'The app uses the number it read.'),
];

const REGIONS = [
  lt('수도권 북부', 'North metro'), lt('수도권 남부', 'South metro'), lt('강원 지역', 'East highlands'), lt('충청 지역', 'Central plains'),
  lt('전라 지역', 'Southwest coast'), lt('경상 북부', 'Northeast valley'), lt('경상 남부', 'Southeast harbor'), lt('제주·섬 지역', 'Island region'),
  lt('특수 우편', 'Special mail'), lt('해외 우편', 'Overseas'),
];

const zip = {
  id: 'zip',
  icon: '📮',
  title: lt('우편번호 자동 분류', 'Postcode sorter'),
  intro: lt('봉투에 손으로 쓴 우편번호 다섯 자리를 읽어 어느 지역으로 보낼지 정합니다.', 'Reads a handwritten five-digit postcode on an envelope and decides where to send it.'),
  realWorld: lt('실제 우편 분류기는 수백만 장의 진짜 손글씨로 학습한 모델로 1초에 수십 통을 읽고, 자신 없는 봉투는 사람에게 넘깁니다. 지역 이름은 가상입니다.', 'Real sorting machines read dozens of envelopes a second with models trained on millions of real samples, and pass unsure ones to people. Region names here are made up.'),
  pipeline: PIPE({ ko: '우편번호', en: 'postcode' }),
  render(el, ctx, flow) {
    const result = h('div', { class: 'big-result' }, '…');
    const note = h('div', { class: 'small muted' });
    const row = digitRow(ctx, 5, flow, (preds) => {
      const r = readNumber(preds);
      if (preds.every((p) => !p)) { result.textContent = '…'; note.textContent = ''; return; }
      const first = preds[0]?.digit;
      result.replaceChildren(`📦 ${r.text} → `, first != null ? h('span', null, ctx.L(REGIONS[first])) : '?');
      const empty = preds.filter((p) => !p).length;
      const low = r.unsure.filter((i) => preds[i]);
      note.replaceChildren(empty
        ? h('span', { class: 'muted' }, ctx.t('use.cnn.fillRest', { n: empty }))
        : low.length ? h('span', { class: 'tag-warn' }, ctx.t('use.cnn.unsure', { n: low.map((i) => i + 1).join(', ') })) : h('span', { class: 'tag-good' }, ctx.t('use.cnn.allSure')));
    });
    el.append(h('p', { class: 'help' }, ctx.t('use.cnn.zipHelp')), row.el, result, note);
  },
};

const calculator = {
  id: 'calc',
  icon: '🧮',
  title: lt('손글씨 계산기', 'Handwriting calculator'),
  intro: lt('숫자 두 개를 손으로 쓰고 연산을 고르면 계산해 줍니다.', 'Write two numbers by hand, pick an operation, get the answer.'),
  realWorld: lt('필기 인식 계산기 앱은 숫자뿐 아니라 기호와 수식 구조까지 인식하는 더 큰 모델을 씁니다.', 'Real handwriting calculator apps use larger models that also read symbols and the layout of a formula.'),
  pipeline: PIPE({ ko: '숫자', en: 'number' }),
  render(el, ctx, flow) {
    let op = '+';
    const out = h('div', { class: 'big-result' }, '…');
    let a = [];
    let b = [];
    const update = () => {
      const ra = readNumber(a.filter(Boolean));
      const rb = readNumber(b.filter(Boolean));
      if (ra.value == null || rb.value == null) { out.textContent = '…'; return; }
      const v = calc(ra.value, op, rb.value);
      out.textContent = `${ra.value} ${op} ${rb.value} = ${v == null ? ctx.t('use.cnn.divZero') : Number(v.toFixed(4))}`;
      flow.pulse(4);
    };
    const rowA = digitRow(ctx, 2, flow, (p) => { a = p; update(); });
    const rowB = digitRow(ctx, 2, flow, (p) => { b = p; update(); });
    const ops = ['+', '-', '×', '÷'].map((o) => h('button', { class: `btn${o === op ? ' btn-primary' : ''}`, type: 'button', onclick: () => { op = o; ops.forEach((x) => x.classList.toggle('btn-primary', x.textContent === o)); update(); } }, o));
    el.append(h('p', { class: 'help' }, ctx.t('use.cnn.calcHelp')),
      h('div', { class: 'row', style: { alignItems: 'center', gap: '16px' } }, rowA.el, h('div', { class: 'stack', style: { gap: '6px' } }, ops), rowB.el), out);
  },
};

const lock = {
  id: 'lock',
  icon: '🔐',
  title: lt('손글씨 숫자 자물쇠', 'Handwritten keypad lock'),
  intro: lt('비밀번호 네 자리를 손으로 써서 금고를 엽니다.', 'Open a safe by handwriting a four-digit code.'),
  realWorld: lt('보안 장치는 인식이 틀릴 때의 위험이 크기 때문에, 확신이 낮으면 다시 입력하게 하거나 다른 인증을 함께 씁니다.', 'Security devices cannot afford misreads, so low confidence means "try again" or a second check.'),
  pipeline: PIPE({ ko: '비밀번호', en: 'code' }),
  render(el, ctx, flow) {
    const secret = String(1000 + Math.floor((ctx.hp.seed * 7919) % 9000));
    const status = h('div', { class: 'big-result' }, '🔒');
    const row = digitRow(ctx, 4, flow, (preds) => {
      if (preds.some((p) => !p)) { status.textContent = '🔒'; return; }
      const r = readNumber(preds, 0.8);
      if (r.unsure.length) status.replaceChildren(h('span', { class: 'tag-warn' }, `🔒 ${ctx.t('use.cnn.lockUnsure')}`));
      else if (r.text === secret) status.replaceChildren(h('span', { class: 'tag-good' }, `🔓 ${ctx.t('use.cnn.lockOpen')}`));
      else status.replaceChildren(h('span', { class: 'tag-bad' }, `🔒 ${ctx.t('use.cnn.lockWrong', { code: r.text })}`));
    });
    el.append(h('p', { class: 'help' }, ctx.t('use.cnn.lockHelp', { code: secret })), row.el, status);
  },
};

const receipt = {
  id: 'receipt',
  icon: '🧾',
  title: lt('영수증 금액 합산', 'Receipt totals'),
  intro: lt('손으로 쓴 품목 금액 세 줄을 읽어 합계를 냅니다.', 'Reads three handwritten line amounts and adds them up.'),
  realWorld: lt('경비 정산 앱은 영수증 사진에서 글자 위치를 찾는 모델과 글자를 읽는 모델을 이어 붙여 씁니다.', 'Expense apps chain a model that finds where text is on a photo with one that reads it.'),
  pipeline: PIPE({ ko: '금액', en: 'amount' }),
  render(el, ctx, flow) {
    const lines = [[], [], []];
    const total = h('div', { class: 'big-result' }, '…');
    const update = () => {
      const vals = lines.map((p) => readNumber(p.filter(Boolean)).value);
      const known = vals.filter((v) => v != null);
      total.textContent = known.length ? `${ctx.t('use.cnn.total')}: ${known.reduce((x, y) => x + y, 0).toLocaleString()} ${ctx.t('use.cnn.won')}` : '…';
    };
    const items = [lt('커피', 'Coffee'), lt('빵', 'Bread'), lt('우유', 'Milk')];
    el.append(h('p', { class: 'help' }, ctx.t('use.cnn.receiptHelp')),
      ...items.map((name, i) => h('div', { class: 'row', style: { gap: '12px' } }, h('b', { style: { width: '60px' } }, ctx.L(name)),
        digitRow(ctx, 2, flow, (p) => { lines[i] = p; update(); }).el, h('span', { class: 'muted' }, `00 ${ctx.t('use.cnn.won')}`))),
      total);
  },
};

export const useCases = [zip, calculator, lock, receipt];

export const USE_TEXT = {
  ko: {
    'use.cnn.zipHelp': '칸마다 숫자 하나씩 쓰세요. 첫 자리로 지역을 정합니다.',
    'use.cnn.fillRest': '남은 칸 {n}개를 채워 주세요',
    'use.cnn.unsure': '{n}번째 칸은 확신이 낮아요 → 사람이 확인',
    'use.cnn.allSure': '모든 칸을 자신 있게 읽었어요 → 자동 분류',
    'use.cnn.calcHelp': '왼쪽과 오른쪽에 두 자리 숫자를 쓰고 연산을 고르세요(한 자리만 써도 됩니다).',
    'use.cnn.divZero': '0으로 나눌 수 없음',
    'use.cnn.lockHelp': '이 금고의 비밀번호는 {code}입니다. 네 칸에 써 보세요. 보안 때문에 80% 이상 확신할 때만 엽니다.',
    'use.cnn.lockOpen': '열렸습니다!',
    'use.cnn.lockWrong': '{code}는 틀린 번호입니다',
    'use.cnn.lockUnsure': '글씨를 확실히 못 읽었어요. 다시 써 주세요',
    'use.cnn.receiptHelp': '품목마다 금액 앞 두 자리를 쓰세요(뒤 00은 고정).',
    'use.cnn.total': '합계',
    'use.cnn.won': '원',
  },
  en: {
    'use.cnn.zipHelp': 'Write one digit per box. The first digit picks the region.',
    'use.cnn.fillRest': 'Fill in the remaining {n} box(es)',
    'use.cnn.unsure': 'Low confidence in box {n} → a person checks it',
    'use.cnn.allSure': 'Read every box confidently → sorted automatically',
    'use.cnn.calcHelp': 'Write a number (one or two digits) on each side and pick an operation.',
    'use.cnn.divZero': 'cannot divide by zero',
    'use.cnn.lockHelp': 'This safe\'s code is {code}. Write it in the four boxes. For security, it only opens when at least 80% sure.',
    'use.cnn.lockOpen': 'Unlocked!',
    'use.cnn.lockWrong': '{code} is the wrong code',
    'use.cnn.lockUnsure': 'Could not read that clearly. Please write it again',
    'use.cnn.receiptHelp': 'Write the first two digits of each amount (the trailing 00 is fixed).',
    'use.cnn.total': 'Total',
    'use.cnn.won': 'won',
  },
};
