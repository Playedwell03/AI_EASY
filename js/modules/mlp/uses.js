// "Use" tab apps for the MLP: two measurements in, a yes/no decision out.
import { h, pct } from '../../ui/dom.js';
import { scatter } from '../../ui/chart.js';
import { mulberry32 } from '../../core/rng.js';
import { step, lt } from '../../ui/useView.js';
import { L } from '../../i18n.js';

const GRID = 41;

// Map a real-world measurement range onto the model's [-1, 1] input plane.
export function toPlane(v, min, max) {
  return Math.max(-1, Math.min(1, ((v - min) / (max - min)) * 2 - 1));
}

// A small decision map: the trained boundary in the background, points on top.
function decisionMap(ctx, { onClick, size = 300 } = {}) {
  const canvas = h('canvas', { style: { aspectRatio: '1 / 1', borderRadius: '10px', cursor: onClick ? 'crosshair' : 'default' } });
  const el = h('div', { class: 'chart-wrap', style: { maxWidth: `${size}px` } }, canvas);
  const chart = scatter(canvas, [], { field: { n: GRID, values: ctx.session.grid(GRID) } });
  if (onClick) canvas.addEventListener('click', (ev) => onClick(chart.toData(ev)));
  return { el, chart };
}

const prob = (ctx, x, y) => ctx.session.predictPoints([[x, y]])[0];

const slider = (label, min, max, value, oninput) => {
  const val = h('span', { class: 'val' }, String(value));
  const input = h('input', { type: 'range', min, max, step: 1, value, oninput: (e) => { val.textContent = e.target.value; oninput(Number(e.target.value)); } });
  return h('div', { class: 'control' }, h('label', null, h('span', null, label), val), input);
};

const spam = {
  id: 'spam',
  icon: '📧',
  title: lt('스팸 메일 필터', 'Spam filter'),
  intro: lt('메일의 링크 수와 느낌표 수를 보고 스팸함으로 보낼지 정합니다.', 'Looks at how many links and exclamation marks a mail has, and decides whether it goes to spam.'),
  realWorld: lt('수백 가지 특징(보낸 사람, 단어, 첨부 파일…)과 수백만 통의 진짜 메일로 학습합니다. 여기서는 내 모델이 배운 경계 모양을 그대로 "스팸 구역"으로 씁니다.', 'Hundreds of features (sender, words, attachments…) learned from millions of real mails. Here, the shape your model learned is used directly as the "spam zone".'),
  pipeline: [
    step('input', '새 메일 도착', 'New mail arrives', '메일 한 통이 들어옵니다.', 'A mail comes in.'),
    step('pre', '특징 세기', 'Count features', '링크 수(0~20)와 느낌표 수(0~20)를 세고 모델 입력 평면에 놓습니다. 0개는 평면의 한가운데, 20개는 가장자리입니다.', 'Counts links (0–20) and exclamation marks (0–20) and places them on the model\'s input plane: 0 at the centre, 20 at the edge.'),
    step('model', '스팸 확률 계산', 'Spam probability', '학습한 MLP가 두 숫자를 보고 "주황 클래스(=스팸)일 확률"을 냅니다.', 'Your trained MLP turns the two numbers into the probability of the orange class (= spam).'),
    step('post', '50% 기준 판단', 'Decide at 50%', '확률이 50% 이상이면 스팸함으로 보냅니다.', 'At 50% or more, it goes to the spam folder.'),
    step('output', '받은편지함 / 스팸함', 'Inbox / Spam', '분류된 결과가 메일함에 표시됩니다.', 'The mail shows up in the chosen folder.'),
  ],
  render(el, ctx, flow) {
    const rand = mulberry32(ctx.hp.seed + 5);
    const subjects = [lt('회의 일정 안내', 'Meeting schedule'), lt('축하합니다!!! 당첨!!!', 'Congrats!!! You won!!!'), lt('지난주 사진 보내요', 'Photos from last week'), lt('무료 쿠폰 받기!!', 'Get a free coupon!!'), lt('과제 제출 확인', 'Assignment received'), lt('지금 클릭하세요!!!', 'Click now!!!')];
    const mails = subjects.map((s, i) => ({ s, links: Math.round(rand() * 20), bangs: Math.round(rand() * 20), i }));
    let links = 3;
    let bangs = 2;
    const map = decisionMap(ctx);
    const verdict = h('div', { class: 'big-result' });
    const list = h('div', { class: 'stack', style: { gap: '6px' } });
    const judge = (l, b) => prob(ctx, toPlane(l, -20, 20), toPlane(b, -20, 20));
    const draw = () => {
      flow.run();
      const p = judge(links, bangs);
      verdict.replaceChildren(p >= 0.5 ? h('span', { class: 'tag-bad' }, `🚫 ${ctx.t('use.mlp.toSpam')} · ${pct(p)}`) : h('span', { class: 'tag-good' }, `📥 ${ctx.t('use.mlp.toInbox')} · ${pct(p)}`));
      map.chart.update({
        points: mails.map((m) => ({ x: toPlane(m.links, -20, 20), y: toPlane(m.bangs, -20, 20), c: judge(m.links, m.bangs) >= 0.5 ? 1 : 0, r: 4 })),
        marker: { x: toPlane(links, -20, 20), y: toPlane(bangs, -20, 20) },
      });
    };
    list.append(...mails.map((m) => {
      const p = judge(m.links, m.bangs);
      return h('div', { class: 'mail-row' }, h('span', null, p >= 0.5 ? '🚫' : '📥'), h('span', null, L(m.s), h('span', { class: 'muted small' }, `  · 🔗${m.links} ❗${m.bangs}`)), h('span', { class: 'mono small' }, pct(p)));
    }));
    el.append(h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        h('b', null, ctx.t('use.mlp.compose')),
        slider(ctx.t('use.mlp.links'), 0, 20, links, (v) => { links = v; draw(); }),
        slider(ctx.t('use.mlp.bangs'), 0, 20, bangs, (v) => { bangs = v; draw(); }),
        verdict,
        h('b', null, ctx.t('use.mlp.inbox')), list),
      h('div', null, map.el, h('p', { class: 'small muted' }, ctx.t('use.mlp.mapHelp')))));
    draw();
  },
};

const factory = {
  id: 'factory',
  icon: '🏭',
  title: lt('공장 품질 검사', 'Factory quality check'),
  intro: lt('컨베이어 위 제품의 온도와 진동을 재서 불량품을 골라냅니다.', 'Measures temperature and vibration of products on a conveyor and pulls out the faulty ones.'),
  realWorld: lt('실제 공장은 수십 개 센서와 카메라를 함께 쓰고, 불량을 놓치는 비용과 멀쩡한 제품을 버리는 비용을 따져 기준선을 정합니다.', 'Real lines combine dozens of sensors and cameras, and set the threshold by weighing the cost of a missed defect against discarding good products.'),
  pipeline: [
    step('input', '센서 측정', 'Sensors', '제품이 지나가면 온도(20~80℃)와 진동(0~10)을 잽니다.', 'As a product passes, temperature (20–80°C) and vibration (0–10) are measured.'),
    step('pre', '범위 맞추기', 'Rescale', '평범한 값(50℃, 진동 0)이 평면 한가운데에 오도록 두 값을 옮깁니다. 극단적인 값일수록 가장자리로 갑니다.', 'Both values are shifted so ordinary readings (50°C, no vibration) sit at the centre of the plane; extreme readings move to the edge.'),
    step('model', '불량 확률', 'Defect probability', '학습한 MLP가 "주황(=불량)"일 확률을 계산합니다.', 'Your MLP computes the probability of orange (= defective).'),
    step('post', '기준선 비교', 'Threshold', '확률이 기준선을 넘으면 불량으로 분류합니다. 기준선을 바꿔 보세요.', 'Above the threshold, it is classified as defective. Try moving the threshold.'),
    step('output', '통과 / 제거', 'Pass / Reject', '정상품은 포장으로, 불량품은 제거 라인으로 갑니다.', 'Good products go to packing, defective ones to the reject line.'),
  ],
  render(el, ctx, flow) {
    const rand = mulberry32(ctx.hp.seed + 11);
    let threshold = 0.5;
    const history = [];
    const map = decisionMap(ctx);
    const belt = h('div', { class: 'row', style: { gap: '6px', minHeight: '40px' } });
    const stats = h('div', { class: 'small' });
    const thVal = h('span', { class: 'val' }, '50%');
    const next = (count) => {
      for (let i = 0; i < count; i++) {
        const temp = 20 + rand() * 60;
        const vib = rand() * 10;
        const p = prob(ctx, toPlane(temp, 20, 80), toPlane(vib, -10, 10));
        history.push({ temp, vib, p });
      }
      flow.run();
      render();
    };
    const render = () => {
      const recent = history.slice(-10);
      belt.replaceChildren(...recent.map((it) => h('span', { class: 'chip', title: `${it.temp.toFixed(0)}℃ · ${it.vib.toFixed(1)}`, style: { cursor: 'default' } }, it.p >= threshold ? '❌' : '📦', h('span', { class: 'mono small' }, pct(it.p)))));
      const bad = history.filter((it) => it.p >= threshold).length;
      stats.textContent = ctx.t('use.mlp.factoryStats', { n: history.length, bad, good: history.length - bad });
      map.chart.update({ points: history.map((it) => ({ x: toPlane(it.temp, 20, 80), y: toPlane(it.vib, 0, 10), c: it.p >= threshold ? 1 : 0, r: 3 })) });
    };
    const th = h('input', { type: 'range', min: 0.05, max: 0.95, step: 0.05, value: 0.5, oninput: (e) => { threshold = Number(e.target.value); thVal.textContent = pct(threshold); flow.pulse(3); render(); } });
    el.append(h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        h('div', { class: 'row' },
          h('button', { class: 'btn btn-primary', type: 'button', onclick: () => next(1) }, ctx.t('use.mlp.nextProduct')),
          h('button', { class: 'btn', type: 'button', onclick: () => next(20) }, ctx.t('use.mlp.next20'))),
        h('div', { class: 'control' }, h('label', null, h('span', null, ctx.t('use.mlp.threshold')), thVal), th),
        h('b', null, ctx.t('use.mlp.belt')), belt, stats),
      h('div', null, map.el, h('p', { class: 'small muted' }, ctx.t('use.mlp.factoryMap')))));
    next(8);
  },
};

const npc = {
  id: 'npc',
  icon: '🎮',
  title: lt('게임 NPC 경계 구역', 'Game guard NPC'),
  intro: lt('플레이어가 지도 어디에 있는지 보고 경비병이 경계 태세를 취할지 정합니다.', 'A guard decides whether to go on alert based on where the player stands on the map.'),
  realWorld: lt('실제 게임은 보통 사람이 규칙을 짜지만, 플레이 기록으로 학습한 모델로 더 자연스러운 행동을 만들기도 합니다.', 'Most games hand-write these rules, but some learn more natural behavior from recorded play.'),
  pipeline: [
    step('input', '플레이어 이동', 'Player moves', '지도를 누르면 플레이어가 그 위치로 이동합니다.', 'Click the map to move the player there.'),
    step('pre', '좌표 읽기', 'Read position', '플레이어의 (x, y) 좌표를 그대로 모델에 줍니다.', 'The player\'s (x, y) position goes straight into the model.'),
    step('model', '위험도 계산', 'Danger score', '학습한 MLP가 이 위치가 "주황(=경계 구역)"일 확률을 냅니다.', 'Your MLP gives the probability that this spot is orange (= guarded zone).'),
    step('post', '행동 고르기', 'Pick behavior', '확률에 따라 평온 / 의심 / 경계 중 하나를 고릅니다.', 'Depending on the probability: calm, suspicious, or alert.'),
    step('output', 'NPC 반응', 'NPC reacts', '경비병의 말풍선과 표정이 바뀝니다.', 'The guard\'s speech bubble and face change.'),
  ],
  render(el, ctx, flow) {
    const face = h('div', { style: { fontSize: '44px' } }, '😴');
    const bubble = h('div', { class: 'gen-text' }, ctx.t('use.mlp.npcIdle'));
    const map = decisionMap(ctx, {
      onClick: ({ x, y }) => {
        flow.run();
        const p = prob(ctx, x, y);
        map.chart.update({ marker: { x, y } });
        const [f, key] = p >= 0.75 ? ['😠', 'use.mlp.npcAlert'] : p >= 0.4 ? ['🤨', 'use.mlp.npcSus'] : ['😌', 'use.mlp.npcCalm'];
        face.textContent = f;
        bubble.textContent = `${ctx.t(key)} (${pct(p)})`;
      },
    });
    el.append(h('div', { class: 'grid-2' },
      h('div', null, map.el, h('p', { class: 'small muted' }, ctx.t('use.mlp.npcHelp'))),
      h('div', { class: 'stack', style: { alignContent: 'start' } }, h('div', { class: 'row' }, face, h('b', null, ctx.t('use.mlp.guard'))), bubble)));
  },
};

export const useCases = [spam, factory, npc];

export const USE_TEXT = {
  ko: {
    'use.mlp.compose': '새 메일 만들어 보기',
    'use.mlp.toSpam': '스팸함으로',
    'use.mlp.toInbox': '받은편지함으로',
    'use.mlp.links': '링크 수',
    'use.mlp.bangs': '느낌표 수',
    'use.mlp.inbox': '도착한 메일들',
    'use.mlp.mapHelp': '배경: 내 모델이 배운 경계(주황 = 스팸 구역). 점: 도착한 메일, 십자: 지금 만든 메일.',
    'use.mlp.nextProduct': '다음 제품',
    'use.mlp.next20': '20개 흘려보내기',
    'use.mlp.threshold': '불량 판정 기준선',
    'use.mlp.belt': '컨베이어 (최근 10개)',
    'use.mlp.factoryStats': '검사 {n}개 · 정상 {good}개 · 불량 {bad}개',
    'use.mlp.factoryMap': '가로: 온도, 세로: 진동. 배경은 내 모델의 판단 지도입니다.',
    'use.mlp.guard': '경비병',
    'use.mlp.npcIdle': '지도를 눌러 플레이어를 움직여 보세요.',
    'use.mlp.npcCalm': '조용하군…',
    'use.mlp.npcSus': '방금 무슨 소리지?',
    'use.mlp.npcAlert': '거기 누구냐!',
    'use.mlp.npcHelp': '주황 구역이 경비병이 지키는 곳입니다. 이 구역 모양은 여러분이 학습시킨 그대로입니다.',
  },
  en: {
    'use.mlp.compose': 'Write a new mail',
    'use.mlp.toSpam': 'To spam',
    'use.mlp.toInbox': 'To inbox',
    'use.mlp.links': 'Links',
    'use.mlp.bangs': 'Exclamation marks',
    'use.mlp.inbox': 'Incoming mail',
    'use.mlp.mapHelp': 'Background: the boundary your model learned (orange = spam zone). Dots: incoming mail; cross: the mail you wrote.',
    'use.mlp.nextProduct': 'Next product',
    'use.mlp.next20': 'Run 20 products',
    'use.mlp.threshold': 'Defect threshold',
    'use.mlp.belt': 'Conveyor (last 10)',
    'use.mlp.factoryStats': '{n} checked · {good} good · {bad} defective',
    'use.mlp.factoryMap': 'Across: temperature, up: vibration. The background is your model\'s decision map.',
    'use.mlp.guard': 'Guard',
    'use.mlp.npcIdle': 'Click the map to move the player.',
    'use.mlp.npcCalm': 'All quiet…',
    'use.mlp.npcSus': 'What was that noise?',
    'use.mlp.npcAlert': 'Who goes there!',
    'use.mlp.npcHelp': 'The orange zone is what the guard protects. Its shape is exactly what you trained.',
  },
};
