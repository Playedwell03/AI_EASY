// "Use" tab apps for the Jev-style decision model.
import { h, pct } from '../../ui/dom.js';
import { lineChart, chartBox } from '../../ui/chart.js';
import { mulberry32 } from '../../core/rng.js';
import { step, lt } from '../../ui/useView.js';
import { L } from '../../i18n.js';
import { FIELDS, ROUTES, makeLoanCases } from './data.js';

// Decide automatically when confidence (max(p, 1−p)) ≥ threshold; otherwise a person decides.
export function autoSplit(pYes, yes, threshold) {
  let auto = 0;
  let wrong = 0;
  pYes.forEach((p, i) => {
    if (Math.max(p, 1 - p) >= threshold) {
      auto++;
      if ((p >= 0.5) !== yes[i]) wrong++;
    }
  });
  return { auto, human: pYes.length - auto, autoRate: auto / pYes.length, errorRate: auto ? wrong / auto : 0 };
}

export function queueCounts(routeProbs) {
  const c = [0, 0, 0];
  for (const p of routeProbs) c[p.indexOf(Math.max(...p))]++;
  return c;
}

const fieldText = (f) => FIELDS.map((d) => (d.type === 'num' ? `${L(d.label)} ${f[d.key]}` : `${L(d.label)} ${L(d.options[f[d.key]])}`)).join(' · ');

const PIPE = (q, decide) => [
  step('input', '신청서 접수', 'Application received', '신청서(상태)의 다섯 항목이 들어옵니다.', 'The five fields of an application (the state) arrive.'),
  step('pre', '토큰 + 질문 붙이기', 'Tokens + question', `항목을 토큰으로 바꾸고 형식이 정해진 질문(${q.ko})을 붙입니다.`, `Fields become tokens and the typed question (${q.en}) is attached.`),
  step('model', '보정된 답', 'Calibrated answer', '학습한 Jev식 모델이 질문 형식에 맞는 답(확률·선택·점수)을 70ms 안팎으로 바로 냅니다. 글을 쓰지 않습니다.', 'Your Jev-style model answers in the question\'s format (probability, choice, score) right away — no text is written.'),
  step('post', decide.ko, decide.en, decide.dko, decide.den),
  step('output', '처리 결과', 'Outcome', '업무 시스템에 결과가 기록됩니다.', 'The outcome is recorded in the business system.'),
];

const desk = {
  id: 'desk',
  icon: '🏦',
  title: lt('대출 심사 창구', 'Loan desk'),
  intro: lt('신청서를 받으면 승인 확률을 보고, 은행이 정한 기준선에 따라 승인/거절합니다.', 'For each application, reads the approval probability and approves or declines by the bank\'s threshold.'),
  realWorld: lt('실제 금융 심사는 법적 설명 의무와 공정성 검증이 필요해서, 모델 점수 하나로 바로 결정하지 않고 근거를 함께 남깁니다. 여기 신청서는 모두 가상입니다.', 'Real credit decisions carry legal duties to explain and to check fairness, so a score alone never decides — reasons are kept too. All applications here are fictional.'),
  pipeline: PIPE(lt('승인할까요?', 'approve?'), { ko: '기준선 비교', en: 'Compare to threshold', dko: '승인 확률이 기준선 이상이면 승인합니다. 기준선을 올리면 위험한 대출은 줄지만 좋은 고객도 놓칩니다.', den: 'Approves at or above the threshold. Raising it avoids risky loans but turns away good customers too.' }),
  render(el, ctx, flow) {
    const s = ctx.session;
    const { fields, answers } = ctx.data.test;
    const preds = s.predict(fields);
    let th = 0.5;
    const list = h('div', { class: 'stack', style: { gap: '6px' } });
    const stats = h('div', { class: 'small' });
    const tv = h('span', { class: 'val' }, '50%');
    const draw = () => {
      flow.run();
      let goodApproved = 0;
      let badApproved = 0;
      let goodDeclined = 0;
      preds.forEach((p, i) => {
        const ok = p.approve >= th;
        if (ok && answers[i].approve) goodApproved++;
        if (ok && !answers[i].approve) badApproved++;
        if (!ok && answers[i].approve) goodDeclined++;
      });
      stats.textContent = ctx.t('use.jev.deskStats', { n: preds.length, a: goodApproved, b: badApproved, c: goodDeclined });
      list.replaceChildren(...preds.slice(0, 6).map((p, i) => h('div', { class: 'mail-row' },
        h('span', null, p.approve >= th ? '✅' : '⛔'),
        h('span', { class: 'small' }, fieldText(fields[i])),
        h('span', { class: 'mono small' }, pct(p.approve)))));
    };
    el.append(h('p', { class: 'help' }, ctx.t('use.jev.deskHelp')),
      h('div', { class: 'control', style: { maxWidth: '340px' } }, h('label', null, h('span', null, ctx.t('use.jev.threshold')), tv),
        h('input', { type: 'range', min: 0.05, max: 0.95, step: 0.05, value: th, oninput: (e) => { th = Number(e.target.value); tv.textContent = pct(th); draw(); } })),
      list, stats);
    draw();
  },
};

const routing = {
  id: 'routing',
  icon: '📨',
  title: lt('문의 자동 배정', 'Automatic routing'),
  intro: lt('[선택] 질문으로 신청서마다 일반 처리 / 심사역 검토 / 거절 안내 중 어느 창구로 보낼지 정합니다.', 'Uses the [choice] question to send each application to the standard, manual-review or decline queue.'),
  realWorld: lt('고객센터 문의 분류, 보험 청구 배정처럼 "정해진 몇 가지 중 하나"를 고르는 업무가 이런 결정형 모델의 대표적인 쓰임입니다.', 'Picking one of a few fixed options — routing support tickets, assigning insurance claims — is the classic job for decision models like this.'),
  pipeline: PIPE(lt('어느 창구로?', 'which queue?'), { ko: '가장 높은 창구로', en: 'Highest queue', dko: '세 창구 확률 중 가장 높은 곳으로 보냅니다.', den: 'Sends it to the queue with the highest probability.' }),
  render(el, ctx, flow) {
    const s = ctx.session;
    let batch = 0;
    const lanes = h('div', { class: 'grid-3' });
    const run = () => {
      flow.run();
      const { fields, answers } = makeLoanCases(20, mulberry32(ctx.hp.seed * 101 + batch++));
      const preds = s.predict(fields);
      const counts = queueCounts(preds.map((p) => p.route));
      const right = preds.filter((p, i) => p.route.indexOf(Math.max(...p.route)) === answers[i].route).length;
      lanes.replaceChildren(...ROUTES.map((r, k) => h('div', { class: 'stat' },
        h('div', { class: 'label' }, L(r)), h('div', { class: 'value' }, String(counts[k])),
        h('div', { class: 'row', style: { gap: '2px', marginTop: '6px' } }, preds.map((p, i) => (p.route.indexOf(Math.max(...p.route)) === k
          ? h('span', { title: fieldText(fields[i]), style: { width: '10px', height: '14px', borderRadius: '2px', background: answers[i].route === k ? 'var(--good)' : 'var(--bad)' } }) : null))))));
      summary.textContent = ctx.t('use.jev.routeStats', { r: right, n: preds.length });
    };
    const summary = h('div', { class: 'small' });
    el.append(h('p', { class: 'help' }, ctx.t('use.jev.routeHelp')),
      h('div', null, h('button', { class: 'btn btn-primary', type: 'button', onclick: run }, ctx.t('use.jev.next20'))), lanes, summary);
    run();
  },
};

const human = {
  id: 'human',
  icon: '🧑‍⚖️',
  title: lt('자동 처리 vs 사람 검토', 'Automate or ask a person'),
  intro: lt('모델이 충분히 확신할 때만 자동으로 처리하고, 애매한 건은 사람에게 넘깁니다. 확률이 정직해야(보정) 이 방식이 통합니다.', 'Automates only when the model is confident enough and hands borderline cases to a person — which only works if its probabilities are honest (calibrated).'),
  realWorld: lt('TypeSafe AI가 Jev를 소개하며 강조한 쓰임(가드레일·검증)이 바로 이것입니다: 보정된 확신도로 자동화 범위를 정합니다. 실제 Jev와 이 축소판은 무관합니다.', 'This is the use TypeSafe AI emphasized when introducing Jev (guardrails, verification): calibrated confidence decides what can be automated. This small model is unrelated to the real Jev.'),
  pipeline: PIPE(lt('승인할까요?', 'approve?'), { ko: '확신도 기준', en: 'Confidence gate', dko: '확신도(max(p, 1−p))가 기준 이상이면 자동 처리, 아니면 사람 검토 대기열로 보냅니다.', den: 'If confidence (max(p, 1−p)) is above the gate, automate; otherwise queue for a person.' }),
  render(el, ctx, flow) {
    const s = ctx.session;
    const box = chartBox();
    const stats = h('div', { class: 'stats' });
    let th = 0.8;
    const tv = h('span', { class: 'val' }, '80%');
    let data = null;
    const draw = () => {
      flow.run();
      const r = autoSplit(data.pYes, data.yes, th);
      stats.replaceChildren(
        h('div', { class: 'stat' }, h('div', { class: 'label' }, ctx.t('use.jev.autoRate')), h('div', { class: 'value' }, pct(r.autoRate))),
        h('div', { class: 'stat' }, h('div', { class: 'label' }, ctx.t('use.jev.errRate')), h('div', { class: 'value' }, pct(r.errorRate))),
        h('div', { class: 'stat' }, h('div', { class: 'label' }, ctx.t('use.jev.promised')), h('div', { class: 'value' }, `≤ ${pct(1 - th)}`), h('div', { class: 'sub' }, ctx.t('use.jev.promisedSub'))));
      chart.update(curves, { markX: Math.round(th * 20) });
    };
    el.append(h('p', { class: 'help' }, ctx.t('use.jev.humanHelp')),
      h('div', { class: 'control', style: { maxWidth: '340px' } }, h('label', null, h('span', null, ctx.t('use.jev.gate')), tv),
        h('input', { type: 'range', min: 0.5, max: 0.99, step: 0.01, value: th, oninput: (e) => { th = Number(e.target.value); tv.textContent = pct(th); draw(); } })),
      stats, box.el, h('p', { class: 'small muted' }, ctx.t('use.jev.curveHelp')));
    const chart = lineChart(box.canvas, { xLabel: ctx.t('use.jev.gateAxis'), xStart: 10, height: 200, yMin: 0, yMax: 1 });
    let curves = [];
    s.calibrationOn('test').then((r) => {
      data = r;
      const ths = Array.from({ length: 10 }, (_, i) => 0.5 + i * 0.05);
      curves = [
        { name: ctx.t('use.jev.autoRate'), values: ths.map((x) => autoSplit(r.pYes, r.yes, x).autoRate), color: '--series-1' },
        { name: ctx.t('use.jev.errRate'), values: ths.map((x) => autoSplit(r.pYes, r.yes, x).errorRate), color: '--series-2' },
        { name: ctx.t('use.jev.promised'), values: ths.map((x) => 1 - x), color: '--muted', dashed: true },
      ];
      draw();
    });
  },
};

export const useCases = [desk, routing, human];

export const USE_TEXT = {
  ko: {
    'use.jev.deskHelp': '시험용 신청서들에 대해 승인 확률을 계산했습니다. 기준선을 움직여 승인과 거절이 어떻게 바뀌는지 보세요.',
    'use.jev.threshold': '승인 기준선',
    'use.jev.deskStats': '{n}건 중 — 좋은 고객 승인 {a}건 · 위험한 고객 승인 {b}건(손실) · 좋은 고객 거절 {c}건(기회 손실)',
    'use.jev.routeHelp': '새 신청서 20건이 들어왔습니다. 막대 하나가 한 건이며, 초록은 올바른 창구, 빨강은 잘못 배정된 건입니다.',
    'use.jev.next20': '새 신청서 20건',
    'use.jev.routeStats': '20건 중 {r}건을 올바른 창구로 보냈습니다.',
    'use.jev.humanHelp': '기준을 높이면 자동 처리 비율은 줄지만, 자동으로 처리한 건의 오류율도 줄어야 합니다. 모델이 잘 보정되어 있다면 오류율이 약속선(1 − 기준) 아래에 머뭅니다.',
    'use.jev.gate': '자동 처리 확신도 기준',
    'use.jev.gateAxis': '기준 (×5%)',
    'use.jev.autoRate': '자동 처리 비율',
    'use.jev.errRate': '자동 처리 중 오류율',
    'use.jev.promised': '약속한 최대 오류율',
    'use.jev.promisedSub': '확률이 정직하다면',
    'use.jev.curveHelp': '가로: 기준 50% → 95%. 주황(실제 오류율)이 회색 점선(약속) 위로 올라가면 모델이 실제보다 자신만만한 것입니다.',
  },
  en: {
    'use.jev.deskHelp': 'Approval probabilities for the test applications. Move the threshold and watch approvals and declines change.',
    'use.jev.threshold': 'Approval threshold',
    'use.jev.deskStats': 'Of {n}: good customers approved {a} · risky customers approved {b} (losses) · good customers declined {c} (missed business)',
    'use.jev.routeHelp': 'Twenty new applications arrived. Each bar is one; green went to the right queue, red to the wrong one.',
    'use.jev.next20': '20 new applications',
    'use.jev.routeStats': '{r} of 20 went to the right queue.',
    'use.jev.humanHelp': 'Raising the gate automates fewer cases, and the error rate among automated ones should drop. If the model is calibrated, the error rate stays under the promise line (1 − gate).',
    'use.jev.gate': 'Confidence needed to automate',
    'use.jev.gateAxis': 'Gate (×5%)',
    'use.jev.autoRate': 'Automated share',
    'use.jev.errRate': 'Error rate when automated',
    'use.jev.promised': 'Promised max error',
    'use.jev.promisedSub': 'if probabilities are honest',
    'use.jev.curveHelp': 'Across: gate 50% → 95%. If orange (actual error) rises above the grey dashed line (promise), the model is more confident than it should be.',
  },
};
