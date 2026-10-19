// Module 9 — Jev-style decision model: state + typed questions → calibrated answers.
import { h, pct, fmt, probBars } from '../../ui/dom.js';
import { reliabilityChart, lineChart, chartBox, heatmap } from '../../ui/chart.js';
import { L } from '../../i18n.js';
import { FIELDS, ROUTES } from './data.js';
import { makeData, createSession, architecture, baseRateLoss, TOKENS } from './model.js';
import { useCases, USE_TEXT } from './uses.js';

const valueText = (d, v) => (d.type === 'num' ? `${v} ${L(d.unit)}` : L(d.options[v]));

export default {
  id: 'jev',
  title: { ko: '결정 모델 (Jev 스타일)', en: 'Decision model (Jev-style)' },
  summary: {
    ko: '글을 쓰지 않고, 정보와 "형식이 정해진 질문"을 받아 확률·선택·점수로 답하는 Transformer입니다. 확률을 믿을 수 있는지(보정)를 검증합니다.',
    en: 'A Transformer that writes no text: it takes information plus typed questions and answers with a probability, a choice and a score — and we check whether its probabilities can be trusted (calibration).',
  },
  archHint: 'Fields + [Q] tokens → Encoder → yes/no · choice · score heads',
  synthetic: true,
  archNote: 'm.disclaimer',
  // Eight tokens is tiny: WebGL's per-operation overhead dominates, so the CPU backend is faster here.
  preferBackend: 'cpu',
  defaults: { seed: 1, n: 1400, dModel: 32, heads: 4, blocks: 1, lr: 0.003, epochs: 20, batchSize: 64 },
  dataKeys: ['seed', 'n'],
  controls: [
    { key: 'blocks', type: 'select', label: 'c.blocks', where: 'arch', options: [1, 2, 3].map((v) => ({ value: v, label: String(v) })) },
    { key: 'heads', type: 'select', label: 'c.heads', where: 'arch', options: [1, 2, 4].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dModel', type: 'select', label: 'c.dModel', where: 'arch', options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    { key: 'n', type: 'range', min: 100, max: 4000, step: 100, label: 'c.n', where: 'data' },
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 250, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 16, max: 256, step: 16, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { n: 150, epochs: 200, dModel: 64 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 2 }, expect: 'diverged' },
  ],
  metrics: [
    { key: 'acc', label: 'm.acc', format: 'pct' },
    { key: 'routeAcc', label: 'm.routeAcc', format: 'pct' },
    { key: 'ece', label: 'metric.ece', format: 'pct', dashed: true },
  ],
  testMetrics: [
    { key: 'acc', label: 'm.acc', format: 'pct', sub: 'm.accSub' },
    { key: 'ece', label: 'metric.ece', format: 'pct', sub: 'm.eceSub' },
    { key: 'routeAcc', label: 'm.routeAcc', format: 'pct' },
    { key: 'scoreMae', label: 'm.scoreMae', format: 'num' },
  ],
  baselineLoss: (hp, data) => baseRateLoss(data),
  baselineLabel: 'm.baseline',
  architecture,
  makeData,
  createSession,
  useCases,

  renderArchIntro(el, ctx) {
    const box = (title, body, cls) => h('div', { class: `stat ${cls || ''}`, style: { flex: '1 1 160px' } }, h('div', { class: 'label' }, title), h('div', { class: 'small' }, body));
    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('m.flowTitle')),
      h('div', { class: 'row', style: { gap: '8px', alignItems: 'stretch' } },
        box(ctx.t('m.flowState'), ctx.t('m.flowStateBody')), h('span', { class: 'pipe-arrow' }, '+'),
        box(ctx.t('m.flowQ'), ctx.t('m.flowQBody')), h('span', { class: 'pipe-arrow' }, '→'),
        box(ctx.t('m.flowModel'), ctx.t('m.flowModelBody')), h('span', { class: 'pipe-arrow' }, '→'),
        box(ctx.t('m.flowOut'), ctx.t('m.flowOutBody'))),
      h('p', { class: 'help', style: { marginTop: '10px', marginBottom: 0 } }, ctx.t('m.flowVsLlm'))));
  },

  renderData(el, data, ctx) {
    const rows = data.train.fields.slice(0, 8).map((f, i) => ({ f, a: data.train.answers[i] }));
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      h('div', { style: { overflowX: 'auto' } }, h('table', { class: 'small', style: { borderCollapse: 'collapse', width: '100%' } },
        h('thead', null, h('tr', null, [...FIELDS.map((d) => L(d.label)), ctx.t('m.colApprove'), ctx.t('m.colRoute'), ctx.t('m.colRisk')].map((t) => h('th', { style: { textAlign: 'left', padding: '4px 8px', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap' } }, t)))),
        h('tbody', null, rows.map(({ f, a }) => h('tr', null,
          ...FIELDS.map((d) => h('td', { style: { padding: '4px 8px', whiteSpace: 'nowrap' } }, valueText(d, f[d.key]))),
          h('td', { style: { padding: '4px 8px' } }, a.approve ? '✅' : '⛔'),
          h('td', { style: { padding: '4px 8px', whiteSpace: 'nowrap' } }, L(ROUTES[a.route])),
          h('td', { class: 'mono', style: { padding: '4px 8px' } }, a.risk.toFixed(2))))))),
      h('p', { class: 'help', style: { marginTop: '10px' } }, ctx.t('m.noiseNote')));
  },

  renderTrainViz(el, ctx) {
    const rel = h('canvas');
    const eceBox = chartBox();
    el.append(h('div', { class: 'grid-2' },
      h('div', null, h('h3', { class: 'small' }, ctx.t('m.relTitle')), h('p', { class: 'help' }, ctx.t('m.relHelp')), h('div', { class: 'chart-wrap' }, rel)),
      h('div', null, h('h3', { class: 'small' }, ctx.t('m.eceTitle')), h('p', { class: 'help' }, ctx.t('m.eceHelp')), eceBox.el)));
    const relChart = reliabilityChart(rel, []);
    const eceChart = lineChart(eceBox.canvas, { xLabel: ctx.t('train.epoch'), height: 200, yMin: 0 });
    return {
      update() {
        const s = ctx.session;
        relChart.update(s ? s.calibration() : []);
        const hist = ctx.runner.history;
        eceChart.update([{ name: ctx.t('metric.ece'), values: hist.map((x) => x.metrics.ece), color: '--series-2' },
          { name: ctx.t('m.goodLine'), values: hist.map(() => 0.05), color: '--good', dashed: true }]);
      },
    };
  },

  async diagnosePanel(el, ctx, history) {
    const last = history.at(-1).metrics;
    const cal = await ctx.session.calibrationOn('val');
    const findings = [];
    if (cal.ece < 0.05) findings.push({ level: 'good', key: 'm.wellCalibrated', vars: { e: pct(cal.ece) } });
    else {
      // Over-confident: claims stronger than reality, weighted over the bins.
      let over = 0;
      let n = 0;
      for (const b of cal.bins) if (b.count) { over += b.count * (Math.abs(b.avgConf - 0.5) - Math.abs(b.accuracy - 0.5)); n += b.count; }
      const key = over / n > 0 ? 'm.overconfident' : 'm.underconfident';
      findings.push({ level: cal.ece > 0.15 ? 'bad' : 'warn', key, vars: { e: pct(cal.ece) } });
    }
    if (last.acc >= 0.8 && cal.ece >= 0.1) findings.push({ level: 'warn', key: 'm.accurateButMiscal' });
    findings.push({ level: 'info', key: 'm.ceiling' });
    const rel = h('canvas');
    el.append(h('div', { class: 'card' },
      h('h3', null, ctx.t('m.relTitle')), h('p', { class: 'help' }, ctx.t('m.relDiagHelp')), h('div', { class: 'chart-wrap', style: { maxWidth: '520px' } }, rel),
      h('p', { class: 'small muted' }, ctx.t('m.binsLegend'))));
    reliabilityChart(rel, cal.bins);
    return findings;
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const form = { ...ctx.data.test.fields[0] };
    const answers = h('div', { class: 'stack' });
    const sens = h('div');
    const attn = h('canvas');
    const inputs = FIELDS.map((d) => {
      const val = h('span', { class: 'val' }, valueText(d, form[d.key]));
      const input = d.type === 'num'
        ? h('input', { type: 'range', min: d.min, max: d.max, step: 1, value: form[d.key] })
        : h('select', { class: 'text-input' }, d.options.map((o, i) => h('option', { value: i, selected: i === form[d.key] }, L(o))));
      input.addEventListener('input', () => { form[d.key] = Number(input.value); val.textContent = valueText(d, form[d.key]); update(); });
      return h('div', { class: 'control' }, h('label', null, h('span', null, L(d.label)), d.type === 'num' ? val : null), input);
    });
    const update = () => {
      const [p] = s.predict([form]);
      answers.replaceChildren(
        h('div', null, h('b', { class: 'small' }, ctx.t('m.qYes')), h('div', { class: 'big-result' }, `${p.approve >= 0.5 ? '✅' : '⛔'} ${pct(p.approve)}`)),
        h('div', null, h('b', { class: 'small' }, ctx.t('m.qChoice')), probBars(ROUTES.map(L), p.route)),
        h('div', null, h('b', { class: 'small' }, ctx.t('m.qScore')), h('div', { class: 'big-result' }, fmt(p.risk))));
      // Sensitivity: how much does the approval probability move if each field changes a little?
      const deltas = FIELDS.map((d) => {
        const f2 = { ...form };
        if (d.type === 'num') f2[d.key] = Math.min(d.max, form[d.key] + (d.max - d.min) * 0.1);
        else f2[d.key] = (form[d.key] + 1) % d.options.length;
        return s.predict([f2])[0].approve - p.approve;
      });
      sens.replaceChildren(h('div', { class: 'bars' }, FIELDS.map((d, i) => h('div', { class: 'bar-row' },
        h('span', { class: 'small' }, L(d.label)),
        h('div', { class: 'track', style: { position: 'relative' } }, h('div', { class: 'fill', style: { position: 'absolute', left: deltas[i] >= 0 ? '50%' : `${50 + deltas[i] * 50}%`, width: `${Math.abs(deltas[i]) * 50}%`, background: deltas[i] >= 0 ? 'var(--good)' : 'var(--bad)' } })),
        h('span', { class: 'num' }, `${deltas[i] >= 0 ? '+' : ''}${(deltas[i] * 100).toFixed(1)}`)))));
      const w = s.attention(form);
      const lastBlock = w[w.length - 1];
      const avg = lastBlock[0].map((row, i) => row.map((_, j) => lastBlock.reduce((a, head) => a + head[i][j], 0) / lastBlock.length));
      heatmap(attn, avg.slice(5), { rowLabels: ['Y/N', 'CH', 'SC'], colLabels: TOKENS.map((t) => L(t).slice(0, 3)), height: 120 });
    };
    el.append(h('div', { class: 'card' }, h('h3', null, ctx.t('test.try')), h('p', { class: 'help' }, ctx.t('m.tryHelp')),
      h('div', { class: 'grid-2' }, h('div', { class: 'stack' }, inputs), answers)),
    h('div', { class: 'grid-2' },
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.sensTitle')), h('p', { class: 'help' }, ctx.t('m.sensHelp')), sens),
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.attnTitle')), h('p', { class: 'help' }, ctx.t('m.attnHelp')), h('div', { class: 'chart-wrap' }, attn))));
    update();
  },

  text: {
    ko: {
      ...USE_TEXT.ko,
      'm.disclaimer': '이 모듈은 TypeSafe AI가 2026년 9월 공개한 결정 모델 Jev의 "아이디어"를 교육용으로 아주 작게 재현한 것입니다. 실제 Jev와는 코드·데이터·성능 모두 무관합니다.',
      'c.blocks': '블록 수',
      'c.heads': '어텐션 헤드 수',
      'c.dModel': '벡터 크기 (d_model)',
      'c.n': '신청서 개수',
      'p.overfit': '신청서 150건으로 200에폭 → 과적합',
      'p.diverge': '학습률 2 → 발산',
      'm.acc': '승인 예측 정확도',
      'm.accSub': '라벨의 10%는 잡음이라 최대 약 90%',
      'm.routeAcc': '창구 배정 정확도',
      'm.scoreMae': '위험도 점수 오차(MAE)',
      'm.eceSub': '5% 미만이면 확률을 믿을 만함',
      'm.baseline': '기준선: 항상 평균 비율로 답할 때',
      'm.flowTitle': '이 모델이 하는 일',
      'm.flowState': '상태 (State)',
      'm.flowStateBody': '신청서 항목 5개',
      'm.flowQ': '형식이 정해진 질문',
      'm.flowQBody': '예/아니오 · 셋 중 선택 · 점수',
      'm.flowModel': 'Transformer 인코더',
      'm.flowModelBody': '질문 토큰이 항목 토큰을 어텐션으로 살핌',
      'm.flowOut': '보정된 답',
      'm.flowOutBody': '확률 · 선택 확률 · 점수 (숫자)',
      'm.flowVsLlm': '챗봇형 언어 모델과 다른 점: 글을 한 글자씩 생성하지 않고 한 번의 계산으로 숫자 답을 냅니다. 그래서 빠르고, 프로그램이 바로 쓸 수 있으며, "80%라고 답했을 때 실제로 80% 맞는지(보정)"를 검사할 수 있습니다.',
      'm.dataIntro': '가상의 대출 신청서입니다. 숨은 규칙(소득↑·근속↑는 좋고, 부채↑·연체↑는 나쁨, 목적별 차이)과 잡음으로 정답을 만들었습니다. 질문 세 개에 대한 정답이 함께 있습니다.',
      'm.colApprove': '승인?',
      'm.colRoute': '창구',
      'm.colRisk': '위험도',
      'm.noiseNote': '정답의 10%는 일부러 뒤집어 두었습니다(사람 심사도 늘 일관되지는 않으니까요). 그래서 어떤 모델도 100% 확신하면 안 됩니다 — 이것이 "보정"을 보는 이유입니다.',
      'm.relTitle': '보정 그래프 (Reliability diagram)',
      'm.relHelp': '가로: 모델이 말한 승인 확률, 세로: 그 구간에서 실제로 승인된 비율. 막대가 점선(대각선)에 붙을수록 "말한 만큼 맞는" 정직한 모델입니다.',
      'm.eceTitle': '보정 오차(ECE) 변화',
      'm.eceHelp': '말한 확률과 실제 비율의 평균 차이입니다. 낮을수록 좋고, 5% 아래면 믿을 만합니다.',
      'm.goodLine': '5% 기준선',
      'm.wellCalibrated': '보정 오차가 {e}로 작습니다. "70%"라고 하면 실제로 약 70% 승인됩니다 — 확률을 믿을 수 있습니다.',
      'm.overconfident': '보정 오차가 {e}입니다. 모델이 실제보다 자신만만합니다(90%라고 했는데 실제는 더 낮음).',
      'm.underconfident': '보정 오차가 {e}입니다. 모델이 실제보다 소심합니다(확신해도 될 때도 50% 근처로 답함).',
      'm.accurateButMiscal': '정확도는 높지만 보정은 나쁩니다. 정답을 많이 맞혀도 확률 숫자 자체는 믿으면 안 되는 상태입니다.',
      'm.ceiling': '라벨의 10%가 잡음이므로 정확도는 약 90%가 최대입니다. 그보다 훨씬 높다면 오히려 이상한 것입니다.',
      'm.relDiagHelp': '빨간 선은 각 구간에서 "말한 확률"과 "실제 비율"의 차이입니다. 짧을수록 좋습니다. 막대가 진할수록 그 구간에 사례가 많습니다.',
      'm.binsLegend': '점선 = 완벽한 보정 · 파란 막대 = 실제 비율 · 빨간 선 = 차이',
      'm.tryHelp': '왼쪽 항목을 바꾸면 세 질문의 답이 즉시 바뀝니다.',
      'm.qYes': '[예/아니오] 승인할까요?',
      'm.qChoice': '[선택] 어느 창구로?',
      'm.qScore': '[점수] 위험도는? (0~1)',
      'm.sensTitle': '무엇이 결정을 바꾸나 (민감도)',
      'm.sensHelp': '각 항목을 조금(범위의 10%)만 올렸을 때 승인 확률이 몇 %p 변하는지입니다.',
      'm.attnTitle': '질문 토큰이 본 곳 (마지막 블록)',
      'm.attnHelp': '행: 질문 토큰(예/아니오, 선택, 점수) · 열: 참고한 토큰. 진할수록 많이 참고했습니다.',
    },
    en: {
      ...USE_TEXT.en,
      'm.disclaimer': 'This module is a tiny educational reproduction of the idea behind Jev, the decision model TypeSafe AI released in September 2026. It shares no code, data or performance with the real Jev.',
      'c.blocks': 'Blocks',
      'c.heads': 'Attention heads',
      'c.dModel': 'Vector size (d_model)',
      'c.n': 'Number of applications',
      'p.overfit': '150 applications for 200 epochs → overfitting',
      'p.diverge': 'Learning rate 2 → diverges',
      'm.acc': 'Approval accuracy',
      'm.accSub': '10% of labels are noise, so ~90% is the ceiling',
      'm.routeAcc': 'Queue accuracy',
      'm.scoreMae': 'Risk score error (MAE)',
      'm.eceSub': 'Under 5%: probabilities can be trusted',
      'm.baseline': 'Baseline: always answer with the average rates',
      'm.flowTitle': 'What this model does',
      'm.flowState': 'State',
      'm.flowStateBody': '5 application fields',
      'm.flowQ': 'Typed questions',
      'm.flowQBody': 'yes/no · pick one of three · score',
      'm.flowModel': 'Transformer encoder',
      'm.flowModelBody': 'question tokens attend over field tokens',
      'm.flowOut': 'Calibrated answers',
      'm.flowOutBody': 'probability · choice probabilities · score (numbers)',
      'm.flowVsLlm': 'Unlike a chatbot language model, it does not generate text token by token — one pass gives numeric answers. That makes it fast, directly usable by software, and checkable: when it says 80%, is it right 80% of the time (calibration)?',
      'm.dataIntro': 'Fictional loan applications. The answers come from a hidden rule (higher income and years good; more debt and missed payments bad; purpose matters) plus noise. Each has answers to the three questions.',
      'm.colApprove': 'Approve?',
      'm.colRoute': 'Queue',
      'm.colRisk': 'Risk',
      'm.noiseNote': '10% of the answers are flipped on purpose (human reviewers are not perfectly consistent either). So no model should ever be 100% sure — which is why we check calibration.',
      'm.relTitle': 'Reliability diagram',
      'm.relHelp': 'Across: the approval probability the model stated; up: how often those cases were really approved. The closer the bars hug the dashed diagonal, the more honest the model.',
      'm.eceTitle': 'Calibration error (ECE) over training',
      'm.eceHelp': 'The average gap between stated probability and reality. Lower is better; under 5% is trustworthy.',
      'm.goodLine': '5% line',
      'm.wellCalibrated': 'Calibration error is only {e}. When it says "70%", about 70% really are approved — the probabilities can be trusted.',
      'm.overconfident': 'Calibration error is {e}. The model is over-confident (says 90% when reality is lower).',
      'm.underconfident': 'Calibration error is {e}. The model is under-confident (hovers near 50% even when it could be sure).',
      'm.accurateButMiscal': 'Accuracy is high but calibration is poor: it gets many answers right, yet its probability numbers should not be trusted.',
      'm.ceiling': 'Because 10% of labels are noise, about 90% accuracy is the ceiling. Much higher would be suspicious.',
      'm.relDiagHelp': 'Red lines show, per bin, the gap between stated probability and reality — shorter is better. Darker bars hold more cases.',
      'm.binsLegend': 'Dashed = perfect calibration · blue bars = actual rate · red = gap',
      'm.tryHelp': 'Change the fields on the left; the three answers update instantly.',
      'm.qYes': '[yes/no] Approve?',
      'm.qChoice': '[choice] Which queue?',
      'm.qScore': '[score] Risk? (0–1)',
      'm.sensTitle': 'What moves the decision (sensitivity)',
      'm.sensHelp': 'How many percentage points the approval probability moves when each field goes up a little (10% of its range).',
      'm.attnTitle': 'Where the question tokens looked (last block)',
      'm.attnHelp': 'Rows: question tokens (yes/no, choice, score) · columns: tokens attended to. Darker means more attention.',
    },
  },
};
