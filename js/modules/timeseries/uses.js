// "Use" tab apps for the forecaster: decisions made from predictions of the future.
import { h, fmt } from '../../ui/dom.js';
import { lineChart, chartBox } from '../../ui/chart.js';
import { step, lt } from '../../ui/useView.js';

export function orderQuantity(forecast, stock, safety) {
  const need = forecast.reduce((a, b) => a + b, 0) + safety - stock;
  return Math.max(0, Math.ceil(need));
}

export const hoursAbove = (xs, limit) => xs.map((v, i) => (v > limit ? i : -1)).filter((i) => i >= 0);

export const outsideBand = (actual, predicted, band) => actual.map((a, i) => (Math.abs(a - predicted[i]) > band ? i : -1)).filter((i) => i >= 0);

// The test range of the series, plus a forecast of `steps` values after its start.
function forecastFrom(ctx, steps) {
  const { series, test } = ctx.data;
  const start = test.t[0];
  const history = series.slice(start - ctx.hp.window, start);
  return { history, future: ctx.session.forecast(history, steps), actual: series.slice(start, start + steps) };
}

const slider = (label, min, max, stepv, value, oninput) => {
  const val = h('span', { class: 'val' }, String(value));
  return h('div', { class: 'control' }, h('label', null, h('span', null, label), val),
    h('input', { type: 'range', min, max, step: stepv, value, oninput: (e) => { val.textContent = e.target.value; oninput(Number(e.target.value)); } }));
};

const PIPE = (input, decide, out) => [
  step('input', input.ko, input.en, '지금까지 기록된 값들입니다.', 'The values recorded so far.'),
  step('pre', '최근 창 잘라내기', 'Take recent window', `가장 최근 기록을 모델이 보는 창 크기만큼 잘라 정규화합니다.`, 'Cuts the most recent stretch (the model\'s window size) and normalizes it.'),
  step('model', '앞날 예측', 'Forecast ahead', '학습한 시계열 모델이 다음 값을 예측하고, 그 예측을 다시 넣어 24시간 앞까지 내다봅니다.', 'Your trained model predicts the next value, feeding its predictions back in to look 24 hours ahead.'),
  step('post', decide.ko, decide.en, decide.dko, decide.den),
  step('output', out.ko, out.en, '결정이 화면에 나타납니다.', 'The decision appears on screen.'),
];

const inventory = {
  id: 'inventory',
  icon: '🏪',
  title: lt('편의점 자동 발주', 'Store auto-ordering'),
  intro: lt('내일 팔릴 양을 예측해서 오늘 몇 개를 주문할지 정합니다.', 'Predicts tomorrow\'s sales to decide how much to order today.'),
  realWorld: lt('실제 유통사는 날씨·요일·행사 같은 정보를 함께 넣고, 품목 수만 개를 동시에 예측합니다. 너무 많이 주문하면 폐기, 적게 하면 품절이라 안전 재고를 함께 정합니다.', 'Real retailers add weather, weekdays and promotions, forecasting tens of thousands of items at once, and balance waste against stock-outs with safety stock.'),
  pipeline: PIPE(lt('판매 기록', 'Sales history'), { ko: '주문량 계산', en: 'Order amount', dko: '예측 판매량 합계 + 안전 재고 - 현재 재고 = 주문량 (0보다 작으면 주문 안 함)', den: 'Forecast total + safety stock − current stock = order (none if below zero)' }, lt('발주서', 'Purchase order')),
  render(el, ctx, flow) {
    const toUnits = (v) => Math.max(0, 10 + 3 * v); // sales per hour
    const { history, future, actual } = forecastFrom(ctx, 24);
    let stock = 120;
    let safety = 20;
    const order = h('div', { class: 'big-result' });
    const box = chartBox();
    const draw = () => {
      flow.run();
      const fc = future.map(toUnits);
      const q = orderQuantity(fc, stock, safety);
      order.replaceChildren(`📝 ${ctx.t('use.ts.order')}: `, h('span', { class: q ? 'tag-warn' : 'tag-good' }, `${q} ${ctx.t('use.ts.units')}`));
      const pad = (arr, n) => [...new Array(n).fill(NaN), ...arr];
      chart.update([
        { name: ctx.t('use.ts.past'), values: [...history.map(toUnits), ...new Array(24).fill(NaN)], color: '--series-1' },
        { name: ctx.t('use.ts.forecast'), values: pad(fc, history.length), color: '--series-2' },
        { name: ctx.t('use.ts.actualLater'), values: pad(actual.map(toUnits), history.length), color: '--muted', dashed: true },
      ]);
      sum.textContent = ctx.t('use.ts.sum', { f: fmt(fc.reduce((a, b) => a + b, 0)), a: fmt(actual.map(toUnits).reduce((a, b) => a + b, 0)) });
    };
    const sum = h('p', { class: 'small muted' });
    el.append(h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        slider(ctx.t('use.ts.stock'), 0, 400, 10, stock, (v) => { stock = v; draw(); }),
        slider(ctx.t('use.ts.safety'), 0, 100, 5, safety, (v) => { safety = v; draw(); }),
        order, sum),
      h('div', null, box.el)));
    const chart = lineChart(box.canvas, { xLabel: ctx.t('use.ts.hour'), xStart: -history.length, height: 220 });
    draw();
  },
};

const power = {
  id: 'power',
  icon: '⚡',
  title: lt('전력 수요 대비', 'Power demand planner'),
  intro: lt('내일 전기 사용량이 몰리는 시간을 예측해 발전기를 미리 켭니다.', 'Predicts when tomorrow\'s electricity use peaks so extra generators start in time.'),
  realWorld: lt('전력 회사는 기온 예보와 과거 수년 치 기록을 함께 써서 15분 단위로 예측합니다. 발전기는 켜는 데 시간이 걸려 예측이 꼭 필요합니다.', 'Utilities forecast in 15-minute steps using temperature forecasts and years of history. Generators take time to start, so forecasting is essential.'),
  pipeline: PIPE(lt('전력 사용 기록', 'Usage history'), { ko: '용량 초과 시간 찾기', en: 'Find over-capacity hours', dko: '예측 사용량이 기본 발전 용량을 넘는 시간을 찾고, 그 1시간 전에 보조 발전기를 켜도록 계획합니다.', den: 'Finds hours where forecast use exceeds base capacity and schedules backup generators one hour earlier.' }, lt('가동 계획표', 'Start-up plan')),
  render(el, ctx, flow) {
    const toKw = (v) => 300 + 60 * v;
    const { future, actual } = forecastFrom(ctx, 24);
    const fc = future.map(toKw);
    // Start at the forecast's 75th percentile so a few hours need backup.
    let cap = Math.round([...fc].sort((a, b) => a - b)[Math.floor(fc.length * 0.75)] / 5) * 5;
    const plan = h('div', { class: 'stack', style: { gap: '4px' } });
    const box = chartBox();
    const draw = () => {
      flow.run();
      const hrs = hoursAbove(fc, cap);
      chart.update([
        { name: ctx.t('use.ts.forecast'), values: fc, color: '--series-2' },
        { name: ctx.t('use.ts.actualLater'), values: actual.map(toKw), color: '--muted', dashed: true },
        { name: ctx.t('use.ts.capacity'), values: fc.map(() => cap), color: '--bad', dashed: true },
      ]);
      plan.replaceChildren(hrs.length
        ? h('div', null, h('b', null, ctx.t('use.ts.startAt', { n: hrs.length })), h('div', { class: 'row', style: { gap: '4px', marginTop: '6px' } }, hrs.map((i) => h('span', { class: 'chip' }, `⚙️ ${String(Math.max(0, i - 1)).padStart(2, '0')}:00`))))
        : h('b', { class: 'tag-good' }, ctx.t('use.ts.noStart')));
    };
    el.append(h('div', { class: 'grid-2' },
      h('div', { class: 'stack' }, slider(ctx.t('use.ts.cap'), 250, 420, 5, cap, (v) => { cap = v; draw(); }), plan),
      h('div', null, box.el)));
    const chart = lineChart(box.canvas, { xLabel: ctx.t('use.ts.hour'), xStart: 0, height: 220 });
    draw();
  },
};

const traffic = {
  id: 'traffic',
  icon: '🖥️',
  title: lt('서버 트래픽 이상 알림', 'Server traffic alerts'),
  intro: lt('예측과 너무 다른 접속량이 들어오면 담당자에게 알림을 보냅니다.', 'Alerts the on-call engineer when traffic is far from what was predicted.'),
  realWorld: lt('클라우드 모니터링 서비스는 수천 개 지표마다 예측 범위를 만들고, 벗어나면 알림을 보냅니다. 범위를 너무 좁히면 거짓 알림이 쏟아집니다.', 'Cloud monitoring builds an expected band for thousands of metrics and alerts on breaches. Make the band too tight and false alarms pour in.'),
  pipeline: [
    step('input', '실시간 접속량', 'Live traffic', '1시간마다 접속량이 들어옵니다.', 'Traffic arrives every hour.'),
    step('pre', '최근 창 잘라내기', 'Take recent window', '직전 기록을 모델 창 크기만큼 잘라 냅니다.', 'Cuts the latest window of history.'),
    step('model', '이번 값 예측', 'Predict this hour', '학습한 모델이 "이번 시간엔 이 정도가 정상"이라는 값을 예측합니다.', 'Your model predicts what this hour should normally look like.'),
    step('post', '허용 범위 비교', 'Compare to band', '실제 값이 예측 ± 허용 범위를 벗어나면 이상으로 봅니다.', 'If the actual value is outside prediction ± band, it is an anomaly.'),
    step('output', '알림', 'Alert', '이상이면 알림 목록에 추가합니다.', 'Anomalies are added to the alert list.'),
  ],
  render(el, ctx, flow) {
    const s = ctx.session;
    const { test } = ctx.data;
    const toReq = (v) => Math.round(1000 + 200 * v);
    const residuals = ctx.data.val.y.map((y, i) => y - s.predictWindows([ctx.data.val.x[i]])[0]);
    const sigma = Math.sqrt(residuals.reduce((a, r) => a + r * r, 0) / residuals.length);
    let k = 3;
    let pos = 0;
    const injected = new Set();
    const actual = [];
    const predicted = [];
    const log = h('div', { class: 'stack', style: { gap: '4px', maxHeight: '180px', overflowY: 'auto' } });
    const box = chartBox();
    const kVal = h('span', { class: 'val' }, '3σ');
    const tick = (n) => {
      for (let j = 0; j < n && pos < test.x.length; j++, pos++) {
        let a = test.y[pos];
        if (injected.has(pos)) a += 6 * sigma * (pos % 2 ? 1 : -1);
        actual.push(a);
        predicted.push(s.predictWindows([test.x[pos]])[0]);
      }
      flow.run();
      redraw();
    };
    const redraw = () => {
      const band = k * sigma;
      const bad = outsideBand(actual, predicted, band);
      chart.update([
        { name: ctx.t('use.ts.actual'), values: actual.map(toReq), color: '--series-1' },
        { name: ctx.t('use.ts.expected'), values: predicted.map(toReq), color: '--series-2' },
        { name: ctx.t('use.ts.upper'), values: predicted.map((p) => toReq(p + band)), color: '--muted', dashed: true },
        { name: ctx.t('use.ts.lower'), values: predicted.map((p) => toReq(p - band)), color: '--muted', dashed: true },
      ]);
      log.replaceChildren(...(bad.length ? bad.slice().reverse().map((i) => h('div', { class: 'mail-row' }, h('span', null, injected.has(i) ? '🚨' : '⚠️'),
        h('span', null, ctx.t(injected.has(i) ? 'use.ts.alertReal' : 'use.ts.alertFalse', { h: i, a: toReq(actual[i]), p: toReq(predicted[i]) })), h('span'))) : [h('span', { class: 'small muted' }, ctx.t('use.ts.noAlerts'))]));
    };
    el.append(h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        h('div', { class: 'row' },
          h('button', { class: 'btn btn-primary', type: 'button', onclick: () => tick(1) }, ctx.t('use.ts.nextHour')),
          h('button', { class: 'btn', type: 'button', onclick: () => tick(12) }, ctx.t('use.ts.next12')),
          h('button', { class: 'btn btn-danger', type: 'button', onclick: () => { injected.add(pos + 1); tick(2); } }, ctx.t('use.ts.inject'))),
        h('div', { class: 'control' }, h('label', null, h('span', null, ctx.t('use.ts.band')), kVal),
          h('input', { type: 'range', min: 0.5, max: 5, step: 0.5, value: 3, oninput: (e) => { k = Number(e.target.value); kVal.textContent = `${k}σ`; flow.pulse(3); redraw(); } })),
        h('b', null, ctx.t('use.ts.alerts')), log),
      h('div', null, box.el)));
    const chart = lineChart(box.canvas, { xLabel: ctx.t('use.ts.hour'), xStart: 0, height: 220 });
    tick(24);
  },
};

export const useCases = [inventory, power, traffic];

export const USE_TEXT = {
  ko: {
    'use.ts.order': '오늘 주문할 양',
    'use.ts.units': '개',
    'use.ts.past': '지난 판매',
    'use.ts.forecast': '모델 예측',
    'use.ts.actualLater': '실제(나중에 알게 됨)',
    'use.ts.sum': '예측한 24시간 판매량 {f}개 · 실제 {a}개',
    'use.ts.stock': '현재 재고',
    'use.ts.safety': '안전 재고',
    'use.ts.hour': '시간',
    'use.ts.capacity': '기본 발전 용량',
    'use.ts.cap': '기본 발전 용량 (kW)',
    'use.ts.startAt': '보조 발전기를 {n}번 미리 켭니다',
    'use.ts.noStart': '내일은 기본 발전만으로 충분합니다',
    'use.ts.actual': '실제 접속량',
    'use.ts.expected': '예측',
    'use.ts.upper': '허용 상한',
    'use.ts.lower': '허용 하한',
    'use.ts.nextHour': '1시간 진행',
    'use.ts.next12': '12시간 진행',
    'use.ts.inject': '장애 일으키기',
    'use.ts.band': '허용 범위 (예측 오차의 몇 배)',
    'use.ts.alerts': '알림 목록',
    'use.ts.noAlerts': '아직 알림이 없습니다',
    'use.ts.alertReal': '{h}시: 접속량 {a} (예상 {p}) — 실제 장애를 잡았습니다',
    'use.ts.alertFalse': '{h}시: 접속량 {a} (예상 {p}) — 거짓 알림',
  },
  en: {
    'use.ts.order': 'Order today',
    'use.ts.units': 'units',
    'use.ts.past': 'Past sales',
    'use.ts.forecast': 'Model forecast',
    'use.ts.actualLater': 'Actual (known later)',
    'use.ts.sum': 'Forecast 24h sales {f} · actual {a}',
    'use.ts.stock': 'Current stock',
    'use.ts.safety': 'Safety stock',
    'use.ts.hour': 'Hour',
    'use.ts.capacity': 'Base capacity',
    'use.ts.cap': 'Base capacity (kW)',
    'use.ts.startAt': 'Start backup generators {n} time(s) in advance',
    'use.ts.noStart': 'Base capacity is enough tomorrow',
    'use.ts.actual': 'Actual traffic',
    'use.ts.expected': 'Forecast',
    'use.ts.upper': 'Upper limit',
    'use.ts.lower': 'Lower limit',
    'use.ts.nextHour': 'Advance 1 hour',
    'use.ts.next12': 'Advance 12 hours',
    'use.ts.inject': 'Cause an outage',
    'use.ts.band': 'Allowed band (× typical forecast error)',
    'use.ts.alerts': 'Alerts',
    'use.ts.noAlerts': 'No alerts yet',
    'use.ts.alertReal': 'Hour {h}: traffic {a} (expected {p}) — caught a real outage',
    'use.ts.alertFalse': 'Hour {h}: traffic {a} (expected {p}) — false alarm',
  },
};
