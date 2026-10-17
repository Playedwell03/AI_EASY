// The 5-tab page every module shares: Architecture → Data → Train → Diagnose → Test.
import { h, fmt, pct } from './dom.js';
import { t, L, onLangChange } from '../i18n.js';
import { TABS, moduleHref } from '../router.js';
import { createRunner } from '../core/runner.js';
import { diagnoseCurves } from '../core/diagnose.js';
import { lineChart, chartBox } from './chart.js';
import { renderArch } from './archView.js';
import { renderUseTab } from './useView.js';

let mounted = null; // { def, hp, data, runner, tab, el, offLang, activePreset, testCache, archSel }

// For debugging in the browser console.
if (typeof window !== 'undefined') window.__aiEasyMounted = () => mounted;

export function unmountModule() {
  if (!mounted) return;
  mounted.runner.reset();
  mounted.offLang();
  mounted.cleanup?.();
  mounted = null;
}

export function mountModule(el, def, tab) {
  if (!mounted || mounted.def !== def) {
    unmountModule();
    const hp = defaultsFor(def);
    mounted = { def, hp, data: def.makeData(hp), el, tab, activePreset: null, testCache: null, archSel: 0 };
    mounted.runner = makeRunner(mounted);
    mounted.ready = ensureBackend(def);
    mounted.offLang = onLangChange(() => render());
  }
  mounted.el = el;
  mounted.tab = tab;
  render();
}

// Small recurrent models run faster on the CPU backend than on WebGL (per-op overhead dominates),
// so a module may declare preferBackend. Switching happens before any session exists.
async function ensureBackend(def) {
  const tf = window.tf;
  if (!tf) return;
  const target = def.preferBackend || window.__aiEasyBackend || tf.getBackend();
  if (tf.getBackend() !== target) {
    await tf.setBackend(target);
    window.dispatchEvent(new Event('aieasy:backend'));
  }
}

function startTraining() {
  const m = mounted;
  return m.ready.then(() => { if (mounted === m) return m.runner.start(); });
}

// Without WebGL, modules may ask for lighter defaults so training still finishes in about a minute.
function defaultsFor(def) {
  const cpu = (window.__aiEasyBackend || window.tf?.getBackend()) === 'cpu';
  return { ...def.defaults, ...(cpu ? def.cpuDefaults : null) };
}

function makeRunner(m) {
  const r = createRunner({
    createSession: () => m.def.createSession(window.tf, m.hp, m.data),
    epochs: m.hp.epochs,
    nextFrame: () => window.tf.nextFrame(),
  });
  r.on('state', () => { m.testCache = null; });
  r.on('epoch', () => { m.testCache = null; });
  return r;
}

// Change hyperparameters: regenerate data if needed and start over.
function applyHp(patch, { preset = null } = {}) {
  const m = mounted;
  const before = m.hp;
  m.hp = { ...m.hp, ...patch };
  const dataKeys = m.def.dataKeys;
  const dataChanged = !dataKeys || Object.keys(patch).some((k) => dataKeys.includes(k) && before[k] !== m.hp[k]);
  m.runner.reset();
  if (dataChanged) m.data = m.def.makeData(m.hp);
  m.runner = makeRunner(m);
  m.activePreset = preset;
  render();
}

function render() {
  const m = mounted;
  if (!m) return;
  m.cleanup?.();
  m.cleanup = null;
  const { def, tab } = m;
  const body = h('div');
  m.el.replaceChildren(
    h('div', { class: 'module-head' },
      h('div', null,
        h('a', { class: 'back', href: '#/' }, '← ', t('nav.home')),
        h('h1', null, L(def.title)),
        h('p', null, L(def.summary))),
      def.synthetic && h('span', { class: 'badge badge-synthetic', title: t('badge.syntheticHint') }, t('badge.synthetic'))),
    h('nav', { class: 'tabs' }, TABS.map((id, i) =>
      h('a', { class: 'tab', href: moduleHref(def.id, id), 'aria-current': id === tab ? 'page' : null },
        h('span', { class: 'n' }, String(i + 1)), t(`tab.${id}`)))),
    body,
    nextStep(def, tab) || '');
  ({ arch: renderArchTab, data: renderDataTab, train: renderTrainTab, diagnose: renderDiagnoseTab, test: renderTestTab, use: renderUse })[tab](body, m);
}

function nextStep(def, tab) {
  const i = TABS.indexOf(tab);
  if (i === TABS.length - 1) return null;
  return h('div', { class: 'tab-foot' },
    h('a', { class: 'btn', href: moduleHref(def.id, TABS[i + 1]) }, `${t('btn.next')}: ${t(`tab.${TABS[i + 1]}`)} →`));
}

const tx = (def, key, vars) => t(key, vars, def.text);
const label = (def, l) => (typeof l === 'string' ? tx(def, l) : L(l));

function controlsFor(m, where) {
  const { def, hp } = m;
  const items = def.controls.filter((c) => c.where === where);
  if (!items.length) return null;
  return h('div', { class: 'controls' }, items.map((c) => {
    const id = `ctl-${def.id}-${c.key}`;
    if (c.type === 'select') {
      return h('div', { class: 'control' },
        h('label', { for: id }, label(def, c.label)),
        h('select', { id, onchange: (e) => applyHp({ [c.key]: parseValue(c, e.target.value) }) },
          c.options.map((o) => h('option', { value: String(o.value), selected: String(o.value) === String(hp[c.key]) }, label(def, o.label)))));
    }
    if (c.type === 'toggle') {
      return h('label', { class: 'control-toggle' },
        h('input', { type: 'checkbox', checked: !!hp[c.key], onchange: (e) => applyHp({ [c.key]: e.target.checked }) }),
        label(def, c.label));
    }
    const val = h('span', { class: 'val' }, formatControl(c, hp[c.key]));
    const input = h('input', {
      id, type: 'range', min: c.min, max: c.max, step: c.step ?? 1,
      value: c.log ? Math.log10(hp[c.key]) : hp[c.key],
      oninput: (e) => { val.textContent = formatControl(c, readRange(c, e.target.value)); },
      onchange: (e) => applyHp({ [c.key]: readRange(c, e.target.value) }),
    });
    return h('div', { class: 'control' }, h('label', { for: id }, h('span', null, label(def, c.label)), val), input);
  }));
}

function parseValue(c, v) {
  const opt = c.options.find((o) => String(o.value) === v);
  return opt ? opt.value : v;
}
function readRange(c, v) {
  const x = Number(v);
  return c.log ? Number((10 ** x).toPrecision(2)) : x;
}
function formatControl(c, v) {
  return c.format === 'pct' ? `${Math.round(v * 100)}%` : String(v);
}

/* ---------- 1. Architecture ---------- */
function renderArchTab(body, m) {
  const archEl = h('div');
  const ctl = controlsFor(m, 'arch');
  body.append(...[
    m.def.archNote && h('div', { class: 'disclaimer', style: { marginBottom: '16px' } }, tx(m.def, m.def.archNote)),
    m.def.renderArchIntro && (() => { const d = h('div', { style: { marginBottom: '16px' } }); m.def.renderArchIntro(d, ctxFor(m)); return d; })(),
    archEl,
    ctl && h('div', { class: 'card', style: { marginTop: '16px' } },
      h('div', { class: 'card-title' }, h('h3', null, t('arch.settings')), h('span', { class: 'small muted' }, t('arch.changeResets'))),
      ctl),
  ].filter(Boolean));
  renderArch(archEl, m.def.architecture(m.hp, m.data), { selected: m.archSel, onSelect: (i) => { m.archSel = i; } });
}

/* ---------- 2. Data ---------- */
function renderDataTab(body, m) {
  const { counts } = m.data.meta;
  const total = counts.train + counts.val + counts.test;
  const seg = (cls, key, n) => h('div', { class: cls, style: { flex: `${n} 1 0` }, title: `${t(key)} ${n}` }, `${t(key)} ${n}`);
  const samples = h('div');
  const dataCtl = controlsFor(m, 'data');
  body.append(h('div', { class: 'stack' },
    h('div', { class: 'card' },
      h('div', { class: 'card-title' }, h('h3', null, t('data.split')), h('span', { class: 'small muted mono' }, `${total}`)),
      h('div', { class: 'split-bar' }, seg('split-train', 'data.train', counts.train), seg('split-val', 'data.val', counts.val), seg('split-test', 'data.test', counts.test)),
      h('p', { class: 'help', style: { marginTop: '12px', marginBottom: 0 } }, m.def.dataWhy ? tx(m.def, m.def.dataWhy) : t('data.why'))),
    dataCtl && h('div', { class: 'card' }, dataCtl),
    h('div', { class: 'card' },
      h('div', { class: 'card-title' }, h('h3', null, t('data.samples')),
        m.def.synthetic && h('span', { class: 'badge badge-synthetic', title: t('badge.syntheticHint') }, t('badge.synthetic'))),
      samples)));
  m.def.renderData(samples, m.data, ctxFor(m));
}

function ctxFor(m) {
  return {
    hp: m.hp, data: m.data, runner: m.runner, def: m.def,
    get session() { return m.runner.session; },
    t: (key, vars) => tx(m.def, key, vars),
    L,
    onCleanup(fn) { const prev = m.cleanup; m.cleanup = () => { prev?.(); fn(); }; },
  };
}

/* ---------- 3. Train ---------- */
function renderTrainTab(body, m) {
  const { def } = m;
  const runner = m.runner;
  const dot = h('span', { class: 'status-dot' });
  const stateText = h('span');
  const epochText = h('span', { class: 'mono small muted' });
  const bar = h('div');
  const errText = h('div', { class: 'small', style: { color: 'var(--bad)' } });
  const startBtn = h('button', { class: 'btn btn-primary', type: 'button', onclick: () => startTraining() });
  const pauseBtn = h('button', { class: 'btn', type: 'button', onclick: () => runner.pause() }, t('btn.pause'));
  const resetBtn = h('button', { class: 'btn', type: 'button', onclick: () => applyHp({}, { preset: m.activePreset }) }, t('btn.reset'));
  const diagLink = h('a', { class: 'btn', href: moduleHref(def.id, 'diagnose') }, `${t('tab.diagnose')} →`);

  const loss = chartBox();
  const metric = chartBox();
  const vizEl = h('div');
  const metricDefs = def.metrics || [];

  body.append(h('div', { class: 'stack' },
    h('div', { class: 'card' },
      h('div', { class: 'card-title' }, h('h3', null, t('train.settings'))),
      controlsFor(m, 'train'),
      h('div', { class: 'row', style: { marginTop: '16px' } }, startBtn, pauseBtn, resetBtn, diagLink),
      h('div', { class: 'status-line', style: { marginTop: '12px' } }, dot, stateText, h('div', { class: 'progress' }, bar), epochText),
      errText),
    def.breakPresets?.length && h('div', { class: 'card' },
      h('div', { class: 'card-title' }, h('h3', null, t('train.presets')),
        m.activePreset && h('button', { class: 'btn btn-sm', type: 'button', onclick: () => { m.hp = defaultsFor(def); applyHp({}); } }, t('btn.defaults'))),
      h('p', { class: 'help' }, t('train.presetsHelp')),
      h('div', { class: 'presets' }, def.breakPresets.map((p) =>
        h('button', {
          class: `btn preset-btn${m.activePreset === p.id ? ' btn-primary' : ''}`, type: 'button',
          onclick: () => { m.hp = defaultsFor(def); applyHp(p.hp, { preset: p.id }); startTraining(); },
        }, label(def, p.label))))),
    h('div', { class: 'grid-2' },
      h('div', { class: 'card' }, h('h3', null, t('train.lossChart')), loss.el, h('p', { class: 'help', style: { marginTop: '8px' } }, t('train.lossHelp'))),
      metricDefs.length
        ? h('div', { class: 'card' }, h('h3', null, t('train.metricChart')), metric.el)
        : null),
    def.renderTrainViz && h('div', { class: 'card' }, h('h3', null, t('train.inside')), vizEl)));

  const lossChart = lineChart(loss.canvas, { xLabel: t('train.epoch'), emptyText: t('chart.empty') });
  const metricChart = metricDefs.length ? lineChart(metric.canvas, { xLabel: t('train.epoch'), emptyText: t('chart.empty') }) : null;
  const viz = def.renderTrainViz ? def.renderTrainViz(vizEl, ctxFor(m)) : null;
  const colors = ['--series-3', '--series-4', '--series-1', '--series-2'];

  function update() {
    const hist = runner.history;
    const s = runner.state;
    dot.className = `status-dot ${s}`;
    stateText.textContent = t(`state.${s}`);
    errText.textContent = s === 'error' ? String(runner.error?.message || runner.error) : '';
    epochText.textContent = `${t('train.epochOf', { n: hist.length, total: runner.epochs })} · ${t('train.seed', { seed: m.hp.seed })}`;
    bar.style.width = `${(hist.length / runner.epochs) * 100}%`;
    startBtn.textContent = s === 'paused' ? t('btn.resume') : t('btn.start');
    startBtn.disabled = s === 'running' || s === 'done' || s === 'diverged' || s === 'error';
    pauseBtn.disabled = s !== 'running';
    diagLink.hidden = !hist.length;
    lossChart.update([
      { name: t('train.trainLoss'), values: hist.map((x) => x.trainLoss), color: '--series-1' },
      { name: t('train.valLoss'), values: hist.map((x) => x.valLoss), color: '--series-2' },
    ]);
    metricChart?.update(metricDefs.map((d, i) => ({
      name: label(def, d.label), values: hist.map((x) => x.metrics?.[d.key]), color: d.color || colors[i % colors.length], dashed: d.dashed,
    })), metricDefs.every((d) => d.format === 'pct') ? { yMin: 0, yMax: 1 } : {});
  }
  const offE = runner.on('epoch', () => { update(); viz?.update(); });
  const offS = runner.on('state', update);
  m.cleanup = () => { offE(); offS(); viz?.destroy?.(); };
  update();
  viz?.update();
}

/* ---------- 4. Diagnose ---------- */
function renderDiagnoseTab(body, m) {
  const { def, runner } = m;
  const hist = runner.history;
  if (!hist.length) {
    body.append(h('div', { class: 'card' }, h('p', null, t('diag.noRun')), h('a', { class: 'btn btn-primary', href: moduleHref(def.id, 'train') }, t('btn.goTrain'))));
    return;
  }
  const baselineLoss = def.baselineLoss(m.hp, m.data);
  const d = diagnoseCurves(hist, { baselineLoss });
  const level = { good: 'good', insufficient: 'info', diverged: 'bad', underfitting: 'warn', overfitting: 'bad' }[d.verdict];
  const icon = { good: '✓', info: '…', bad: '!', warn: '~' }[level];
  const findingsEl = h('ul', { class: 'findings' });
  const extraEl = h('div', { class: 'stack' });
  const extraBanner = h('div');
  const addFindings = (list) => list.forEach((f) => findingsEl.append(h('li', { class: f.level }, tx(def, f.key, f.vars))));
  addFindings(d.findings);

  const last = hist[hist.length - 1];
  const stat = (lab, value, sub) => h('div', { class: 'stat' }, h('div', { class: 'label' }, lab), h('div', { class: 'value' }, value), sub && h('div', { class: 'sub' }, sub));
  const loss = chartBox();

  body.append(h('div', { class: 'stack' },
    h('div', { class: `verdict ${level}` },
      h('div', { class: 'icon', 'aria-hidden': 'true' }, icon),
      h('div', null, h('h2', null, t(`verdict.${d.verdict}`)), h('p', null, t(`verdictdesc.${d.verdict}`)))),
    extraBanner,
    h('div', { class: 'grid-2' },
      h('div', { class: 'card' }, h('h3', null, t('diag.evidence')), findingsEl,
        d.advice.length ? h('div', { style: { marginTop: '14px' } }, h('h3', null, t('diag.advice')),
          h('ol', { class: 'advice' }, d.advice.map((a) => h('li', null, tx(def, a))))) : null),
      h('div', { class: 'card' }, h('h3', null, t('train.lossChart')), loss.el,
        d.bestEpoch ? h('p', { class: 'small muted', style: { marginTop: '6px' } }, t('diag.bestEpoch', { n: d.bestEpoch })) : null)),
    h('div', { class: 'card' },
      h('h3', null, t('diag.metrics')),
      h('div', { class: 'stats' },
        stat(t('train.trainLoss'), fmt(last.trainLoss)),
        stat(t('train.valLoss'), fmt(last.valLoss)),
        stat(tx(def, def.baselineLabel || 'diag.baselineShort'), fmt(baselineLoss)),
        (def.metrics || []).map((md) => stat(label(def, md.label), md.format === 'pct' ? pct(last.metrics?.[md.key]) : fmt(last.metrics?.[md.key])))),
      h('p', { class: 'help', style: { marginTop: '12px', marginBottom: 0 } }, t('diag.howTo'))),
    extraEl));

  lineChart(loss.canvas, { xLabel: t('train.epoch'), markX: d.bestEpoch || null }).update([
    { name: t('train.trainLoss'), values: hist.map((x) => x.trainLoss), color: '--series-1' },
    { name: t('train.valLoss'), values: hist.map((x) => x.valLoss), color: '--series-2' },
    { name: t('diag.baseline', { v: fmt(baselineLoss) }), values: hist.map(() => baselineLoss), color: '--muted', dashed: true },
  ]);

  if (def.diagnosePanel && runner.session && d.verdict !== 'diverged') {
    const token = {};
    m.diagToken = token;
    Promise.resolve(def.diagnosePanel(extraEl, ctxFor(m), hist)).then((extra = []) => {
      if (m.diagToken !== token || !extraEl.isConnected) return;
      addFindings(extra);
      if (d.verdict === 'good' && extra.some((f) => f.level === 'bad')) {
        extraBanner.replaceChildren(h('div', { class: 'verdict warn' }, h('div', { class: 'icon' }, '!'),
          h('div', null, h('p', null, t('diag.extraProblem')))));
      }
    });
  }
}

/* ---------- 6. Use ---------- */
function renderUse(body, m) {
  const off = renderUseTab(body, m, ctxFor(m));
  if (off) { const prev = m.cleanup; m.cleanup = () => { prev?.(); off(); }; }
}

/* ---------- 5. Test ---------- */
function renderTestTab(body, m) {
  const { def, runner } = m;
  if (!runner.session || !runner.history.length || runner.state === 'running') {
    body.append(h('div', { class: 'card' },
      h('p', null, runner.state === 'running' ? t('test.waitTraining') : t('test.needTrain')),
      h('a', { class: 'btn btn-primary', href: moduleHref(def.id, 'train') }, t('btn.goTrain'))));
    return;
  }
  const statsEl = h('div', { class: 'stats' }, h('span', { class: 'muted small' }, t('test.running')));
  const tryEl = h('div', { class: 'stack' });
  body.append(h('div', { class: 'stack' },
    h('div', { class: 'card' }, h('h3', null, t('test.results')), h('p', { class: 'help' }, t('test.explain')), statsEl),
    tryEl));
  const show = (res) => {
    statsEl.replaceChildren(...(def.testMetrics || []).map((md) => h('div', { class: 'stat' },
      h('div', { class: 'label' }, label(def, md.label)),
      h('div', { class: 'value' }, md.format === 'pct' ? pct(res[md.key]) : fmt(res[md.key])),
      md.sub && h('div', { class: 'sub' }, tx(def, md.sub, res)))));
  };
  if (m.testCache) show(m.testCache);
  else {
    runner.session.evaluateTest().then((res) => { m.testCache = res; if (statsEl.isConnected) show(res); });
  }
  def.renderTest(tryEl, ctxFor(m));
}
