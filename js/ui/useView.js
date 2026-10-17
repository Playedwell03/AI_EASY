// "Use" tab: plug the trained model into small apps and show where in the app it works.
import { h } from './dom.js';
import { t, L } from '../i18n.js';
import { moduleHref } from '../router.js';

export function renderUseTab(body, m, ctx) {
  const { def, runner } = m;
  if (!runner.session || !runner.history.length || runner.state === 'running') {
    body.append(h('div', { class: 'card' },
      h('p', null, runner.state === 'running' ? t('test.waitTraining') : t('use.needTrain')),
      h('a', { class: 'btn btn-primary', href: moduleHref(def.id, 'train') }, t('btn.goTrain'))));
    return;
  }
  const cases = def.useCases || [];
  m.useSel = Math.min(m.useSel ?? 0, cases.length - 1);
  const picker = h('div', { class: 'use-picker' });
  const stage = h('div', { class: 'stack' });
  body.append(h('div', { class: 'stack' },
    h('div', { class: 'card' }, h('h3', null, t('use.pick')), h('p', { class: 'help' }, t('use.pickHelp')), picker),
    stage));

  const cards = cases.map((u, i) => h('button', {
    class: 'use-card', type: 'button', 'aria-pressed': String(i === m.useSel),
    onclick: () => { m.useSel = i; cards.forEach((c, j) => c.setAttribute('aria-pressed', String(j === i))); open(u); },
  }, h('span', { class: 'ico', 'aria-hidden': 'true' }, u.icon), h('div', null, h('b', null, L(u.title)), h('span', null, L(u.intro)))));
  picker.append(...cards);

  let cleanup = null;
  function open(u) {
    cleanup?.();
    cleanup = null;
    const desc = h('div', { class: 'step-desc' });
    const steps = u.pipeline.map((s, i) => h('button', {
      class: `pipe-step ${s.kind}`, type: 'button', 'aria-pressed': 'false',
      onclick: () => select(i),
    },
    h('span', { class: 'k' }, s.kind === 'model' ? `🧠 ${t('use.modelStep')}` : t(`use.kind.${s.kind}`)),
    h('span', { class: 'l' }, L(s.label))));
    const select = (i) => {
      steps.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
      desc.replaceChildren(h('b', null, L(u.pipeline[i].label)), ' — ', L(u.pipeline[i].desc));
    };
    const bar = h('div', { class: 'pipeline' });
    steps.forEach((b, i) => { if (i) bar.append(h('span', { class: 'pipe-arrow', 'aria-hidden': 'true' }, '→')); bar.append(b); });
    const timers = new Map();
    const flow = {
      pulse(i) {
        const b = steps[i];
        if (!b) return;
        b.classList.add('pulse');
        clearTimeout(timers.get(i));
        timers.set(i, setTimeout(() => b.classList.remove('pulse'), 700));
      },
      // Pulse every step in order, like data flowing through the app.
      run() { steps.forEach((_, i) => setTimeout(() => flow.pulse(i), i * 140)); },
    };
    const appBody = h('div', { class: 'app-body' });
    stage.replaceChildren(
      h('div', { class: 'card' },
        h('div', { class: 'card-title' }, h('h3', null, t('use.flow')), h('span', { class: 'small muted' }, t('use.flowHelp'))),
        bar, desc),
      h('div', { class: 'app-frame' },
        h('div', { class: 'app-bar' }, h('span', { class: 'dots' }, h('i'), h('i'), h('i')), h('span', null, `${u.icon} ${L(u.title)}`)),
        appBody),
      h('div', { class: 'real-world' }, h('b', null, `${t('use.realWorld')}: `), L(u.realWorld)));
    select(u.pipeline.findIndex((s) => s.kind === 'model'));
    const out = u.render(appBody, ctx, flow);
    cleanup = () => { timers.forEach(clearTimeout); out?.destroy?.(); };
  }
  if (cases.length) open(cases[m.useSel]);
  return () => cleanup?.();
}

// Helpers for building use cases.
export const step = (kind, ko, en, dko, den) => ({ kind, label: { ko, en }, desc: { ko: dko, en: den } });
export const lt = (ko, en) => ({ ko, en });
