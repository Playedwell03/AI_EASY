import { h } from './dom.js';
import { t, L } from '../i18n.js';
import { TABS, moduleHref } from '../router.js';

export function renderHome(el, modules) {
  el.replaceChildren(
    h('section', { class: 'hero' },
      h('h1', null, t('home.title')),
      h('p', null, t('home.intro')),
      modules[0] && h('a', { class: 'btn btn-primary', href: moduleHref(modules[0].id) }, t('home.start'), ' →')),
    h('h3', { class: 'muted small', style: { marginTop: '28px' } }, t('home.flow')),
    h('div', { class: 'flow' }, TABS.map((tab) =>
      h('div', { class: 'flow-step' }, h('b', null, t(`tab.${tab}`)), h('span', null, t(`tabdesc.${tab}`))))),
    h('h2', null, t('home.modules')),
    h('div', { class: 'module-grid' }, modules.map((m, i) =>
      h('a', { class: 'card module-card', href: moduleHref(m.id) },
        h('div', { class: 'row', style: { justifyContent: 'space-between' } },
          h('span', { class: 'num' }, String(i + 1).padStart(2, '0')),
          m.synthetic && h('span', { class: 'badge badge-synthetic' }, t('badge.synthetic'))),
        h('h3', null, L(m.title)),
        h('p', null, L(m.summary)),
        m.archHint && h('div', { class: 'arch-hint' }, m.archHint)))),
  );
}
