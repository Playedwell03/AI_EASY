import { h } from './dom.js';
import { t, L } from '../i18n.js';

const shape = (s) => `[${s.join(', ')}]`;

// layers: [{ name, kind, inShape, outShape, params, desc: LocalText }]
export function renderArch(el, layers, { selected = 0, onSelect } = {}) {
  let current = Math.min(selected, layers.length - 1);
  const detail = h('div', { class: 'card arch-detail' });
  const blocks = layers.map((layer, i) =>
    h('button', {
      class: 'arch-block', type: 'button', 'aria-pressed': String(i === current),
      onclick: () => select(i),
    },
    h('span', { class: `kind kind-${layer.kind}` }, layer.kind),
    h('span', { class: 'name' }, layer.name),
    h('span', { class: 'shape' }, shape(layer.outShape))));

  function select(i) {
    current = i;
    blocks.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
    const layer = layers[i];
    detail.replaceChildren(
      h('h3', null, layer.name),
      h('p', null, L(layer.desc)),
      h('dl', null,
        h('dt', null, t('arch.input')), h('dd', null, shape(layer.inShape)),
        h('dt', null, t('arch.output')), h('dd', null, shape(layer.outShape)),
        h('dt', null, t('arch.params')), h('dd', null, layer.params.toLocaleString())));
    onSelect?.(i);
  }

  const stack = h('div', { class: 'arch-stack' });
  blocks.forEach((b, i) => {
    if (i) stack.append(h('div', { class: 'arch-arrow', 'aria-hidden': 'true' }, '↓'));
    stack.append(b);
  });
  const total = layers.reduce((s, l) => s + l.params, 0);
  el.replaceChildren(h('div', { class: 'arch-layout' },
    h('div', { class: 'stack' },
      h('p', { class: 'help' }, t('arch.clickHint')),
      stack,
      h('p', { class: 'small muted' }, `${t('arch.totalParams')}: `, h('b', { class: 'mono' }, total.toLocaleString()))),
    detail));
  if (layers.length) select(current);
}
