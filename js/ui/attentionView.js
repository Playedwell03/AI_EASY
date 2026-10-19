// Attention heatmap with block/head selectors (Transformer, BERT, Jev-style modules).
import { h } from './dom.js';
import { heatmap } from './chart.js';

const show = (c) => (c === ' ' ? '␣' : c);

// Heatmap of one layer/head for some text, with selectors. Returns { update(text) }.
export function attentionExplorer(el, ctx, getMaps, { help } = {}) {
  const layerSel = h('select', { class: 'text-input', style: { width: 'auto' } });
  const headSel = h('select', { class: 'text-input', style: { width: 'auto' } });
  const canvas = h('canvas');
  let maps = [];
  let chars = [];
  const draw = () => {
    const m = maps[Number(layerSel.value) || 0]?.[Number(headSel.value) || 0];
    if (!m) return;
    heatmap(canvas, m, { rowLabels: chars, colLabels: chars, height: Math.min(420, 30 + chars.length * 16) });
  };
  layerSel.addEventListener('change', draw);
  headSel.addEventListener('change', draw);
  el.append(
    help && h('p', { class: 'help' }, help),
    h('div', { class: 'row', style: { marginBottom: '8px' } },
      h('label', { class: 'small muted' }, ctx.t('tf.layer')), layerSel,
      h('label', { class: 'small muted' }, ctx.t('tf.head')), headSel),
    h('div', { class: 'chart-wrap', style: { maxWidth: '520px' } }, canvas),
    h('p', { class: 'small muted', style: { marginTop: '6px' } }, ctx.t('tf.howToRead')));
  return {
    update(text) {
      const r = getMaps(text);
      maps = r.maps;
      chars = r.chars.map(show);
      if (!maps.length) return;
      if (layerSel.options.length !== maps.length) {
        layerSel.replaceChildren(...maps.map((_, i) => h('option', { value: i }, String(i + 1))));
        headSel.replaceChildren(...maps[0].map((_, i) => h('option', { value: i }, String(i + 1))));
      }
      draw();
    },
  };
}

export const TF_TEXT = {
  ko: {
    'tf.layer': '블록',
    'tf.head': '헤드',
    'tf.howToRead': '행: 지금 글자 · 열: 참고하는 글자. 칸이 진할수록 그 글자를 많이 참고합니다.',
    'tf.attnTrain': '아래 문장에 대한 어텐션입니다. 학습 전에는 고르게 퍼져 있다가, 학습할수록 특정 글자(바로 앞 글자, 띄어쓰기, 목적어 등)에 집중하는 무늬가 생깁니다.',
    'tf.attnTest': '입력한 문장의 마지막 부분에 대한 어텐션입니다. 대각선 위쪽이 비어 있는 것은 미래 글자를 볼 수 없게 가렸기(마스크) 때문입니다.',
    'tf.attnTitle': '어텐션 지도',
  },
  en: {
    'tf.layer': 'Block',
    'tf.head': 'Head',
    'tf.howToRead': 'Rows: current character · columns: characters it draws on. Darker cells mean more attention.',
    'tf.attnTrain': 'Attention for the text below. Before training it is spread evenly; as training goes on, patterns appear that focus on particular characters (the previous one, spaces, the object).',
    'tf.attnTest': 'Attention for the end of the text you typed. The empty area above the diagonal is the mask: it cannot see future characters.',
    'tf.attnTitle': 'Attention map',
  },
};

