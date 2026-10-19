// Tiny DOM helper: h('div', { class: 'card', onclick }, child, 'text', [more]).

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'html') el.innerHTML = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

export function fmt(x, digits = 3) {
  if (x == null || !Number.isFinite(x)) return '—';
  if (Math.abs(x) >= 1000) return x.toFixed(0);
  return Number(x.toPrecision(digits)).toString();
}

export function pct(x) {
  return x == null || !Number.isFinite(x) ? '—' : `${(x * 100).toFixed(1)}%`;
}

// Draw a grayscale image (values 0..1) into a canvas at its native size.
export function drawGray(canvas, pixels, w, hgt = w) {
  canvas.width = w;
  canvas.height = hgt;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, hgt);
  for (let i = 0; i < w * hgt; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Labelled probability bars; returns the element.
export function probBars(labels, probs, { highlight } = {}) {
  const top = highlight ?? probs.indexOf(Math.max(...probs));
  return h('div', { class: 'bars' }, labels.map((lab, i) =>
    h('div', { class: `bar-row${i === top ? ' top' : ''}` },
      h('span', { class: 'mono', style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, lab),
      h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(probs[i] * 100).toFixed(1)}%` } })),
      h('span', { class: 'num' }, `${(probs[i] * 100).toFixed(1)}%`))));
}
