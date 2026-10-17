// Lightweight canvas charts that follow the page's CSS color tokens.
import { cssVar, fmt, h } from './dom.js';

function setup(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement?.clientWidth || 300;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, height);
  ctx.font = `11px ${cssVar('--mono') || 'monospace'}`;
  return { ctx, w, hgt: height };
}

function color(c) {
  return c?.startsWith('--') ? cssVar(c) : c;
}

// Resolve any CSS color string to [r, g, b].
function rgb(c) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 1;
  const x = cv.getContext('2d');
  x.fillStyle = c;
  x.fillRect(0, 0, 1, 1);
  return Array.from(x.getImageData(0, 0, 1, 1).data.slice(0, 3));
}

function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function observe(canvas, draw) {
  if (typeof ResizeObserver === 'undefined') return;
  let lastW = canvas.clientWidth;
  const ro = new ResizeObserver(() => {
    if (!canvas.isConnected) { ro.disconnect(); return; }
    if (canvas.clientWidth !== lastW) { lastW = canvas.clientWidth; draw(); }
  });
  ro.observe(canvas);
}

/**
 * series: [{ name, values: number[], color: '--series-1' | css color, dashed? }]
 * opts: { xLabel, yLabel, markX (1-based x to mark), height, yMin, yMax, xStart (default 1), emptyText }
 * Non-finite values are skipped, so a diverged run still draws up to its last real point.
 */
export function lineChart(canvas, opts) {
  let state = { ...opts };
  const wrap = canvas.parentElement;
  let legend = null;
  let empty = null;
  if (wrap) {
    legend = h('div', { class: 'legend' });
    wrap.after(legend);
    if (opts.emptyText) { empty = h('div', { class: 'chart-empty' }, opts.emptyText); wrap.append(empty); }
  }

  function draw() {
    const { series = [], height = 220, xStart = 1 } = state;
    const { ctx, w, hgt } = setup(canvas, height);
    const pad = { l: 44, r: 12, t: 10, b: 26 };
    const all = series.flatMap((s) => s.values.filter(Number.isFinite));
    const n = Math.max(0, ...series.map((s) => s.values.length));
    if (empty) empty.hidden = n > 0;
    if (legend) {
      legend.replaceChildren(...series.filter((s) => s.name).map((s) =>
        h('span', null, h('i', { style: { background: color(s.color) } }), s.name)));
    }
    let yMin = state.yMin ?? Math.min(...all, Infinity);
    let yMax = state.yMax ?? Math.max(...all, -Infinity);
    if (!all.length) { yMin = 0; yMax = 1; }
    if (yMax - yMin < 1e-9) { yMax += 0.5; yMin -= 0.5; }
    const span = yMax - yMin;
    if (state.yMin == null) yMin -= span * 0.05;
    if (state.yMax == null) yMax += span * 0.05;
    const xMax = Math.max(n - 1, 1);
    const X = (i) => pad.l + (i / xMax) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (hgt - pad.t - pad.b);

    ctx.strokeStyle = cssVar('--grid');
    ctx.fillStyle = cssVar('--muted');
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of niceTicks(yMin, yMax)) {
      const y = Math.round(Y(v)) + 0.5;
      if (y < pad.t - 1 || y > hgt - pad.b + 1) continue;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(fmt(v, 3), pad.l - 6, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xt = niceTicks(xStart, xStart + xMax, Math.min(6, Math.max(1, xMax))).filter((v) => Number.isInteger(v));
    for (const v of xt) ctx.fillText(String(v), X(v - xStart), hgt - pad.b + 6);
    if (state.xLabel) { ctx.textAlign = 'right'; ctx.fillText(state.xLabel, w - pad.r, hgt - 12); }

    if (state.markX != null && n) {
      const x = X(state.markX - xStart);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = cssVar('--good');
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, hgt - pad.b); ctx.stroke();
      ctx.restore();
    }

    for (const s of series) {
      ctx.strokeStyle = color(s.color);
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dashed ? [5, 4] : []);
      ctx.beginPath();
      let pen = false;
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v)) { pen = false; return; }
        const y = Math.max(pad.t, Math.min(hgt - pad.b, Y(v)));
        if (pen) ctx.lineTo(X(i), y); else ctx.moveTo(X(i), y);
        pen = true;
      });
      ctx.stroke();
      if (s.values.length === 1 && Number.isFinite(s.values[0])) {
        ctx.fillStyle = color(s.color);
        ctx.beginPath(); ctx.arc(X(0), Y(s.values[0]), 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.setLineDash([]);
  }

  draw();
  observe(canvas, draw);
  return {
    update(series, extra = {}) { state = { ...state, ...extra, series }; draw(); },
    redraw: draw,
  };
}

/** matrix[r][c] → colored cells; opts: { rowLabels, colLabels, height, showValues, rowTitle, colTitle } */
export function heatmap(canvas, matrix, opts = {}) {
  function draw() {
    const rows = matrix.length;
    const cols = matrix[0]?.length || 0;
    const height = opts.height ?? Math.min(360, 28 + rows * 28);
    const { ctx, w, hgt } = setup(canvas, height);
    const pad = { l: opts.rowLabels ? 40 : 6, t: opts.colLabels ? 22 : 6, r: 6, b: 6 };
    const cw = (w - pad.l - pad.r) / cols;
    const ch = (hgt - pad.t - pad.b) / rows;
    const max = Math.max(1e-9, ...matrix.flat());
    const accent = cssVar('--accent');
    const surface = cssVar('--surface-2');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c] / max;
        ctx.fillStyle = surface;
        ctx.fillRect(pad.l + c * cw, pad.t + r * ch, cw - 1, ch - 1);
        ctx.globalAlpha = Math.max(0, Math.min(1, v));
        ctx.fillStyle = accent;
        ctx.fillRect(pad.l + c * cw, pad.t + r * ch, cw - 1, ch - 1);
        ctx.globalAlpha = 1;
        if (opts.showValues && cw > 18 && ch > 14) {
          ctx.fillStyle = v > 0.55 ? '#fff' : cssVar('--text');
          ctx.fillText(String(matrix[r][c]), pad.l + c * cw + cw / 2, pad.t + r * ch + ch / 2);
        }
      }
    }
    ctx.fillStyle = cssVar('--muted');
    if (opts.colLabels) opts.colLabels.forEach((lab, c) => cw > 7 && ctx.fillText(lab, pad.l + c * cw + cw / 2, 10));
    if (opts.rowLabels) {
      ctx.textAlign = 'right';
      opts.rowLabels.forEach((lab, r) => ch > 7 && ctx.fillText(lab, pad.l - 6, pad.t + r * ch + ch / 2));
    }
  }
  draw();
  observe(canvas, draw);
  return { update(m) { matrix = m; draw(); } };
}

/** labels: string[], values: number[] (0..1 shown as bars); opts: { height, color } */
export function barChart(canvas, { labels, values, height = 180, color: c = '--series-1' }) {
  function draw() {
    const { ctx, w, hgt } = setup(canvas, height);
    const pad = { l: 8, r: 8, t: 10, b: 22 };
    const max = Math.max(1e-9, ...values);
    const bw = (w - pad.l - pad.r) / values.length;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    values.forEach((v, i) => {
      const bh = (v / max) * (hgt - pad.t - pad.b);
      ctx.fillStyle = color(c);
      ctx.fillRect(pad.l + i * bw + bw * 0.15, hgt - pad.b - bh, bw * 0.7, bh);
      ctx.fillStyle = cssVar('--muted');
      ctx.fillText(labels[i], pad.l + i * bw + bw / 2, hgt - pad.b + 5);
    });
  }
  draw();
  observe(canvas, draw);
  return { update(v) { values = v; draw(); } };
}

/**
 * Scatter points with optional background field (for decision boundaries).
 * points: [{ x, y, c (class index), ring? }]; opts: { domain: [min,max], field: { n, values (n*n, 0..1) }, colors, height }
 */
export function scatter(canvas, points, opts = {}) {
  let state = { points, ...opts };
  function draw() {
    const { domain = [-1, 1], field, colors = ['--series-1', '--series-2', '--series-3', '--series-4'] } = state;
    const height = state.height ?? (canvas.clientWidth || 300);
    const { ctx, w, hgt } = setup(canvas, Math.min(height, 420));
    const [lo, hi] = domain;
    const X = (v) => ((v - lo) / (hi - lo)) * w;
    const Y = (v) => (1 - (v - lo) / (hi - lo)) * hgt;
    if (field) {
      // Paint the n×n field into a tiny image and let the browser smooth it up to size.
      const [r0, g0, b0] = rgb(color(colors[0]));
      const [r1, g1, b1] = rgb(color(colors[1]));
      const off = document.createElement('canvas');
      off.width = off.height = field.n;
      const octx = off.getContext('2d');
      const img = octx.createImageData(field.n, field.n);
      for (let j = 0; j < field.n; j++) {
        for (let i = 0; i < field.n; i++) {
          const p = field.values[j * field.n + i];
          const k = ((field.n - 1 - j) * field.n + i) * 4;
          const one = p >= 0.5;
          img.data[k] = one ? r1 : r0;
          img.data[k + 1] = one ? g1 : g0;
          img.data[k + 2] = one ? b1 : b0;
          img.data[k + 3] = Math.round((0.1 + Math.abs(p - 0.5) * 0.6) * 255);
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, w, hgt);
    } else {
      ctx.fillStyle = cssVar('--surface-2');
      ctx.fillRect(0, 0, w, hgt);
    }
    for (const p of state.points) {
      ctx.beginPath();
      ctx.arc(X(p.x), Y(p.y), p.r ?? 3.2, 0, Math.PI * 2);
      ctx.fillStyle = color(colors[p.c % colors.length]);
      ctx.globalAlpha = p.alpha ?? 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (p.ring) {
        ctx.strokeStyle = cssVar('--bad');
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 6, 0, Math.PI * 2); ctx.stroke();
      }
    }
    if (state.marker) {
      ctx.strokeStyle = cssVar('--text');
      ctx.lineWidth = 2;
      const mx = X(state.marker.x), my = Y(state.marker.y);
      ctx.beginPath(); ctx.moveTo(mx - 7, my); ctx.lineTo(mx + 7, my); ctx.moveTo(mx, my - 7); ctx.lineTo(mx, my + 7); ctx.stroke();
    }
  }
  draw();
  observe(canvas, draw);
  return {
    update(next) { state = { ...state, ...next }; draw(); },
    // Convert a mouse event to data coordinates.
    toData(ev) {
      const r = canvas.getBoundingClientRect();
      const [lo, hi] = state.domain || [-1, 1];
      return { x: lo + ((ev.clientX - r.left) / r.width) * (hi - lo), y: lo + (1 - (ev.clientY - r.top) / r.height) * (hi - lo) };
    },
  };
}

// Convenience: a card-ready chart container. Returns { el, canvas }.
export function chartBox() {
  const canvas = h('canvas');
  const wrap = h('div', { class: 'chart-wrap' }, canvas);
  return { el: h('div', null, wrap), canvas };
}
