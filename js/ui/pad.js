// Shared image helpers: a small grayscale image tile and a mouse/touch drawing pad.
import { h, drawGray } from './dom.js';
import { DIGIT_SIZE as S } from '../shared/digits.js';

export function imgCell(pixels, caption, { wrong = false, size = S } = {}) {
  return h('div', { class: `img-cell${wrong ? ' wrong' : ''}` }, drawGray(h('canvas'), pixels, size), caption);
}

// A mouse/touch drawing pad. onChange(imageData) fires after each stroke.
export function drawPad(onChange, { size = 168, brush = 13 } = {}) {
  const canvas = h('canvas', { width: size, height: size, style: { width: `${size}px`, height: `${size}px`, background: '#000', borderRadius: '10px', touchAction: 'none', cursor: 'crosshair' } });
  const ctx = canvas.getContext('2d');
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = brush;
  let drawing = false;
  const pos = (e) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) * (size / r.width), (e.clientY - r.top) * (size / r.height)]; };
  canvas.addEventListener('pointerdown', (e) => { drawing = true; canvas.setPointerCapture(e.pointerId); const [x, y] = pos(e); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.1, y); ctx.stroke(); });
  canvas.addEventListener('pointermove', (e) => { if (!drawing) return; const [x, y] = pos(e); ctx.lineTo(x, y); ctx.stroke(); });
  const end = () => { if (!drawing) return; drawing = false; onChange(ctx.getImageData(0, 0, size, size)); };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointerleave', end);
  return { canvas, clear() { ctx.clearRect(0, 0, size, size); } };
}

