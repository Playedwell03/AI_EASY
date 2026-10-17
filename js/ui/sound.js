// Browser-side sound helpers: play a clip, draw waveform / spectrogram, record from the microphone.
import { h, cssVar } from './dom.js';
import { SR, resample } from '../shared/audio.js';

let audioCtx = null;

export function playClip(clip) {
  try {
    audioCtx ??= new (window.AudioContext || window.webkitAudioContext)();
    const buf = audioCtx.createBuffer(1, clip.length, SR);
    buf.copyToChannel(clip, 0);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start();
  } catch {
    // No audio output available; the pictures still tell the story.
  }
}

export function waveCanvas(clip, { width = 160, height = 40 } = {}) {
  const c = h('canvas', { width, height, style: { width: `${width}px`, height: `${height}px`, display: 'block' } });
  const ctx = c.getContext('2d');
  ctx.strokeStyle = cssVar('--series-1');
  ctx.beginPath();
  const step = clip.length / width;
  for (let x = 0; x < width; x++) {
    let lo = 1;
    let hi = -1;
    for (let i = Math.floor(x * step); i < (x + 1) * step; i++) { lo = Math.min(lo, clip[i]); hi = Math.max(hi, clip[i]); }
    ctx.moveTo(x + 0.5, (1 - hi) * height / 2);
    ctx.lineTo(x + 0.5, (1 - lo) * height / 2);
  }
  ctx.stroke();
  return c;
}

// spec: [frames][bins] in [0,1] → image with time across and low pitch at the bottom.
export function specCanvas(spec, { width = 160, height = 64 } = {}) {
  const frames = spec.length;
  const bins = spec[0].length;
  const c = h('canvas', { width: frames, height: bins, style: { width: `${width}px`, height: `${height}px`, imageRendering: 'pixelated', display: 'block', borderRadius: '4px' } });
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(frames, bins);
  for (let t = 0; t < frames; t++) {
    for (let b = 0; b < bins; b++) {
      const v = Math.max(0, Math.min(1, spec[t][b]));
      const k = ((bins - 1 - b) * frames + t) * 4;
      // Dark blue → orange → yellow, readable in both themes.
      img.data[k] = Math.round(255 * Math.min(1, v * 1.6));
      img.data[k + 1] = Math.round(255 * Math.max(0, v * 1.4 - 0.4));
      img.data[k + 2] = Math.round(90 * (1 - v) + 30);
      img.data[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// Record `seconds` from the microphone and return SR-rate samples. Throws if not allowed/available.
export async function recordMic(seconds = 0.5) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    audioCtx ??= new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    const need = Math.ceil(seconds * audioCtx.sampleRate) + 4096;
    await new Promise((resolve) => {
      let got = 0;
      proc.onaudioprocess = (e) => {
        const d = e.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(d));
        got += d.length;
        if (got >= need) resolve();
      };
      src.connect(proc);
      proc.connect(audioCtx.destination);
    });
    proc.disconnect();
    src.disconnect();
    const all = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
    let o = 0;
    for (const c of chunks) { all.set(c, o); o += c.length; }
    return resample(all.subarray(2048), audioCtx.sampleRate, SR, SR * seconds);
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
}
