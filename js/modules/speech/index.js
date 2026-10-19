// Module 8 — Sound classification from spectrograms (a stand-in for keyword spotting).
import { h, pct, probBars } from '../../ui/dom.js';
import { heatmap } from '../../ui/chart.js';
import { mulberry32 } from '../../core/rng.js';
import { confusionMatrix } from '../../core/metrics.js';
import { KINDS, synthClip, features } from '../../shared/audio.js';
import { playClip, waveCanvas, specCanvas, recordMic } from '../../ui/sound.js';
import { makeData, createSession, architecture } from './model.js';
import { useCases, USE_TEXT } from './uses.js';

export const kindLabel = (ctx, k) => ctx.t(`k.${KINDS[k] ?? k}`);

function clipCard(ctx, clip, title) {
  return h('div', { class: 'stat', style: { display: 'grid', gap: '6px', justifyItems: 'start' } },
    h('div', { class: 'row', style: { justifyContent: 'space-between', width: '100%' } }, h('b', { class: 'small' }, title),
      h('button', { class: 'btn btn-sm', type: 'button', onclick: () => playClip(clip) }, '▶')),
    waveCanvas(clip), specCanvas(features(clip)));
}

export default {
  id: 'speech',
  title: { ko: '소리 인식', en: 'Sound recognition' },
  summary: {
    ko: '소리를 "시간-높이 그림(스펙트로그램)"으로 바꾼 뒤 CNN으로 분류합니다. 음성 명령 인식("헤이 ○○")의 기본 원리입니다.',
    en: 'Turns sound into a time-by-pitch picture (spectrogram) and classifies it with a CNN — the basic idea behind wake words like "Hey ○○".',
  },
  archHint: 'Sound → Spectrogram → Conv → Pool → Dense → Softmax(6)',
  synthetic: true,
  defaults: { seed: 1, n: 1200, filters: 8, dense: 32, lr: 0.003, epochs: 8, batchSize: 32 },
  cpuDefaults: { n: 300, filters: 4, epochs: 10 },
  dataKeys: ['seed', 'n'],
  controls: [
    { key: 'filters', type: 'select', label: 'c.filters', where: 'arch', options: [4, 8, 16].map((v) => ({ value: v, label: String(v) })) },
    { key: 'dense', type: 'select', label: 'c.dense', where: 'arch', options: [16, 32, 64].map((v) => ({ value: v, label: String(v) })) },
    { key: 'n', type: 'range', min: 60, max: 2400, step: 60, label: 'c.n', where: 'data' },
    { key: 'lr', type: 'range', min: -4, max: 0.5, step: 0.1, log: true, label: 'hp.lr', where: 'train' },
    { key: 'epochs', type: 'range', min: 2, max: 80, label: 'hp.epochs', where: 'train' },
    { key: 'batchSize', type: 'range', min: 8, max: 128, step: 8, label: 'hp.batchSize', where: 'train' },
  ],
  breakPresets: [
    { id: 'overfit', label: 'p.overfit', hp: { n: 24, epochs: 80, dense: 64, filters: 8 }, expect: 'overfitting' },
    { id: 'diverge', label: 'p.diverge', hp: { lr: 1 }, expect: 'diverged' },
  ],
  metrics: [
    { key: 'train_acc', label: 'metric.trainAccuracy', format: 'pct' },
    { key: 'acc', label: 'metric.valAccuracy', format: 'pct' },
  ],
  testMetrics: [
    { key: 'acc', label: 'metric.testAccuracy', format: 'pct', sub: 'm.chance' },
    { key: 'loss', label: 'metric.testLoss', format: 'num' },
  ],
  baselineLoss: () => Math.log(KINDS.length),
  architecture,
  makeData,
  createSession,
  useCases,

  renderData(el, data, ctx) {
    const first = KINDS.map((_, k) => data.train.y.indexOf(k));
    el.append(
      h('p', { class: 'help' }, ctx.t('m.dataIntro')),
      h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: '10px' } },
        first.map((i, k) => clipCard(ctx, data.train.clips[i], kindLabel(ctx, k)))),
      h('p', { class: 'small muted', style: { marginTop: '10px' } }, ctx.t('m.specHelp')));
  },

  renderTrainViz(el, ctx) {
    const sample = KINDS.map((_, k) => ctx.data.val.y.indexOf(k)).filter((i) => i >= 0);
    const grid = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: '10px' } });
    el.append(h('p', { class: 'help' }, ctx.t('m.vizIntro')), grid);
    return {
      update() {
        const s = ctx.session;
        const probs = s ? s.predictSpecs(sample.map((i) => ctx.data.val.x[i])) : null;
        grid.replaceChildren(...sample.map((i, j) => {
          const k = ctx.data.val.y[i];
          const p = probs?.[j];
          const guess = p ? p.indexOf(Math.max(...p)) : -1;
          return h('div', { class: 'stat', style: { display: 'grid', gap: '4px' } },
            h('b', { class: 'small' }, kindLabel(ctx, k)), specCanvas(ctx.data.val.x[i], { width: 150, height: 48 }),
            p ? h('span', { class: `small ${guess === k ? 'tag-good' : 'tag-bad'}` }, `→ ${kindLabel(ctx, guess)} ${pct(p[guess])}`) : null);
        }));
      },
    };
  },

  async diagnosePanel(el, ctx) {
    const { val } = ctx.data;
    const probs = ctx.session.predictSpecs(val.x);
    const pred = probs.map((p) => p.indexOf(Math.max(...p)));
    const cm = confusionMatrix(val.y, pred, KINDS.length);
    const canvas = h('canvas');
    const labels = KINDS.map((_, k) => kindLabel(ctx, k));
    el.append(h('div', { class: 'card' }, h('h3', null, ctx.t('m.confusion')), h('p', { class: 'help' }, ctx.t('m.confusionHelp')), h('div', { class: 'chart-wrap', style: { maxWidth: '520px' } }, canvas),
      h('p', { class: 'small muted' }, labels.map((l, i) => `${i}: ${l}`).join(' · '))));
    heatmap(canvas, cm, { rowLabels: KINDS.map((_, i) => String(i)), colLabels: KINDS.map((_, i) => String(i)), showValues: true, height: 230 });
    let worst = null;
    for (let a = 0; a < KINDS.length; a++) for (let b = 0; b < KINDS.length; b++) if (a !== b && cm[a][b] && (!worst || cm[a][b] > worst.n)) worst = { a: labels[a], b: labels[b], n: cm[a][b] };
    return worst ? [{ level: 'info', key: 'm.worstPair', vars: worst }] : [{ level: 'good', key: 'm.noConfusion' }];
  },

  renderTest(el, ctx) {
    const s = ctx.session;
    const { test } = ctx.data;
    let round = 0;
    let you = 0;
    let model = 0;
    let current = 0;
    const score = h('div', { class: 'big-result' });
    const reveal = h('div');
    const pickRow = h('div', { class: 'row', style: { gap: '6px' } });
    const rand = mulberry32(ctx.hp.seed + 404);
    const next = () => {
      current = Math.floor(rand() * test.clips.length);
      reveal.replaceChildren(h('span', { class: 'muted small' }, ctx.t('m.listenFirst')));
      pickRow.querySelectorAll('button').forEach((b) => { b.disabled = false; });
      playClip(test.clips[current]);
    };
    const guess = (k) => {
      const p = s.predictSpecs([test.x[current]])[0];
      const mk = p.indexOf(Math.max(...p));
      const truth = test.y[current];
      round++;
      if (k === truth) you++;
      if (mk === truth) model++;
      pickRow.querySelectorAll('button').forEach((b) => { b.disabled = true; });
      reveal.replaceChildren(h('div', { class: 'row', style: { gap: '16px', alignItems: 'center' } }, specCanvas(test.x[current]),
        h('div', null,
          h('div', null, ctx.t('m.truth'), ': ', h('b', null, kindLabel(ctx, truth))),
          h('div', { class: k === truth ? 'tag-good' : 'tag-bad' }, ctx.t('m.you'), ': ', kindLabel(ctx, k)),
          h('div', { class: mk === truth ? 'tag-good' : 'tag-bad' }, ctx.t('m.model'), ': ', kindLabel(ctx, mk), ` (${pct(p[mk])})`))));
      score.textContent = ctx.t('m.score', { you, model, n: round });
    };
    pickRow.append(...KINDS.map((_, k) => h('button', { class: 'chip', type: 'button', onclick: () => guess(k) }, kindLabel(ctx, k))));

    // Microphone (optional).
    const micOut = h('div', { class: 'small' });
    const mic = async () => {
      micOut.textContent = ctx.t('m.recording');
      try {
        const clip = await recordMic(0.5);
        const p = s.predictClips([clip])[0];
        micOut.replaceChildren(h('div', { class: 'row', style: { gap: '12px', alignItems: 'center' } }, specCanvas(features(clip)),
          h('div', { style: { minWidth: '220px' } }, probBars(KINDS.map((_, k) => kindLabel(ctx, k)), p))));
      } catch {
        micOut.replaceChildren(h('span', { class: 'tag-warn' }, ctx.t('m.micDenied')));
      }
    };
    el.append(
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.gameTitle')), h('p', { class: 'help' }, ctx.t('m.gameHelp')),
        h('div', { class: 'row' }, h('button', { class: 'btn btn-primary', type: 'button', onclick: next }, ctx.t('m.play')),
          h('button', { class: 'btn', type: 'button', onclick: () => playClip(test.clips[current]) }, ctx.t('m.replay'))),
        h('div', { style: { margin: '10px 0' } }, pickRow), reveal, score),
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.micTitle')), h('p', { class: 'help' }, ctx.t('m.micHelp')),
        h('div', null, h('button', { class: 'btn', type: 'button', onclick: mic }, `🎤 ${ctx.t('m.record')}`)), micOut),
      h('div', { class: 'card' }, h('h3', null, ctx.t('m.makeTitle')), h('p', { class: 'help' }, ctx.t('m.makeHelp')),
        h('div', { class: 'row', style: { gap: '6px' } }, KINDS.map((kind, k) => h('button', {
          class: 'chip', type: 'button',
          onclick: (e) => {
            const clip = synthClip(kind, mulberry32(Date.now() % 100000));
            playClip(clip);
            const p = s.predictClips([clip])[0];
            const card = e.target.closest('.card');
            card.querySelector('.made')?.remove();
            card.append(h('div', { class: 'made', style: { marginTop: '10px' } }, probBars(KINDS.map((_, j) => kindLabel(ctx, j)), p, { highlight: k })));
          },
        }, kindLabel(ctx, k))))));
  },

  text: {
    ko: {
      ...USE_TEXT.ko,
      'k.tone_low': '낮은 음',
      'k.tone_high': '높은 음',
      'k.chirp_up': '올라가는 소리',
      'k.chirp_down': '내려가는 소리',
      'k.two_beeps': '삑삑 두 번',
      'k.noise': '잡음',
      'c.filters': '첫 합성곱층 필터 수',
      'c.dense': 'Dense 층 뉴런 수',
      'c.n': '소리 개수',
      'p.overfit': '소리 24개로 80에폭 → 과적합',
      'p.diverge': '학습률 1 → 발산',
      'm.chance': '무작위로 찍으면 약 17%',
      'm.dataIntro': '여섯 종류의 짧은 소리입니다. ▶를 눌러 들어 보세요. 위는 파형(시간에 따른 떨림), 아래는 스펙트로그램(가로=시간, 세로=높이, 밝을수록 셈)입니다. 같은 종류도 높이·시작 시점·크기·배경 잡음이 매번 다릅니다.',
      'm.specHelp': '사람 귀에는 비슷하게 들려도 스펙트로그램에서는 모양이 뚜렷이 다릅니다: 올라가는 소리는 위로 비스듬한 선, 삑삑 두 번은 끊어진 선 두 개입니다.',
      'm.vizIntro': '종류별 검증 소리 하나씩에 대한 모델의 현재 답입니다.',
      'm.confusion': '혼동 행렬 (검증 데이터)',
      'm.confusionHelp': '행은 정답, 열은 예측입니다. 대각선에 몰려 있을수록 좋습니다.',
      'm.worstPair': '가장 많이 헷갈린 경우: "{a}"를 "{b}"(으)로 {n}번 착각했습니다.',
      'm.noConfusion': '검증 데이터에서 헷갈린 소리가 없습니다.',
      'm.gameTitle': '사람 vs 모델: 듣고 맞히기',
      'm.gameHelp': '재생 버튼을 누르고 소리를 들은 뒤 종류를 고르세요. 모델도 같은 소리를 맞힙니다.',
      'm.play': '새 소리 듣기',
      'm.replay': '다시 듣기',
      'm.listenFirst': '소리를 듣고 아래에서 골라 주세요.',
      'm.truth': '정답',
      'm.you': '나',
      'm.model': '모델',
      'm.score': '{n}판 중 나 {you}번, 모델 {model}번 맞힘',
      'm.micTitle': '마이크로 시험해 보기 (선택)',
      'm.micHelp': '버튼을 누르면 0.5초 동안 녹음합니다. 휘파람을 올려 불거나, 짧게 두 번 "삑" 소리를 내 보세요. 실제 목소리는 학습하지 않았으니 엉뚱한 답이 나와도 정상입니다.',
      'm.record': '0.5초 녹음',
      'm.recording': '녹음 중…',
      'm.micDenied': '마이크를 쓸 수 없습니다(권한 거부 또는 미지원). 위의 "듣고 맞히기"와 아래 "소리 만들기"는 그대로 쓸 수 있어요.',
      'm.makeTitle': '새 소리 만들어 시험하기',
      'm.makeHelp': '버튼을 누르면 그 종류의 새 소리를 즉석에서 만들어 들려주고, 모델의 판단을 보여 줍니다.',
    },
    en: {
      ...USE_TEXT.en,
      'k.tone_low': 'Low tone',
      'k.tone_high': 'High tone',
      'k.chirp_up': 'Rising chirp',
      'k.chirp_down': 'Falling chirp',
      'k.two_beeps': 'Double beep',
      'k.noise': 'Noise',
      'c.filters': 'Filters in first conv layer',
      'c.dense': 'Dense layer neurons',
      'c.n': 'Number of sounds',
      'p.overfit': '24 sounds for 80 epochs → overfitting',
      'p.diverge': 'Learning rate 1 → diverges',
      'm.chance': 'Random guessing: about 17%',
      'm.dataIntro': 'Six kinds of short sounds. Press ▶ to listen. Top: waveform (vibration over time). Bottom: spectrogram (across = time, up = pitch, brighter = louder). Each copy differs in pitch, timing, loudness and background noise.',
      'm.specHelp': 'Sounds that seem alike to the ear look clearly different in a spectrogram: a rising chirp is a slanted line, a double beep is two separate dashes.',
      'm.vizIntro': 'The model\'s current answer for one validation sound of each kind.',
      'm.confusion': 'Confusion matrix (validation data)',
      'm.confusionHelp': 'Rows are the true kind, columns the prediction. The more on the diagonal, the better.',
      'm.worstPair': 'Most common mix-up: "{a}" heard as "{b}", {n} time(s).',
      'm.noConfusion': 'No sounds were mixed up on the validation data.',
      'm.gameTitle': 'You vs. the model: listen and guess',
      'm.gameHelp': 'Press play, listen, then pick the kind. The model guesses the same sound.',
      'm.play': 'Play a new sound',
      'm.replay': 'Replay',
      'm.listenFirst': 'Listen, then choose below.',
      'm.truth': 'Answer',
      'm.you': 'You',
      'm.model': 'Model',
      'm.score': 'After {n} round(s): you {you}, model {model}',
      'm.micTitle': 'Try your microphone (optional)',
      'm.micHelp': 'Records half a second. Try a rising whistle or two short beeps. It never learned real voices, so odd answers are expected.',
      'm.record': 'Record 0.5 s',
      'm.recording': 'Recording…',
      'm.micDenied': 'The microphone is not available (permission denied or unsupported). "Listen and guess" above and "Make a sound" below still work.',
      'm.makeTitle': 'Make a new sound and test it',
      'm.makeHelp': 'Each button makes a brand-new sound of that kind, plays it, and shows the model\'s judgement.',
    },
  },
};
