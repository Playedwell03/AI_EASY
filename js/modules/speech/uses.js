// "Use" tab apps for the sound classifier.
import { h, pct } from '../../ui/dom.js';
import { mulberry32 } from '../../core/rng.js';
import { KINDS, synthClip, features } from '../../shared/audio.js';
import { playClip, specCanvas } from '../../ui/sound.js';
import { step, lt } from '../../ui/useView.js';

const K = Object.fromEntries(KINDS.map((k, i) => [k, i]));
const label = (ctx, k) => ctx.t(`k.${KINDS[k]}`);

// Act only when the top class has an action and the model is sure enough.
export function decideAction(probs, mapping, minConf) {
  const k = probs.indexOf(Math.max(...probs));
  if (probs[k] < minConf) return null;
  return mapping[k] ?? null;
}

// True when at least `count` of the last `window` flags are set.
export function sustained(flags, window, count) {
  return flags.slice(-window).filter(Boolean).length >= count && flags.length > 0;
}

const PIPE = (input, decide, out) => [
  step('input', input.ko, input.en, '마이크로 0.5초 길이의 소리가 들어옵니다(여기서는 합성 소리).', 'Half a second of sound arrives from the microphone (synthetic here).'),
  step('pre', '스펙트로그램', 'Spectrogram', '소리를 시간-높이 그림으로 바꿉니다.', 'Turns the sound into a time-by-pitch picture.'),
  step('model', '소리 종류 판별', 'Classify sound', '학습한 CNN이 여섯 종류 각각의 확률을 냅니다.', 'Your trained CNN gives a probability for each of the six kinds.'),
  step('post', decide.ko, decide.en, decide.dko, decide.den),
  step('output', out.ko, out.en, '결과가 화면과 기기에 반영됩니다.', 'The result shows on screen and in the device.'),
];

function soundButtons(ctx, onSound, kinds = KINDS.map((_, i) => i)) {
  let n = 0;
  return h('div', { class: 'row', style: { gap: '6px' } }, kinds.map((k) => h('button', {
    class: 'chip', type: 'button',
    onclick: () => { const clip = synthClip(KINDS[k], mulberry32(ctx.hp.seed * 1000 + n++)); playClip(clip); onSound(clip, k); },
  }, `🔊 ${label(ctx, k)}`)));
}

const home = {
  id: 'home',
  icon: '🏠',
  title: lt('스마트홈 소리 알림', 'Smart-home sound alerts'),
  intro: lt('집 안 소리를 듣고 초인종·화재경보 같은 중요한 소리만 휴대폰으로 알려 줍니다.', 'Listens at home and only notifies your phone about important sounds like the doorbell or a fire alarm.'),
  realWorld: lt('청각장애인용 알림 기기, 스마트 스피커의 "소리 감지" 기능이 같은 방식입니다. 실제로는 수백 가지 생활 소리로 학습합니다.', 'Alert devices for deaf people and the "sound detection" in smart speakers work this way, trained on hundreds of real household sounds.'),
  pipeline: PIPE(lt('집 안 소리', 'Household sound'), { ko: '알림 규칙', en: 'Alert rules', dko: '"삑삑 두 번"=초인종, "올라가는 소리"=화재경보처럼 정해 두고, 확신이 70% 이상일 때만 알립니다.', den: 'Maps "double beep" to doorbell and "rising chirp" to fire alarm, and only notifies when at least 70% sure.' }, lt('휴대폰 알림', 'Phone notification')),
  render(el, ctx, flow) {
    const map = { [K.two_beeps]: 'door', [K.chirp_up]: 'fire', [K.tone_high]: 'kettle' };
    const feed = h('div', { class: 'stack', style: { gap: '6px', maxHeight: '260px', overflowY: 'auto' } });
    el.append(h('p', { class: 'help' }, ctx.t('use.sp.homeHelp')),
      soundButtons(ctx, (clip, truth) => {
        flow.run();
        const p = ctx.session.predictClips([clip])[0];
        const act = decideAction(p, map, 0.7);
        const k = p.indexOf(Math.max(...p));
        feed.prepend(h('div', { class: 'mail-row' }, h('span', null, act ? { door: '🔔', fire: '🔥', kettle: '☕' }[act] : '·'),
          h('span', null, act ? h('b', null, ctx.t(`use.sp.${act}`)) : h('span', { class: 'muted' }, ctx.t('use.sp.ignored')),
            h('span', { class: 'small muted' }, `  (${ctx.t('use.sp.heard')}: ${label(ctx, k)} ${pct(p[k])} · ${ctx.t('use.sp.actual')}: ${label(ctx, truth)})`)),
          specCanvas(features(clip), { width: 60, height: 24 })));
      }),
      h('b', null, ctx.t('use.sp.phone')), feed);
  },
};

const remote = {
  id: 'remote',
  icon: '📺',
  title: lt('소리 리모컨', 'Sound remote control'),
  intro: lt('정해진 소리를 내면 TV가 볼륨·채널을 바꿉니다.', 'Make a set sound and the TV changes volume or channel.'),
  realWorld: lt('음성 명령 기기는 먼저 작은 모델로 "호출어"만 계속 듣다가, 들리면 큰 모델을 깨워 명령을 이해합니다. 여기 모델이 그 작은 "호출어 감지기" 역할입니다.', 'Voice devices keep a tiny model listening for a wake word, then wake a bigger model to understand the command. Your model plays that tiny wake-word detector role.'),
  pipeline: PIPE(lt('명령 소리', 'Command sound'), { ko: '명령 매핑', en: 'Map to command', dko: '올라가는 소리=볼륨 업, 내려가는 소리=볼륨 다운, 삑삑=채널 변경. 60% 미만이면 무시합니다.', den: 'Rising = volume up, falling = volume down, double beep = next channel. Ignored under 60%.' }, lt('TV 동작', 'TV reacts')),
  render(el, ctx, flow) {
    const map = { [K.chirp_up]: 'up', [K.chirp_down]: 'down', [K.two_beeps]: 'ch' };
    let vol = 10;
    let ch = 1;
    const tv = h('div', { style: { background: '#111', color: '#eee', borderRadius: '12px', padding: '18px', fontFamily: 'var(--mono)', minHeight: '110px', display: 'grid', gap: '6px', maxWidth: '360px' } });
    const draw = (msg) => { tv.replaceChildren(h('div', null, `📺 CH ${ch}`), h('div', null, `🔊 ${'█'.repeat(vol)}${'░'.repeat(20 - vol)} ${vol}`), h('div', { style: { color: '#ffc' } }, msg || '')); };
    el.append(h('p', { class: 'help' }, ctx.t('use.sp.remoteHelp')),
      soundButtons(ctx, (clip) => {
        flow.run();
        const p = ctx.session.predictClips([clip])[0];
        const act = decideAction(p, map, 0.6);
        if (act === 'up') vol = Math.min(20, vol + 2);
        if (act === 'down') vol = Math.max(0, vol - 2);
        if (act === 'ch') ch = (ch % 9) + 1;
        const k = p.indexOf(Math.max(...p));
        draw(act ? ctx.t(`use.sp.cmd.${act}`) : ctx.t('use.sp.noCmd', { k: label(ctx, k), p: pct(p[k]) }));
      }, [K.chirp_up, K.chirp_down, K.two_beeps, K.tone_low, K.noise]),
      tv);
    draw(ctx.t('use.sp.tvReady'));
  },
};

const machine = {
  id: 'machine',
  icon: '⚙️',
  title: lt('기계 소리 모니터링', 'Machine sound monitoring'),
  intro: lt('공장 기계가 내는 소리를 계속 듣다가, 평소와 다른 잡음이 이어지면 경보를 울립니다.', 'Listens to a machine continuously and sounds the alarm when unusual noise keeps coming.'),
  realWorld: lt('실제 설비 예지보전은 진동·소리 센서로 고장 전 징후를 찾습니다. 한 번의 판단 실수로 경보가 울리지 않도록 "최근 여러 번 중 몇 번" 규칙을 함께 씁니다.', 'Real predictive maintenance listens with vibration and sound sensors for early signs of failure, and uses "k out of the last n" rules so one misjudgement does not trigger an alarm.'),
  pipeline: PIPE(lt('기계 소리 (0.5초마다)', 'Machine sound (every 0.5 s)'), { ko: '연속 판단', en: 'Sustained check', dko: '"잡음"으로 판단된 소리가 최근 5번 중 3번 이상이면 경보를 울립니다.', den: 'Alarms when "noise" is detected in at least 3 of the last 5 sounds.' }, lt('경보등', 'Alarm light')),
  render(el, ctx, flow) {
    const flags = [];
    const strip = h('div', { class: 'row', style: { gap: '3px' } });
    const lamp = h('div', { class: 'big-result' });
    let t = 0;
    let broken = false;
    const rand = mulberry32(ctx.hp.seed + 77);
    const tick = (n) => {
      for (let i = 0; i < n; i++) {
        const normal = broken ? rand() < 0.25 : rand() < 0.93;
        const kind = normal ? K.tone_low : K.noise;
        const p = ctx.session.predictClips([synthClip(KINDS[kind], mulberry32(ctx.hp.seed * 7 + t++))])[0];
        const isNoise = p.indexOf(Math.max(...p)) === K.noise;
        flags.push(isNoise);
        strip.append(h('span', { title: `${label(ctx, kind)} → ${pct(p[K.noise])}`, style: { width: '10px', height: '22px', borderRadius: '2px', background: isNoise ? 'var(--bad)' : 'var(--good)' } }));
      }
      while (strip.children.length > 60) strip.firstChild.remove();
      flow.run();
      const alarm = sustained(flags, 5, 3);
      lamp.replaceChildren(alarm ? h('span', { class: 'tag-bad' }, `🚨 ${ctx.t('use.sp.alarm')}`) : h('span', { class: 'tag-good' }, `🟢 ${ctx.t('use.sp.running')}`));
    };
    el.append(h('p', { class: 'help' }, ctx.t('use.sp.machineHelp')),
      h('div', { class: 'row' },
        h('button', { class: 'btn btn-primary', type: 'button', onclick: () => tick(10) }, ctx.t('use.sp.listen10')),
        h('button', { class: 'btn btn-danger', type: 'button', onclick: (e) => { broken = !broken; e.target.textContent = ctx.t(broken ? 'use.sp.fix' : 'use.sp.break'); } }, ctx.t('use.sp.break'))),
      strip, lamp, h('p', { class: 'small muted' }, ctx.t('use.sp.stripHelp')));
    tick(20);
  },
};

export const useCases = [home, remote, machine];

export const USE_TEXT = {
  ko: {
    'use.sp.homeHelp': '아래 버튼으로 집 안에서 나는 소리를 흉내 내 보세요. 중요한 소리만 휴대폰 알림으로 갑니다.',
    'use.sp.door': '누군가 초인종을 눌렀어요',
    'use.sp.fire': '화재경보가 울려요!',
    'use.sp.kettle': '주전자 물이 끓어요',
    'use.sp.ignored': '알릴 필요 없는 소리',
    'use.sp.heard': '들은 것',
    'use.sp.actual': '실제',
    'use.sp.phone': '휴대폰 알림',
    'use.sp.remoteHelp': '올라가는 소리 = 볼륨 업, 내려가는 소리 = 볼륨 다운, 삑삑 두 번 = 다음 채널. 다른 소리는 무시해야 합니다.',
    'use.sp.cmd.up': '볼륨 업',
    'use.sp.cmd.down': '볼륨 다운',
    'use.sp.cmd.ch': '다음 채널',
    'use.sp.noCmd': '명령 아님 ({k} {p})',
    'use.sp.tvReady': '명령을 기다리는 중',
    'use.sp.machineHelp': '0.5초마다 기계 소리를 듣고 초록(정상)·빨강(잡음)으로 판단합니다. "고장 내기"를 누르면 잡음이 많아집니다.',
    'use.sp.listen10': '5초 더 듣기',
    'use.sp.break': '고장 내기',
    'use.sp.fix': '수리하기',
    'use.sp.alarm': '이상 소리가 계속됩니다 — 점검 필요',
    'use.sp.running': '정상 가동 중',
    'use.sp.stripHelp': '막대 하나 = 소리 하나. 가끔 빨강이 하나 섞여도 경보는 울리지 않습니다(최근 5개 중 3개 규칙).',
  },
  en: {
    'use.sp.homeHelp': 'Use the buttons to imitate sounds at home. Only important ones reach your phone.',
    'use.sp.door': 'Someone rang the doorbell',
    'use.sp.fire': 'Fire alarm!',
    'use.sp.kettle': 'The kettle is boiling',
    'use.sp.ignored': 'Nothing worth notifying',
    'use.sp.heard': 'heard',
    'use.sp.actual': 'actual',
    'use.sp.phone': 'Phone notifications',
    'use.sp.remoteHelp': 'Rising = volume up, falling = volume down, double beep = next channel. Other sounds should be ignored.',
    'use.sp.cmd.up': 'Volume up',
    'use.sp.cmd.down': 'Volume down',
    'use.sp.cmd.ch': 'Next channel',
    'use.sp.noCmd': 'Not a command ({k} {p})',
    'use.sp.tvReady': 'Waiting for a command',
    'use.sp.machineHelp': 'Every half second it listens and marks green (normal) or red (noise). Press "Break it" to make noise more common.',
    'use.sp.listen10': 'Listen 5 more seconds',
    'use.sp.break': 'Break it',
    'use.sp.fix': 'Repair',
    'use.sp.alarm': 'Unusual noise keeps coming — inspect now',
    'use.sp.running': 'Running normally',
    'use.sp.stripHelp': 'One bar = one sound. A single red bar does not trigger the alarm (3-of-last-5 rule).',
  },
};
