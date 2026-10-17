// Small template-generated corpora for the language modules, plus character-level helpers.
// Sentences pair objects with matching verbs ("milk" → "drink"), so there are real patterns to learn.
import { mulberry32 } from '../core/rng.js';
import { splitIndices } from '../core/split.js';

const KO = {
  time: ['아침에', '저녁에', '오늘', '매일', '주말에'],
  subj: ['고양이는', '강아지는', '아이는', '할머니는', '선생님은', '친구는', '토끼는'],
  place: ['집에서', '공원에서', '학교에서', '바다에서', '숲에서'],
  pair: [['사과를', '먹는다'], ['우유를', '마신다'], ['책을', '읽는다'], ['공을', '던진다'], ['꽃을', '본다'], ['빵을', '만든다'], ['노래를', '부른다']],
};

const EN = {
  time: ['today', 'every day', 'at night', 'in the morning'],
  subj: ['the cat', 'the dog', 'a child', 'my friend', 'the teacher', 'a rabbit'],
  place: ['at home', 'in the park', 'at school', 'by the sea'],
  pair: [['eats', 'an apple'], ['drinks', 'some milk'], ['reads', 'a book'], ['throws', 'a ball'], ['sees', 'a flower'], ['bakes', 'some bread'], ['sings', 'a song']],
};

const pick = (arr, rand) => arr[Math.floor(rand() * arr.length)];

export function makeSentence(lang, rand) {
  if (lang === 'ko') {
    const [obj, verb] = pick(KO.pair, rand);
    return `${pick(KO.time, rand)} ${pick(KO.subj, rand)} ${pick(KO.place, rand)} ${obj} ${verb}.`;
  }
  const [verb, obj] = pick(EN.pair, rand);
  return `${pick(EN.subj, rand)} ${verb} ${obj} ${pick(EN.place, rand)} ${pick(EN.time, rand)}.`;
}

// Generate sentences (seeded) until the text reaches `chars` characters.
export function makeCorpus(lang, chars, seed = 42) {
  const rand = mulberry32(seed);
  const out = [];
  let len = 0;
  while (len < chars) {
    const s = makeSentence(lang, rand);
    out.push(s);
    len += s.length + 1;
  }
  return out;
}

export function buildVocab(text) {
  const itos = [...new Set(text)].sort();
  const stoi = Object.fromEntries(itos.map((c, i) => [c, i]));
  return { stoi, itos };
}

export function encode(text, stoi) {
  return [...text].map((c) => stoi[c] ?? -1).filter((i) => i >= 0);
}

/**
 * Character-level next-token data. Sentences are split 80/10/10 *by sentence* so no window spans
 * two splits; each split becomes windows of seqLen ids (x) and the id that follows (y).
 * Reserve extra special tokens in the vocab with `specials` (e.g. ['[MASK]']).
 */
export function makeTextData({ lang, chars, seqLen, seed, fraction = 1, stride = 1, specials = [] }) {
  const sentences = makeCorpus(lang, chars, 42);
  const rand = mulberry32(seed);
  const idx = splitIndices(sentences.length, [0.8, 0.1, 0.1], rand);
  const keep = Math.max(2, Math.round(idx.train.length * fraction));
  const parts = { train: idx.train.slice(0, keep), val: idx.val, test: idx.test };
  const full = sentences.join(' ');
  const { itos } = buildVocab(full);
  for (const s of specials) itos.push(s);
  const stoi = Object.fromEntries(itos.map((c, i) => [c, i]));
  const out = { vocab: { itos, stoi }, sentences: {}, meta: { counts: {} } };
  for (const [name, ids] of Object.entries(parts)) {
    const text = ids.map((i) => sentences[i]).join(' ');
    const enc = encode(text, stoi);
    const x = [];
    const y = [];
    for (let i = 0; i + seqLen < enc.length; i += stride) {
      x.push(enc.slice(i, i + seqLen));
      y.push(enc[i + seqLen]);
    }
    out[name] = { x, y, text };
    out.sentences[name] = ids.map((i) => sentences[i]);
    out.meta.counts[name] = x.length;
  }
  return out;
}

// Sample an index from probabilities with a temperature, using rand().
export function sampleIndex(probs, temperature, rand) {
  if (temperature <= 0.01) return probs.indexOf(Math.max(...probs));
  const logits = probs.map((p) => Math.log(Math.max(p, 1e-12)) / temperature);
  const m = Math.max(...logits);
  const e = logits.map((l) => Math.exp(l - m));
  const sum = e.reduce((a, b) => a + b, 0);
  let r = rand() * sum;
  for (let i = 0; i < e.length; i++) {
    r -= e[i];
    if (r <= 0) return i;
  }
  return e.length - 1;
}
