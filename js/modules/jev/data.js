// Synthetic loan applications with three typed questions: approve? (yes/no), which queue? (choice), risk (score).
import { mulberry32, randn } from '../../core/rng.js';
import { splitIndices } from '../../core/split.js';

export const FIELDS = [
  { key: 'income', type: 'num', min: 20, max: 120, unit: { ko: '백만원/년', en: 'M/yr' }, label: { ko: '연 소득', en: 'Annual income' } },
  { key: 'debt', type: 'num', min: 0, max: 100, unit: { ko: '백만원', en: 'M' }, label: { ko: '기존 부채', en: 'Existing debt' } },
  { key: 'years', type: 'num', min: 0, max: 30, unit: { ko: '년', en: 'yrs' }, label: { ko: '근속 연수', en: 'Years employed' } },
  { key: 'missed', type: 'num', min: 0, max: 6, unit: { ko: '회', en: 'times' }, label: { ko: '연체 횟수', en: 'Missed payments' } },
  { key: 'purpose', type: 'cat', options: [{ ko: '주택', en: 'Home' }, { ko: '사업', en: 'Business' }, { ko: '교육', en: 'Education' }, { ko: '생활비', en: 'Living costs' }], label: { ko: '대출 목적', en: 'Purpose' } },
];

export const ROUTES = [{ ko: '일반 처리', en: 'Standard' }, { ko: '심사역 검토', en: 'Manual review' }, { ko: '거절 안내', en: 'Decline' }];

const PURPOSE_EFFECT = [0.6, -0.4, 0.3, -0.5];

// The hidden rule the model must discover (centered so roughly half are approved).
export function creditScore(f) {
  return 0.045 * (f.income - 60) - 0.04 * (f.debt - 40) + 0.08 * (f.years - 10) - 0.7 * (f.missed - 1.5) + PURPOSE_EFFECT[f.purpose];
}

export function makeLoanCases(n, rand) {
  const fields = [];
  const answers = [];
  const clean = [];
  for (let i = 0; i < n; i++) {
    const f = {
      income: Math.round(20 + rand() * 100),
      debt: Math.round(rand() * 100),
      years: Math.round(rand() ** 1.3 * 30),
      missed: Math.min(6, Math.floor(-Math.log(1 - rand() * 0.999) * 1.3)),
      purpose: Math.floor(rand() * 4),
    };
    const s = creditScore(f) + 0.4 * randn(rand);
    const approveClean = s > 0 ? 1 : 0;
    // 10% of decisions were made differently by people — no model can be 100% sure.
    const approve = rand() < 0.1 ? 1 - approveClean : approveClean;
    const route = s > 1 ? 0 : s > -1 ? 1 : 2;
    const risk = 1 / (1 + Math.exp(s + 0.3 * randn(rand)));
    fields.push(f);
    answers.push({ approve, route, risk });
    clean.push({ approve: approveClean });
  }
  return { fields, answers, clean };
}

export function makeData(hp) {
  const rand = mulberry32(hp.seed);
  const all = makeLoanCases(hp.n, rand);
  const idx = splitIndices(hp.n, [0.7, 0.15, 0.15], rand);
  const pick = (ids) => ({ fields: ids.map((i) => all.fields[i]), answers: ids.map((i) => all.answers[i]) });
  const train = pick(idx.train);
  const val = pick(idx.val);
  const test = pick(idx.test);
  return { train, val, test, meta: { counts: { train: train.fields.length, val: val.fields.length, test: test.fields.length } } };
}

// Loss of a "know nothing" model that always answers with the training base rates.
export function baseRateLoss(data) {
  const a = data.train.answers;
  const p = a.reduce((s, x) => s + x.approve, 0) / a.length;
  const bce = -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
  const freq = [0, 1, 2].map((k) => a.filter((x) => x.route === k).length / a.length);
  const ce = -freq.reduce((s, q) => s + (q ? q * Math.log(q) : 0), 0);
  const mean = a.reduce((s, x) => s + x.risk, 0) / a.length;
  const mse = a.reduce((s, x) => s + (x.risk - mean) ** 2, 0) / a.length;
  return bce + ce + mse;
}
