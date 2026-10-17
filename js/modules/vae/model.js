// Autoencoder / VAE with a 2D latent space, trained with a custom loop.
// Loss per pixel: binary cross-entropy reconstruction + β·KL/pixels (KL only in variational mode).
import { mulberry32, shuffle } from '../../core/rng.js';
import { makeDigitSplits, DIGIT_SIZE } from '../../shared/digits.js';
import { init } from '../../core/tfSession.js';

const P = DIGIT_SIZE * DIGIT_SIZE;
const EPS = 1e-6;

// Calm, clean digits: speckle is unpredictable noise, and large random poses would need more than
// two latent numbers to describe. With less variation the 2D latent can focus on "which digit, what style".
export function makeData(hp) {
  return makeDigitSplits(hp.n, hp.seed, { speckle: false, variation: 0.35 });
}

// Mean per-pixel entropy of the target images. Binary cross-entropy can never go below this,
// so we subtract it: what remains is 0 for a perfect copy.
export function targetEntropy(images) {
  let sum = 0;
  for (const img of images) {
    for (let i = 0; i < P; i++) {
      const p = Math.min(1 - EPS, Math.max(EPS, img[i]));
      sum -= p * Math.log(p) + (1 - p) * Math.log(1 - p);
    }
  }
  return sum / (images.length * P);
}

// Reconstruction error of always predicting the average training image, measured on validation data.
export function meanImageLoss(data) {
  if (data._meanLoss != null) return data._meanLoss;
  const mean = new Float64Array(P);
  for (const img of data.train.x) for (let i = 0; i < P; i++) mean[i] += img[i] / data.train.x.length;
  let sum = 0;
  for (const img of data.val.x) {
    for (let i = 0; i < P; i++) {
      const p = Math.min(1 - EPS, Math.max(EPS, mean[i]));
      sum -= img[i] * Math.log(p) + (1 - img[i]) * Math.log(1 - p);
    }
  }
  data._meanLoss = sum / (data.val.x.length * P) - targetEntropy(data.val.x);
  return data._meanLoss;
}

function flat(tf, images) {
  const buf = new Float32Array(images.length * P);
  images.forEach((img, i) => buf.set(img, i * P));
  return tf.tensor2d(buf, [images.length, P]);
}

export function createSession(tf, hp, data) {
  const encoder = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [P], units: hp.hidden, activation: 'relu', kernelInitializer: init(tf, hp.seed, 0) }),
    tf.layers.dense({ units: 4, kernelInitializer: init(tf, hp.seed, 1) }), // [mu1, mu2, logvar1, logvar2]
  ] });
  const decoder = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [2], units: hp.hidden, activation: 'relu', kernelInitializer: init(tf, hp.seed, 2) }),
    tf.layers.dense({ units: P, activation: 'sigmoid', kernelInitializer: init(tf, hp.seed, 3) }),
  ] });
  const optimizer = tf.train.adam(hp.lr);
  const vars = [...encoder.trainableWeights, ...decoder.trainableWeights].map((w) => w.val);
  const X = { train: flat(tf, data.train.x), val: flat(tf, data.val.x), test: flat(tf, data.test.x) };
  const H = { train: targetEntropy(data.train.x), val: targetEntropy(data.val.x), test: targetEntropy(data.test.x) };
  const rand = mulberry32(hp.seed * 31 + 7);
  const beta = hp.variational ? hp.beta : 0;
  let step = 0;

  // Returns scalar tensors { total, recon, kl } for a batch; noisy=true samples z (training).
  function losses(x, noisy) {
    const out = encoder.apply(x);
    const mu = out.slice([0, 0], [-1, 2]);
    const logvar = hp.variational ? out.slice([0, 2], [-1, 2]).clipByValue(-8, 8) : tf.zerosLike(mu);
    const z = hp.variational && noisy
      ? mu.add(tf.randomNormal(mu.shape, 0, 1, 'float32', hp.seed * 1000 + step++).mul(logvar.mul(0.5).exp()))
      : mu;
    const xr = decoder.apply(z).clipByValue(EPS, 1 - EPS);
    const recon = x.mul(xr.log()).add(tf.scalar(1).sub(x).mul(tf.scalar(1).sub(xr).log())).neg().mean();
    const kl = hp.variational
      ? tf.scalar(1).add(logvar).sub(mu.square()).sub(logvar.exp()).sum(1).mul(-0.5).mean().div(P)
      : tf.scalar(0);
    return { total: recon.add(kl.mul(beta)), recon, kl };
  }

  async function evaluate(split) {
    const t = tf.tidy(() => { const l = losses(X[split], false); return tf.stack([l.total, l.recon, l.kl]); });
    const [total, recon, kl] = await t.data();
    t.dispose();
    // Report KL in nats per image; reconstruction as cross-entropy above the target's own entropy.
    return { total: total - H[split], recon: recon - H[split], kl: kl * P };
  }

  let disposed = false;
  return {
    async epoch() {
      const n = data.train.x.length;
      const order = shuffle(Array.from({ length: n }, (_, i) => i), rand);
      for (let b = 0; b < n; b += hp.batchSize) {
        const ids = tf.tensor1d(order.slice(b, b + hp.batchSize), 'int32');
        const cost = optimizer.minimize(() => {
          const xb = tf.gather(X.train, ids);
          return losses(xb, true).total;
        }, true, vars);
        ids.dispose();
        cost?.dispose();
        if ((b / hp.batchSize) % 8 === 7) await tf.nextFrame?.();
      }
      const tr = await evaluate('train');
      const v = await evaluate('val');
      return { trainLoss: tr.total, valLoss: v.total, metrics: { recon: v.recon, kl: v.kl } };
    },
    async evaluateTest() {
      const r = await evaluate('test');
      return { loss: r.total, recon: r.recon, kl: r.kl };
    },
    encode(images) {
      return tf.tidy(() => encoder.predict(flat(tf, images)).slice([0, 0], [-1, 2]).arraySync());
    },
    decode(zs) {
      return tf.tidy(() => {
        const out = decoder.predict(tf.tensor2d(zs, [zs.length, 2])).dataSync();
        return zs.map((_, i) => out.slice(i * P, (i + 1) * P));
      });
    },
    reconstruct(images) {
      return this.decode(this.encode(images));
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      optimizer.dispose();
      encoder.dispose();
      decoder.dispose();
      Object.values(X).forEach((t) => t.dispose());
    },
  };
}

export function architecture(hp) {
  const h = hp.hidden;
  const rows = [
    { name: 'Input 20×20 → 400', kind: 'input', inShape: [P], outShape: [P], params: 0,
      desc: { ko: '숫자 이미지 400픽셀을 한 줄로 펼쳐 넣습니다.', en: 'The 400 pixels of a digit image, unrolled into one row.' } },
    { name: `Encoder · Dense ${h}`, kind: 'dense', inShape: [P], outShape: [h], params: P * h + h,
      desc: { ko: '인코더: 이미지를 점점 작은 요약으로 압축합니다.', en: 'Encoder: squeezes the image into a smaller summary.' } },
  ];
  if (hp.variational) {
    rows.push({ name: 'μ, log σ² (2 + 2)', kind: 'latent', inShape: [h], outShape: [4], params: h * 4 + 4,
      desc: { ko: 'VAE는 점 하나 대신 "평균(μ)과 퍼짐(σ)"을 가진 작은 구름을 냅니다. 학습 중에는 이 구름에서 무작위로 한 점을 뽑아(재매개변수화) 디코더에 넘깁니다. 그래서 잠재공간이 빈틈없이 매끄럽게 채워집니다.', en: 'Instead of a single point, a VAE outputs a small cloud with a mean (μ) and spread (σ). During training it samples one point from that cloud (the reparameterization trick) and passes it on, which fills the latent space smoothly without gaps.' } });
    rows.push({ name: 'z ~ N(μ, σ²) · 2D latent', kind: 'latent', inShape: [4], outShape: [2], params: 0,
      desc: { ko: '이미지 전체가 단 두 개의 숫자(z₁, z₂)로 요약됩니다. 이 2차원 평면이 "잠재공간"이고, 테스트 탭에서 직접 돌아다녀 볼 수 있습니다.', en: 'The whole image is summarized by just two numbers (z₁, z₂). That 2D plane is the latent space, which you can explore on the Test tab.' } });
  } else {
    rows.push({ name: 'z · 2D latent', kind: 'latent', inShape: [h], outShape: [2], params: h * 2 + 2,
      desc: { ko: '이미지 전체가 단 두 개의 숫자로 압축됩니다(일반 오토인코더). 잠재공간에 빈틈이 생기기 쉬워서, 아무 점이나 골라 디코딩하면 이상한 그림이 나올 수 있습니다.', en: 'The whole image is compressed into just two numbers (plain autoencoder). Its latent space tends to have gaps, so decoding a random point can give odd pictures.' } });
  }
  rows.push(
    { name: `Decoder · Dense ${h}`, kind: 'dense', inShape: [2], outShape: [h], params: 2 * h + h,
      desc: { ko: '디코더: 두 숫자에서 다시 이미지를 펼쳐 냅니다.', en: 'Decoder: unfolds the two numbers back into an image.' } },
    { name: 'Dense 400 · sigmoid', kind: 'output', inShape: [h], outShape: [P], params: h * P + P,
      desc: { ko: '픽셀 400개의 밝기(0~1)를 만듭니다. 원본과 얼마나 같은지가 "재구성 오차"입니다.', en: 'Produces brightness (0–1) for all 400 pixels. How close this is to the original is the reconstruction error.' } },
  );
  return rows;
}
