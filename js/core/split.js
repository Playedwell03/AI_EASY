import { shuffle } from './rng.js';

// Shuffle indices 0..n-1 and cut into train / validation / test.
export function splitIndices(n, ratios = [0.7, 0.15, 0.15], rand) {
  const idx = shuffle(Array.from({ length: n }, (_, i) => i), rand);
  const nTrain = Math.round(n * ratios[0]);
  const nVal = Math.round(n * ratios[1]);
  return {
    train: idx.slice(0, nTrain),
    val: idx.slice(nTrain, nTrain + nVal),
    test: idx.slice(nTrain + nVal),
  };
}
