import mlp from './modules/mlp/index.js';
import cnn from './modules/cnn/index.js';
import vae from './modules/vae/index.js';
import timeseries from './modules/timeseries/index.js';
import rnn from './modules/rnn/index.js';
import transformer from './modules/transformer/index.js';
import bert from './modules/bert/index.js';
import speech from './modules/speech/index.js';
import jev from './modules/jev/index.js';

// Every module, in the order shown on the home page.
export const MODULES = [mlp, cnn, vae, timeseries, rnn, transformer, bert, speech, jev];
