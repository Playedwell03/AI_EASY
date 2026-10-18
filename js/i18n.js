// Korean/English text. t(key, vars, dict) looks in dict (a module's text) first, then COMMON.

export const COMMON = {
  ko: {
    'app.name': 'AI Easy',
    'app.tagline': '클릭하며 배우는 딥러닝',
    'lang.toggle': 'English',
    'nav.home': '모든 모듈',
    'notice.cpuFallback': '이 브라우저에서는 GPU(WebGL)를 쓸 수 없어 CPU로 학습합니다. 조금 느릴 수 있어요.',
    'notice.tfMissing': 'TensorFlow.js를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.',

    'home.title': '모델이 어떻게 배우는지, 직접 눌러 보며 확인하세요',
    'home.intro': '각 모듈은 실제 딥러닝 분야의 대표 구조를 작게 줄인 것입니다. 구조를 살펴보고, 데이터를 보고, 브라우저 안에서 직접 학습시킨 다음, "정말 잘 배운 걸까?"를 그래프와 근거로 확인합니다. 설치할 것은 없습니다.',
    'home.start': '처음이라면 여기부터: 기초 신경망',
    'home.flow': '모든 모듈은 같은 6단계로 진행됩니다',
    'home.modules': '모듈',

    'tab.arch': '구조',
    'tab.data': '데이터',
    'tab.train': '학습',
    'tab.diagnose': '진단',
    'tab.test': '테스트',
    'tab.use': '활용',
    'tabdesc.arch': '모델이 어떤 층으로 이루어졌는지',
    'tabdesc.data': '무엇을 보고 배우는지',
    'tabdesc.train': '직접 학습시켜 보기',
    'tabdesc.diagnose': '잘 배웠는지 근거로 판단',
    'tabdesc.test': '처음 보는 데이터로 확인',
    'tabdesc.use': '실제 서비스에 넣어 써 보기',
    'use.pick': '어떤 곳에 써 볼까요?',
    'use.pickHelp': '학습시킨 모델을 실제 서비스를 흉내 낸 작은 앱에 끼워 넣었습니다. 체험을 고르면, 앱이 일하는 순서 중 어디에서 내 모델이 쓰이는지 함께 보여 줍니다.',
    'use.needTrain': '먼저 학습을 해야 내 모델을 앱에 넣어 볼 수 있습니다.',
    'use.flow': '이 앱이 일하는 순서',
    'use.flowHelp': '단계를 누르면 설명이 나옵니다. 앱을 조작하면 지금 실행되는 단계가 빛납니다.',
    'use.modelStep': '내가 학습시킨 모델',
    'use.realWorld': '실제 서비스라면',
    'use.kind.input': '입력',
    'use.kind.pre': '전처리',
    'use.kind.model': '모델',
    'use.kind.post': '판단',
    'use.kind.output': '결과',

    'btn.start': '학습 시작',
    'btn.resume': '이어서 학습',
    'btn.pause': '일시정지',
    'btn.reset': '초기화',
    'btn.break': '일부러 망가뜨려 보기',
    'btn.defaults': '기본 설정으로',
    'btn.goTrain': '학습하러 가기',
    'btn.next': '다음 단계',

    'state.idle': '대기 중',
    'state.running': '학습 중…',
    'state.paused': '일시정지됨',
    'state.done': '학습 완료',
    'state.diverged': '학습이 발산해서 멈췄습니다',
    'state.error': '오류로 학습이 멈췄습니다',

    'arch.clickHint': '블록을 누르면 그 층이 하는 일을 볼 수 있어요.',
    'arch.input': '입력 모양',
    'arch.output': '출력 모양',
    'arch.params': '학습되는 숫자(파라미터)',
    'arch.totalParams': '전체 파라미터 수',
    'arch.settings': '구조 설정',
    'arch.changeResets': '구조를 바꾸면 학습이 초기화됩니다.',

    'data.split': '데이터 나누기',
    'data.train': '학습용(Train)',
    'data.val': '검증용(Validation)',
    'data.test': '테스트용(Test)',
    'data.why': '왜 셋으로 나눌까요? 모델은 학습용 데이터만 보고 배웁니다. 검증용 데이터는 학습 도중 "처음 보는 문제도 잘 푸는지" 확인하는 모의고사이고, 테스트용 데이터는 모든 결정이 끝난 뒤 딱 한 번 보는 실제 시험입니다. 학습용 점수만 좋으면 문제를 외운 것일 수도 있어요.',
    'data.samples': '샘플 미리보기',
    'badge.synthetic': '합성 데이터',
    'badge.syntheticHint': '이 데이터는 브라우저에서 규칙으로 만들어 낸 연습용 데이터입니다.',

    'train.settings': '학습 설정',
    'train.lossChart': '손실(Loss) — 낮을수록 좋음',
    'train.metricChart': '지표',
    'train.epoch': '에폭(Epoch)',
    'train.epochOf': '에폭 {n} / {total}',
    'train.trainLoss': '학습 손실',
    'train.valLoss': '검증 손실',
    'train.lossHelp': '손실은 모델이 얼마나 틀렸는지를 나타내는 숫자입니다. 두 선이 함께 내려가면 잘 배우고 있는 것이고, 학습 손실만 내려가고 검증 손실이 올라가면 문제를 외우기 시작한 것입니다.',
    'train.presets': '일부러 망가뜨려 보기',
    'train.presetsHelp': '나쁜 설정으로 학습해 보고, 진단 탭이 문제를 잡아내는지 확인해 보세요.',
    'train.inside': '모델 안을 들여다보기',
    'train.seed': '시드(Seed) {seed}',
    'train.warmup': '첫 에폭은 계산 준비(GPU 프로그램 만들기) 때문에 조금 더 걸립니다. 이 탭을 화면에 띄워 두세요 — 다른 탭으로 가면 브라우저가 학습을 늦춥니다.',

    'hp.lr': '학습률(Learning rate)',
    'hp.epochs': '에폭 수',
    'hp.batchSize': '배치 크기',
    'hp.seed': '시드',

    'diag.title': '이 학습, 잘 된 걸까요?',
    'diag.noRun': '아직 학습 기록이 없습니다. 먼저 학습 탭에서 학습을 돌려 주세요.',
    'diag.evidence': '근거',
    'diag.advice': '다음에 해 볼 것',
    'diag.metrics': '핵심 지표',
    'diag.bestEpoch': '검증 손실이 가장 낮았던 지점: 에폭 {n}',
    'diag.baseline': '무작위로 찍었을 때의 손실(기준선): {v}',
    'diag.baselineShort': '기준선(무작위 추측) 손실',
    'diag.extraProblem': '손실 곡선은 좋아 보이지만, 아래 추가 검사에서 문제가 발견되었습니다. 근거 목록을 확인해 보세요.',
    'diag.howTo': '판정 방법: 학습 손실과 검증 손실 곡선의 모양, 그리고 "아무것도 배우지 않고 찍었을 때"의 손실과 비교해 판단합니다.',

    'verdict.good': '잘 학습되었습니다',
    'verdict.insufficient': '판단하기엔 아직 이릅니다',
    'verdict.diverged': '학습이 발산했습니다',
    'verdict.underfitting': '학습이 부족합니다 (과소적합)',
    'verdict.overfitting': '문제를 외우기 시작했습니다 (과적합)',
    'verdictdesc.good': '학습용과 검증용 모두에서 손실이 충분히 내려갔고, 둘의 차이도 작습니다.',
    'verdictdesc.insufficient': '에폭이 너무 적어서 곡선의 모양을 볼 수 없습니다.',
    'verdictdesc.diverged': '손실이 줄어들지 않고 폭발했습니다. 보통 학습률이 너무 클 때 생깁니다.',
    'verdictdesc.underfitting': '모델이 데이터의 규칙을 충분히 잡아내지 못했습니다.',
    'verdictdesc.overfitting': '학습 데이터는 점점 잘 맞히지만, 처음 보는 데이터에서는 오히려 나빠지고 있습니다.',

    'f.lossNaN': '에폭 {epoch}에서 손실이 숫자가 아닌 값(NaN/무한대)이 되었습니다.',
    'f.lossExploded': '손실이 {from}에서 {to}로 오히려 커졌습니다.',
    'f.worseThanBaseline': '손실({loss})이 무작위로 찍었을 때({baseline})보다 훨씬 큽니다. 모델이 망가진 상태입니다.',
    'f.spiked': '에폭 {epoch}에서 손실이 무작위로 찍었을 때의 {x}배까지 치솟았습니다. 학습이 불안정합니다.',
    'f.tooFewEpochs': '지금까지 {n}에폭만 학습했습니다. 최소 3에폭은 필요합니다.',
    'f.valRoseAfter': '검증 손실이 에폭 {epoch}에서 가장 낮았고, 그 뒤로 {pct}% 다시 올라갔습니다.',
    'f.bigGap': '학습 손실({train})은 거의 0인데 검증 손실({val})은 훨씬 높습니다(차이: 기준선의 {gapPct}%). 학습 데이터를 외웠다는 뜻입니다.',
    'f.valStalled': '검증 손실은 더 이상 내려가지 않고 있습니다.',
    'f.trainKeptFalling': '그동안 학습 손실은 계속 내려갔습니다. 학습 데이터만 외우고 있다는 신호입니다.',
    'f.nearBaseline': '학습 손실({loss})이 무작위로 찍었을 때({baseline})와 큰 차이가 없습니다. 처음보다 {pct}%만 줄었습니다.',
    'f.beatsBaseline': '검증 손실이 무작위로 찍었을 때보다 {ratio:1}배 낮습니다.',
    'f.gapSmall': '학습 손실과 검증 손실의 차이가 작습니다(기준선의 {gapPct}%). 외운 게 아니라 규칙을 배웠다는 뜻입니다.',
    'f.gapLarge': '학습 손실과 검증 손실의 차이가 꽤 있습니다(기준선의 {gapPct}%). 과적합으로 가는 중일 수 있어요.',
    'f.stillImproving': '검증 손실이 아직 내려가는 중입니다.',

    'adv.lowerLr': '학습률을 10분의 1로 줄여 보세요.',
    'adv.trainMore': '에폭 수를 늘려 더 학습해 보세요.',
    'adv.earlyStop': '검증 손실이 가장 낮았던 에폭까지만 학습하세요(조기 종료).',
    'adv.regularize': '모델을 작게 하거나 규제(정규화)를 켜 보세요.',
    'adv.moreData': '학습 데이터를 늘려 보세요.',
    'adv.biggerModel': '층이나 뉴런 수를 늘려 모델을 키워 보세요.',
    'adv.raiseLr': '학습률을 조금 키워 보세요.',

    'test.title': '처음 보는 데이터로 시험',
    'test.needTrain': '먼저 학습을 해야 테스트할 수 있습니다.',
    'test.results': '테스트 세트 결과',
    'test.try': '직접 해 보기',
    'test.mistakes': '틀린 예시',
    'test.running': '테스트 중…',
    'test.waitTraining': '학습이 진행 중입니다. 학습이 끝나거나 일시정지한 뒤에 테스트할 수 있어요.',
    'test.explain': '테스트 세트는 학습에도, 설정을 고르는 데에도 한 번도 쓰지 않은 데이터입니다. 여기 점수가 실제 실력에 가장 가깝습니다.',

    'metric.accuracy': '정확도',
    'metric.loss': '손실',
    'metric.valAccuracy': '검증 정확도',
    'metric.trainAccuracy': '학습 정확도',
    'metric.testAccuracy': '테스트 정확도',
    'metric.testLoss': '테스트 손실',
    'metric.perplexity': '퍼플렉서티(Perplexity)',
    'metric.mse': '평균제곱오차(MSE)',
    'metric.mae': '평균절대오차(MAE)',
    'metric.ece': '보정 오차(ECE)',
    'chart.truth': '정답',
    'chart.pred': '예측',
    'chart.empty': '학습을 시작하면 여기에 그래프가 그려집니다.',
  },
  en: {
    'app.name': 'AI Easy',
    'app.tagline': 'Learn deep learning by clicking',
    'lang.toggle': '한국어',
    'nav.home': 'All modules',
    'notice.cpuFallback': 'GPU (WebGL) is not available in this browser, so training runs on the CPU. It may be a bit slower.',
    'notice.tfMissing': 'Could not load TensorFlow.js. Check your internet connection and reload.',

    'home.title': 'See how models learn, by trying it yourself',
    'home.intro': 'Each module is a small version of a real deep learning architecture. Look at its structure, look at the data, train it right here in your browser, then check with charts and evidence whether it really learned. Nothing to install.',
    'home.start': 'New here? Start with: Basic neural network',
    'home.flow': 'Every module follows the same 6 steps',
    'home.modules': 'Modules',

    'tab.arch': 'Architecture',
    'tab.data': 'Data',
    'tab.train': 'Train',
    'tab.diagnose': 'Diagnose',
    'tab.test': 'Test',
    'tab.use': 'Use',
    'tabdesc.arch': 'What layers the model is made of',
    'tabdesc.data': 'What it learns from',
    'tabdesc.train': 'Train it yourself',
    'tabdesc.diagnose': 'Judge, with evidence, if it learned',
    'tabdesc.test': 'Check on data it has never seen',
    'tabdesc.use': 'Put it to work in an app',
    'use.pick': 'Where shall we use it?',
    'use.pickHelp': 'Your trained model is plugged into small apps that imitate real services. Pick one to see, step by step, where in the app your model does its job.',
    'use.needTrain': 'Train the model first, then you can plug it into an app.',
    'use.flow': 'How this app works',
    'use.flowHelp': 'Click a step for details. When you use the app, the step that is running lights up.',
    'use.modelStep': 'Your trained model',
    'use.realWorld': 'In a real service',
    'use.kind.input': 'Input',
    'use.kind.pre': 'Prepare',
    'use.kind.model': 'Model',
    'use.kind.post': 'Decide',
    'use.kind.output': 'Result',

    'btn.start': 'Start training',
    'btn.resume': 'Resume',
    'btn.pause': 'Pause',
    'btn.reset': 'Reset',
    'btn.break': 'Break it on purpose',
    'btn.defaults': 'Back to defaults',
    'btn.goTrain': 'Go train',
    'btn.next': 'Next step',

    'state.idle': 'Idle',
    'state.running': 'Training…',
    'state.paused': 'Paused',
    'state.done': 'Training finished',
    'state.diverged': 'Stopped: training diverged',
    'state.error': 'Stopped because of an error',

    'arch.clickHint': 'Click a block to see what that layer does.',
    'arch.input': 'Input shape',
    'arch.output': 'Output shape',
    'arch.params': 'Learned numbers (parameters)',
    'arch.totalParams': 'Total parameters',
    'arch.settings': 'Architecture settings',
    'arch.changeResets': 'Changing the architecture resets training.',

    'data.split': 'Splitting the data',
    'data.train': 'Train',
    'data.val': 'Validation',
    'data.test': 'Test',
    'data.why': 'Why three parts? The model learns only from the training data. Validation data is a practice exam during training that checks whether it can solve problems it has not seen. Test data is the real exam, used once after every decision is made. A great training score alone may just mean the model memorized the answers.',
    'data.samples': 'Sample preview',
    'badge.synthetic': 'Synthetic data',
    'badge.syntheticHint': 'This practice data is generated by rules right in your browser.',

    'train.settings': 'Training settings',
    'train.lossChart': 'Loss — lower is better',
    'train.metricChart': 'Metrics',
    'train.epoch': 'Epoch',
    'train.epochOf': 'Epoch {n} / {total}',
    'train.trainLoss': 'Train loss',
    'train.valLoss': 'Validation loss',
    'train.lossHelp': 'Loss is a number for how wrong the model is. If both lines go down together, it is learning. If only the training loss goes down while validation loss rises, it has started memorizing.',
    'train.presets': 'Break it on purpose',
    'train.presetsHelp': 'Train with a bad setting, then see whether the Diagnose tab catches the problem.',
    'train.inside': 'Look inside the model',
    'train.seed': 'Seed {seed}',
    'train.warmup': 'The first epoch takes longer while the GPU programs are prepared. Keep this tab on screen — browsers slow down background tabs.',

    'hp.lr': 'Learning rate',
    'hp.epochs': 'Epochs',
    'hp.batchSize': 'Batch size',
    'hp.seed': 'Seed',

    'diag.title': 'Did this training work?',
    'diag.noRun': 'No training history yet. Train the model on the Train tab first.',
    'diag.evidence': 'Evidence',
    'diag.advice': 'What to try next',
    'diag.metrics': 'Key metrics',
    'diag.bestEpoch': 'Lowest validation loss at epoch {n}',
    'diag.baseline': 'Loss of random guessing (baseline): {v}',
    'diag.baselineShort': 'Baseline (random guess) loss',
    'diag.extraProblem': 'The loss curves look fine, but the extra checks below found a problem. See the evidence list.',
    'diag.howTo': 'How this is judged: from the shape of the training and validation loss curves, compared with the loss of guessing without learning anything.',

    'verdict.good': 'Trained well',
    'verdict.insufficient': 'Too early to tell',
    'verdict.diverged': 'Training diverged',
    'verdict.underfitting': 'Not learned enough (underfitting)',
    'verdict.overfitting': 'Started memorizing (overfitting)',
    'verdictdesc.good': 'Loss went down far enough on both training and validation data, and the gap between them is small.',
    'verdictdesc.insufficient': 'Too few epochs to see the shape of the curves.',
    'verdictdesc.diverged': 'Instead of shrinking, the loss blew up. This usually means the learning rate is too high.',
    'verdictdesc.underfitting': 'The model has not picked up the patterns in the data.',
    'verdictdesc.overfitting': 'It keeps getting better on training data but worse on data it has not seen.',

    'f.lossNaN': 'At epoch {epoch} the loss became not-a-number (NaN/infinity).',
    'f.lossExploded': 'Loss grew from {from} to {to} instead of shrinking.',
    'f.worseThanBaseline': 'Loss ({loss}) is far worse than random guessing ({baseline}). The model is broken.',
    'f.spiked': 'At epoch {epoch} the loss shot up to {x}x that of random guessing. Training is unstable.',
    'f.tooFewEpochs': 'Only {n} epoch(s) so far. At least 3 are needed.',
    'f.valRoseAfter': 'Validation loss was lowest at epoch {epoch}, then rose again by {pct}%.',
    'f.bigGap': 'Training loss ({train}) is near zero but validation loss ({val}) is much higher (gap: {gapPct}% of baseline). It memorized the training data.',
    'f.valStalled': 'Validation loss is no longer going down.',
    'f.trainKeptFalling': 'Meanwhile training loss kept falling — a sign it is memorizing the training data.',
    'f.nearBaseline': 'Training loss ({loss}) is not far from random guessing ({baseline}). It dropped only {pct}% from the start.',
    'f.beatsBaseline': 'Validation loss is {ratio:1}x lower than random guessing.',
    'f.gapSmall': 'The gap between training and validation loss is small ({gapPct}% of baseline). It learned the rule rather than memorizing.',
    'f.gapLarge': 'There is a noticeable gap between training and validation loss ({gapPct}% of baseline). It may be heading toward overfitting.',
    'f.stillImproving': 'Validation loss is still going down.',

    'adv.lowerLr': 'Lower the learning rate to one tenth.',
    'adv.trainMore': 'Train for more epochs.',
    'adv.earlyStop': 'Stop at the epoch with the lowest validation loss (early stopping).',
    'adv.regularize': 'Make the model smaller or turn on regularization.',
    'adv.moreData': 'Use more training data.',
    'adv.biggerModel': 'Make the model bigger with more layers or units.',
    'adv.raiseLr': 'Raise the learning rate a little.',

    'test.title': 'Exam on unseen data',
    'test.needTrain': 'Train the model first, then you can test it.',
    'test.results': 'Test set results',
    'test.try': 'Try it yourself',
    'test.mistakes': 'Mistakes',
    'test.running': 'Testing…',
    'test.waitTraining': 'Training is in progress. You can test once it finishes or is paused.',
    'test.explain': 'The test set was never used for training or for choosing settings. This score is the closest to real-world performance.',

    'metric.accuracy': 'Accuracy',
    'metric.loss': 'Loss',
    'metric.valAccuracy': 'Validation accuracy',
    'metric.trainAccuracy': 'Train accuracy',
    'metric.testAccuracy': 'Test accuracy',
    'metric.testLoss': 'Test loss',
    'metric.perplexity': 'Perplexity',
    'metric.mse': 'Mean squared error (MSE)',
    'metric.mae': 'Mean absolute error (MAE)',
    'metric.ece': 'Calibration error (ECE)',
    'chart.truth': 'Actual',
    'chart.pred': 'Predicted',
    'chart.empty': 'Charts appear here once training starts.',
  },
};

const STORE_KEY = 'aieasy.lang';
let lang = readStored() || 'ko';
const listeners = new Set();

function readStored() {
  try {
    const v = globalThis.localStorage?.getItem(STORE_KEY);
    return v === 'ko' || v === 'en' ? v : null;
  } catch {
    return null;
  }
}

export function getLang() {
  return lang;
}

export function setLang(l) {
  if (l !== 'ko' && l !== 'en') return;
  lang = l;
  try {
    globalThis.localStorage?.setItem(STORE_KEY, l);
  } catch {}
  if (globalThis.document) document.documentElement.lang = l;
  for (const fn of listeners) fn(l);
}

export function onLangChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

// "{name}" substitutes vars.name; "{name:1}" formats a number with 1 decimal.
export function t(key, vars, dict) {
  const s = dict?.[lang]?.[key] ?? COMMON[lang][key] ?? key;
  if (!vars) return s;
  return s.replace(/\{(\w+)(?::(\d))?\}/g, (m, k, d) => {
    if (!(k in vars)) return m;
    const v = vars[k];
    return d !== undefined && typeof v === 'number' ? v.toFixed(Number(d)) : String(v);
  });
}

// Pick the current language from a { ko, en } pair.
export function L(pair) {
  return pair?.[lang] ?? pair?.ko ?? '';
}
