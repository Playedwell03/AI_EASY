import { t, getLang, setLang, onLangChange } from './i18n.js';
import { onRoute, parseRoute } from './router.js';
import { MODULES } from './registry.js';
import { renderHome } from './ui/home.js';
import { mountModule, unmountModule } from './ui/moduleView.js';

const app = document.getElementById('app');
const notice = document.getElementById('notice');
const langBtn = document.getElementById('lang-toggle');
const backendPill = document.getElementById('backend');
let noticeKey = null;

function showNotice(key) {
  noticeKey = key;
  notice.textContent = t(key);
  notice.hidden = false;
}

function renderChrome() {
  document.documentElement.lang = getLang();
  langBtn.textContent = t('lang.toggle');
  for (const el of document.querySelectorAll('[data-t]')) el.textContent = t(el.dataset.t);
  if (noticeKey) notice.textContent = t(noticeKey);
}

async function initBackend() {
  if (!window.tf) { showNotice('notice.tfMissing'); return; }
  let ok = false;
  try { ok = await tf.setBackend('webgl'); } catch { ok = false; }
  if (!ok) {
    await tf.setBackend('cpu');
    showNotice('notice.cpuFallback');
  }
  await tf.ready();
  // Remember the best backend this browser has; modules may temporarily prefer another.
  window.__aiEasyBackend = tf.getBackend();
  showBackend();
  window.addEventListener('aieasy:backend', showBackend);
}

function showBackend() {
  backendPill.textContent = tf.getBackend();
  backendPill.hidden = false;
}

function render(route) {
  if (route.name === 'module') {
    const def = MODULES.find((m) => m.id === route.id);
    if (def) { mountModule(app, def, route.tab); window.scrollTo?.(0, 0); return; }
  }
  unmountModule();
  renderHome(app, MODULES);
}

langBtn.addEventListener('click', () => setLang(getLang() === 'ko' ? 'en' : 'ko'));
onLangChange(() => {
  renderChrome();
  const route = parseRoute(location.hash);
  if (route.name === 'home') renderHome(app, MODULES);
  // Module views listen for language changes themselves, so training keeps running.
});

renderChrome();
await initBackend();
onRoute(render);
