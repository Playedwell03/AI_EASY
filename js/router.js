// Hash routes: #/ → home, #/m/<id>/<tab> → a module tab (tab defaults to "arch").

export const TABS = ['arch', 'data', 'train', 'diagnose', 'test'];

export function parseRoute(hash) {
  const parts = (hash || '').replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts[0] === 'm' && parts[1]) {
    return { name: 'module', id: parts[1], tab: TABS.includes(parts[2]) ? parts[2] : 'arch' };
  }
  return { name: 'home' };
}

export function moduleHref(id, tab = 'arch') {
  return `#/m/${id}/${tab}`;
}

export function onRoute(fn) {
  const run = () => fn(parseRoute(location.hash));
  window.addEventListener('hashchange', run);
  run();
}
