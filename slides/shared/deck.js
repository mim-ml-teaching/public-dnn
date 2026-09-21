// Shared deck bootstrap: reveal.js + KaTeX auto-render + lazy widget mounting.
// Usage in a lecture: <script type="module" src="../../shared/deck.js"></script>
// Widgets: <div data-widget="warp2d" data-config='{"preset":"skew"}'></div>
import Reveal from '../vendor/reveal/reveal.mjs';
import Notes from '../vendor/reveal/plugin/notes.mjs';
import { applyTheme } from './theme.js';

const widgetModules = {
  slido: () => import('./widgets/slido.js'),
  linechart: () => import('./widgets/linechart.js'),
  relu1d: () => import('./widgets/relu1d.js'),
  sawtooth: () => import('./widgets/sawtooth.js'),
  playground2d: () => import('./widgets/playground2d.js'),
  sgd2d: () => import('./widgets/sgd2d.js'),
  handgrad: () => import('./widgets/handgrad.js'),
  paramscale: () => import('./widgets/paramscale.js'),
  nudge: () => import('./widgets/nudge.js'),
  neuron2d: () => import('./widgets/neuron2d.js'),
  perceptron2d: () => import('./widgets/perceptron2d.js'),
  boolnet: () => import('./widgets/boolnet.js'),
  flow2d: () => import('./widgets/flow2d.js'),
  warp1d: () => import('./widgets/warp1d.js'),
  ode2d: () => import('./widgets/ode2d.js'),
  path1d: () => import('./widgets/path1d.js'),
  deform2d: () => import('./widgets/deform2d.js'),
  flowrows: () => import('./widgets/flowrows.js'),
  melt: () => import('./widgets/melt.js'),
  condfam: () => import('./widgets/condfam.js'),
  jacdet: () => import('./widgets/jacdet.js'),
  genimg: () => import('./widgets/genimg.js'),
  gauss1d: () => import('./widgets/gauss1d.js'),
  gmmfit: () => import('./widgets/gmmfit.js'),
  paramcost: () => import('./widgets/paramcost.js'),
  condmarg: () => import('./widgets/condmarg.js'),
  ytvideo: () => import('./widgets/ytvideo.js'),
  chunkflow: () => import('./widgets/chunkflow.js'),
};

function renderMath(root) {
  window.renderMathInElement(root, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
    trust: true,   // for \htmlClass: coloured symbols via CSS classes, so that the light theme still applies
    macros: { '\\cq': '\\htmlClass{cq}{#1}', '\\cp': '\\htmlClass{cp}{#1}', '\\cb': '\\htmlClass{cb}{#1}' },   // \cq{q}: the data distribution (blue), \cp{p_\theta}: the model (accent), \cb{…}: blue, matching blue samples in a figure
  });
}

async function mountWidget(el) {
  if (el.dataset.mounted) return;
  el.dataset.mounted = '1';
  const gen = el.dataset.gen = String((+el.dataset.gen || 0) + 1);
  const load = widgetModules[el.dataset.widget];
  if (!load) { el.textContent = `Unknown widget: ${el.dataset.widget}`; return; }
  const config = el.dataset.config ? JSON.parse(el.dataset.config) : {};
  const mod = await load();
  if (el.dataset.gen !== gen || !el.dataset.mounted) return; // unmounted while loading
  await mod.mount(el, config); // async widgets (data fetch) finish before the print view freezes them
}

// Browsers allow only ~16 live WebGL contexts; release widgets on slides we are not near.
function unmountWidget(el) {
  if (!el.dataset.mounted) return;
  el.querySelectorAll('canvas').forEach(c => {
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    gl?.getExtension('WEBGL_lose_context')?.loseContext();
  });
  el.replaceChildren();
  el.removeAttribute('style');
  el.classList.remove('widget');
  delete el.dataset.mounted;
  el.dataset.gen = String((+el.dataset.gen || 0) + 1);
}

// Development aid (localhost only). A hard reload refreshes what the page loads up front, not the widget
// modules imported lazily later, so an edited widget (or util.js) could be served stale from the browser cache.
// Re-fetch the widget modules and everything they import with cache: 'reload' before any widget mounts.
async function refreshModuleCache() {
  if (!/^(localhost|127\.0\.0\.1)$/.test(location.hostname)) return;
  const seen = new Set();
  const visit = async url => {
    if (seen.has(url)) return;
    seen.add(url);
    let text;
    try { text = await (await fetch(url, { cache: 'reload' })).text(); } catch { return; }
    const specs = [...text.matchAll(/^\s*import\s+(?:[^'"]*?\bfrom\s+)?['"]([^'"]+)['"]/gm), ...text.matchAll(/import\(\s*['"]([^'"]+)['"]\s*\)/g)].map(m => m[1]);
    await Promise.all(specs.filter(sp => /^\.{0,2}\//.test(sp)).map(sp => visit(new URL(sp, url).href)));
  };
  await Promise.all(Object.values(widgetModules).map(fn => { const m = /import\('([^']+)'\)/.exec(fn.toString()); return m && visit(new URL(m[1], import.meta.url).href); }));
}
await refreshModuleCache();

await applyTheme(); // light colours for ?print-pdf and ?light

const deck = new Reveal({
  width: 1280,
  height: 720,
  margin: 0.04,
  hash: true,
  slideNumber: 'c/t',
  transition: 'fade',
  transitionSpeed: 'fast',
  center: false,
  pdfSeparateFragments: false,
  plugins: [Notes],
});

await deck.initialize();
renderMath(document.querySelector('.reveal .slides'));

// Wait n animation frames; falls back to a timer because headless printing can starve requestAnimationFrame.
const frames = n => new Promise(res => {
  let done = false;
  const fin = () => { if (!done) { done = true; res(); } };
  const f = k => (k ? requestAnimationFrame(() => f(k - 1)) : fin());
  f(n);
  setTimeout(fin, 120 * n + 200);
});

// Print view: replace a rendered WebGL canvas by a still image and free its context, so that decks with
// more 3D figures than the browser's WebGL context limit (~16) still print every one of them.
function freezeWebGL(el) {
  el.querySelectorAll('canvas').forEach(c => {
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (!gl) return;
    const img = new Image();
    img.src = c.toDataURL('image/png');
    img.style.cssText = c.style.cssText;
    img.style.width = c.clientWidth + 'px';
    img.style.height = c.clientHeight + 'px';
    img.style.margin = '0';
    c.replaceWith(img);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  });
}

// Print view: mount every widget, one at a time, then set window.__deckReady (make-pdf.mjs waits for it).
async function mountAllForPrint() {
  for (const el of document.querySelectorAll('[data-widget]')) {
    await mountWidget(el);
    await frames(3);
    freezeWebGL(el);
  }
  await document.fonts.ready;
  renderMath(document.querySelector('.reveal .slides'));
  await frames(3);
  window.__deckReady = true;
}

// Mount widgets on the current slide and its neighbours; in print view mount everything.
let printing = false;
function mountAround() {
  if (deck.isPrintView()) {
    if (!printing) { printing = true; mountAllForPrint(); }
    return;
  }
  const cur = deck.getCurrentSlide();
  const keep = [cur?.previousElementSibling, cur, cur?.nextElementSibling].filter(Boolean);
  document.querySelectorAll('.reveal .slides section').forEach(sec => {
    if (!keep.includes(sec)) sec.querySelectorAll('[data-widget][data-mounted]').forEach(unmountWidget);
  });
  keep.forEach(s => s.querySelectorAll('[data-widget]').forEach(mountWidget));
}

deck.on('ready', mountAround);
deck.on('slidechanged', mountAround);
mountAround();

window.deck = deck;
