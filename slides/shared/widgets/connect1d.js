// From a dense layer to a convolution in two steps, drawn in 1D (a row of pixels → a row of outputs):
// dense (every output sees every input) → local (each output sees 3 neighbours) → local + shared weights
// (the same 3 weights everywhere = a convolution). Line colour = weight identity: many colours = many
// different weights; in the shared panel only three colours (w₁, w₂, w₃) appear.
// config: { n: 8, width: 1180, height: 250 }
import { C, h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};

export function mount(el, cfg) {
  el.classList.add('widget');
  const n = cfg.n || 8, W = cfg.width || 1180, H = cfg.height || 250;
  const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const panels = [
    { title: 'dense', sub: 'every output sees every input' },
    { title: 'local', sub: 'each output sees 3 neighbours' },
    { title: 'local + shared', sub: 'the same 3 weights everywhere' },
  ];
  const pw = W / 3, top = 62, bot = H - 30, r = 9;
  const hue = i => `hsl(${(i * 47) % 360}, 65%, 62%)`;
  const shared = [C.red, C.green, C.blue];
  panels.forEach((p, k) => {
    const x0 = k * pw + 30, x1 = (k + 1) * pw - 30, X = i => x0 + (x1 - x0) * i / (n - 1);
    svg.append(S('text', { x: (x0 + x1) / 2, y: 20, fill: C.accent, 'font-size': 20, 'font-weight': 600, 'text-anchor': 'middle' }, p.title));
    svg.append(S('text', { x: (x0 + x1) / 2, y: 40, fill: C.dim, 'font-size': 15, 'text-anchor': 'middle' }, p.sub));
    let idx = 0;
    for (let o = 0; o < n; o++) for (let i = 0; i < n; i++) {
      const d = i - o;
      if (k > 0 && Math.abs(d) > 1) continue;
      const color = k === 0 ? '#5a606c' : k === 1 ? hue(idx++) : shared[d + 1];
      svg.append(S('line', { x1: X(i), y1: bot - r, x2: X(o), y2: top + r, stroke: color, 'stroke-width': k === 0 ? 1 : 2.4, opacity: k === 0 ? 0.8 : 1 }));
    }
    for (let i = 0; i < n; i++) {
      svg.append(S('circle', { cx: X(i), cy: top, r, fill: C.bg3, stroke: C.blue, 'stroke-width': 2 }));
      svg.append(S('circle', { cx: X(i), cy: bot, r, fill: C.bg3, stroke: C.dim, 'stroke-width': 2 }));
    }
    if (k === 2) {
      ['w₁', 'w₂', 'w₃'].forEach((t, j) => svg.append(S('text', { x: x1 - 70 + j * 32, y: H - 4, fill: shared[j], 'font-size': 16, 'font-weight': 700 }, t)));
    }
  });
  el.append(svg);
}
