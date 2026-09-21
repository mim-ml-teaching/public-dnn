// Labelled scatter plot from a JSON object of points { name: { x-key: v, y-key: v } }, with the diagonal y = x
// and a least-squares line. Used for "original test set vs new test set" accuracy.
// config: { src, path: "models", x: "orig_top1", y: "v2_top1", xlabel, ylabel, min: 0.5, max: 0.85,
//           width: 560, height: 430, pct: true, place: { "name": "l" | "r" | "la" | "lb" | "ra" | "rb" } }
import { C, h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};

export async function mount(el, cfg) {
  el.classList.add('widget');
  let data;
  try { data = await (await fetch(cfg.src)).json(); } catch { el.append(h('div', { class: 'dim' }, 'data not available yet')); return; }
  const pts = Object.entries(cfg.path ? data[cfg.path] : data).map(([name, d]) => ({ name, x: d[cfg.x], y: d[cfg.y] }));
  const W = cfg.width || 560, H = cfg.height || 430, M = { l: 66, r: 20, t: 16, b: 56 };
  const lo = cfg.min ?? 0.4, hi = cfg.max ?? 0.85;
  const X = v => M.l + (W - M.l - M.r) * (v - lo) / (hi - lo), Y = v => H - M.b - (H - M.t - M.b) * (v - lo) / (hi - lo);
  const f = v => (cfg.pct === false ? v.toFixed(2) : `${Math.round(v * 100)}%`);
  const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: 'font-family: Inter, Arial, sans-serif; display:block' });
  for (let t = Math.ceil(lo * 10) / 10; t <= hi + 1e-9; t += 0.1) {
    svg.append(S('line', { x1: X(lo), x2: X(hi), y1: Y(t), y2: Y(t), stroke: C.line, 'stroke-width': 1 }));
    svg.append(S('line', { x1: X(t), x2: X(t), y1: Y(lo), y2: Y(hi), stroke: C.line, 'stroke-width': 1 }));
    svg.append(S('text', { x: M.l - 8, y: Y(t) + 5, fill: C.dim, 'font-size': 14, 'text-anchor': 'end' }, f(t)));
    svg.append(S('text', { x: X(t), y: H - M.b + 20, fill: C.dim, 'font-size': 14, 'text-anchor': 'middle' }, f(t)));
  }
  svg.append(S('line', { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), stroke: C.dim, 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }));
  svg.append(S('text', { x: X(hi) - 6, y: Y(hi) + 18, fill: C.dim, 'font-size': 14, 'text-anchor': 'end' }, 'same accuracy'));
  // least-squares line
  const n = pts.length, mx = pts.reduce((a, p) => a + p.x, 0) / n, my = pts.reduce((a, p) => a + p.y, 0) / n;
  const b = pts.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0) / pts.reduce((a, p) => a + (p.x - mx) ** 2, 0), a0 = my - b * mx;
  const x1 = Math.min(...pts.map(p => p.x)) - 0.02, x2 = Math.max(...pts.map(p => p.x)) + 0.02;
  svg.append(S('line', { x1: X(x1), y1: Y(a0 + b * x1), x2: X(x2), y2: Y(a0 + b * x2), stroke: C.accent, 'stroke-width': 2 }));
  pts.forEach(p => {
    svg.append(S('circle', { cx: X(p.x), cy: Y(p.y), r: 6, fill: C.blue, stroke: C.bg, 'stroke-width': 1.5 }));
    const pl = (cfg.place || {})[p.name] || 'r';              // r, l, ra, rb, la, lb: side and above/below
    const left = pl[0] === 'l', dy = pl[1] === 'a' ? -10 : pl[1] === 'b' ? 18 : 5;
    svg.append(S('text', { x: X(p.x) + (left ? -10 : 10), y: Y(p.y) + dy, fill: C.fg, 'font-size': 14, 'text-anchor': left ? 'end' : 'start' }, p.name));
  });
  svg.append(S('text', { x: (M.l + W - M.r) / 2, y: H - 12, fill: C.fg, 'font-size': 15, 'text-anchor': 'middle' }, cfg.xlabel || ''));
  svg.append(S('text', { x: 16, y: (M.t + H - M.b) / 2, fill: C.fg, 'font-size': 15, 'text-anchor': 'middle', transform: `rotate(-90 16 ${(M.t + H - M.b) / 2})` }, cfg.ylabel || ''));
  el.append(svg);
}
