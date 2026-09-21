// Upsampling without parameters, three panels: nearest neighbour, "bed of nails" (zeros), and max unpooling
// (put each value back where the max of the matching pooling layer was).
// config: { cell: 34, panels: ["nearest", "nails", "maxunpool"] }
import { C } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const X = [[1, 2], [3, 4]];
const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
const COL = [[C.red, C.green], [C.blue, C.purple]];

export function mount(el, cfg) {
  el.classList.add('widget');
  const cs = cfg.cell || 34, panels = cfg.panels || ['nearest', 'nails', 'maxunpool'];
  const g = (svg, x0, y0, rows, fn, c = cs) => {
    rows.forEach((row, r) => row.forEach((v, cI) => {
      const o = fn(r, cI, v);
      svg.append(S('rect', { x: x0 + cI * c, y: y0 + r * c, width: c, height: c, fill: o.fill || C.bg2, stroke: o.stroke || C.line, 'stroke-width': o.sw || 1 }));
      if (o.text !== undefined) svg.append(S('text', { x: x0 + cI * c + c / 2, y: y0 + r * c + c / 2 + c * 0.16, fill: o.color || C.fg, 'font-size': Math.round(c * 0.45), 'text-anchor': 'middle', 'font-weight': o.bold ? 700 : 400 }, o.text));
    }));
  };
  const arrow = (svg, x1, y, x2, label) => {
    svg.append(S('line', { x1, y1: y, x2: x2 - 8, y2: y, stroke: '#9aa1ae', 'stroke-width': 2 }));
    svg.append(S('polygon', { points: `${x2},${y} ${x2 - 10},${y - 5} ${x2 - 10},${y + 5}`, fill: '#9aa1ae' }));
    if (label) svg.append(S('text', { x: (x1 + x2) / 2, y: y - 9, fill: C.dim, 'font-size': 14, 'text-anchor': 'middle' }, label));
  };
  const title = (svg, x, y, t, sub) => {
    svg.append(S('text', { x, y, fill: C.accent, 'font-size': 18, 'font-weight': 600 }, t));
    if (sub) svg.append(S('text', { x, y: y + 20, fill: C.dim, 'font-size': 14 }, sub));
  };
  const row = h0 => { const d = document.createElement('div'); d.style.cssText = 'display:flex; gap:2.2em; align-items:flex-start; flex-wrap:wrap'; return d; };
  const wrap = row();
  const in2 = (svg, x0, y0) => g(svg, x0, y0, X, (r, c, v) => ({ fill: rgba(COL[r][c], 0.3), stroke: COL[r][c], text: v, bold: true }));
  for (const p of panels) {
    const svg = S('svg', { style: 'font-family: Inter, Arial, sans-serif; display:block; overflow:visible' });
    if (p === 'nearest' || p === 'nails') {
      const W = 2 * cs + 70 + 4 * cs + 4, H = 4 * cs + 60;
      svg.setAttribute('width', W); svg.setAttribute('height', H);
      title(svg, 0, 18, p === 'nearest' ? 'nearest neighbour' : 'bed of nails', p === 'nearest' ? 'copy each value 2×2' : 'value in one corner, zeros elsewhere');
      in2(svg, 0, 50 + cs);
      arrow(svg, 2 * cs + 8, 50 + 2 * cs, 2 * cs + 62);
      const Y = [0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => (p === 'nearest' ? X[r >> 1][c >> 1] : r % 2 || c % 2 ? 0 : X[r >> 1][c >> 1])));
      g(svg, 2 * cs + 70, 50, Y, (r, c, v) => ({ fill: v ? rgba(COL[r >> 1][c >> 1], 0.3) : C.bg2, text: v, color: v ? C.fg : C.dim }));
    } else {
      // max pooling earlier in the network remembers where the max was; unpooling puts values back there
      const A = [[1, 2, 6, 3], [3, 5, 2, 1], [1, 2, 2, 1], [7, 3, 4, 8]];
      const where = [[1, 1], [0, 2], [3, 0], [3, 3]];  // argmax of each 2×2 block (row, col)
      const W = 4 * cs * 0.8 + 50 + 2 * cs + 60 + 4 * cs + 4, H = 4 * cs + 60;
      svg.setAttribute('width', W); svg.setAttribute('height', H);
      title(svg, 0, 18, 'max unpooling', 'put values back where the max was');
      const c8 = cs * 0.8, isMax = (r, c) => where.some(([a, b]) => a === r && b === c);
      g(svg, 0, 50 + (4 * cs - 4 * c8) / 2, A, (r, c, v) => ({ fill: isMax(r, c) ? 'rgba(242,177,52,0.25)' : C.bg2, text: v, color: isMax(r, c) ? C.fg : C.dim }), c8);
      svg.append(S('text', { x: 2 * c8, y: 50 + (4 * cs + 4 * c8) / 2 + 20, fill: C.dim, 'font-size': 13, 'text-anchor': 'middle' }, 'max pooling (encoder)'));
      const x1 = 4 * c8 + 50;
      in2(svg, x1, 50 + cs);
      svg.append(S('text', { x: x1 + cs, y: 50 + 3 * cs + 22, fill: C.dim, 'font-size': 13, 'text-anchor': 'middle' }, 'decoder input'));
      arrow(svg, x1 + 2 * cs + 8, 50 + 2 * cs, x1 + 2 * cs + 52);
      const x2 = x1 + 2 * cs + 60;
      const Y = [0, 1, 2, 3].map(r => [0, 1, 2, 3].map(c => { const k = where.findIndex(([a, b]) => a === r && b === c); return k < 0 ? 0 : X[k >> 1][k & 1]; }));
      g(svg, x2, 50, Y, (r, c, v) => ({ fill: v ? rgba(COL[r >> 1][c >> 1], 0.3) : C.bg2, text: v, color: v ? C.fg : C.dim }));
    }
    wrap.append(svg);
  }
  el.append(wrap);
}
