// Bar chart(s): one or more panels side by side or stacked, values printed on the bars, optional horizontal
// reference line, labels under the bars (rotated when there are many). Data inline or from a JSON file.
// config: {
//   panels: [{ title, bars: [{ label, value, color, note }], ymax, yfmt: "pct"|"num"|"M"|"MB", hline: { y, label (string or lines), side: "right" } }],
//   layout: "row" | "column", width: 1100, height: 330, barColor, zoom: 1 (>1 = larger text and bars)
// }
import { C, fmt } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const F = {
  pct: v => `${fmt(v, 1)}%`,
  num: v => fmt(v, 2),
  M: v => (v >= 1e6 ? `${fmt(v / 1e6, 1)}M` : v >= 1e3 ? `${fmt(v / 1e3, 0)}k` : `${v}`),
};

export function mount(el, cfg) {
  el.classList.add('widget');
  const panels = cfg.panels, col = cfg.layout === 'column';
  const W = cfg.width || 1100, H = cfg.height || 330, z = cfg.zoom || 1;   // zoom: drawn at W/z × H/z, shown at W × H
  const VW = W / z, VH = H / z;
  const pw = col ? VW : (VW - 30 * (panels.length - 1)) / panels.length, ph = col ? (VH - 20 * (panels.length - 1)) / panels.length : VH;
  const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${VW} ${VH}`, style: 'font-family: Inter, Arial, sans-serif; display:block' });
  panels.forEach((p, k) => {
    const x0 = col ? 0 : k * (pw + 30), y0 = col ? k * (ph + 20) : 0, bars = p.bars, many = bars.length > 10;
    const M = { l: 14, r: p.hline?.side === 'right' ? 150 : 14, t: p.title ? 34 : 12, b: p.hideLabels ? 10 : many ? 70 : bars.some(b => b.note) ? 58 : 44 };
    const f = F[p.yfmt || 'num'];
    const ymax = p.ymax || Math.max(...bars.map(b => b.value)) * 1.15;
    const bw = (pw - M.l - M.r) / bars.length;
    const Y = v => y0 + M.t + (ph - M.t - M.b) * (1 - v / ymax);
    if (p.title) svg.append(S('text', { x: x0 + M.l, y: y0 + 20, fill: C.fg, 'font-size': 18, 'font-weight': 600 }, p.title));
    svg.append(S('line', { x1: x0 + M.l, x2: x0 + pw - M.r, y1: Y(0), y2: Y(0), stroke: C.line, 'stroke-width': 1.5 }));
    bars.forEach((b, i) => {
      const bx = x0 + M.l + i * bw, gap = Math.min(10, bw * 0.18), color = b.color || p.color || cfg.barColor || C.blue;
      svg.append(S('rect', { x: bx + gap / 2, y: Y(b.value), width: bw - gap, height: Y(0) - Y(b.value), fill: color, rx: 3 }));
      if (!p.hideValues && (!many || b.value > ymax * 0.04))
        svg.append(S('text', { x: bx + bw / 2, y: Y(b.value) - 6, fill: C.fg, 'font-size': many ? 12 : 15, 'font-weight': 600, 'text-anchor': 'middle' }, b.text ?? f(b.value)));
      if (!p.hideLabels) {
        const lx = bx + bw / 2, ly = Y(0) + 18;
        svg.append(many
          ? S('text', { x: lx, y: ly - 4, fill: C.dim, 'font-size': 12, 'text-anchor': 'end', transform: `rotate(-50 ${lx} ${ly - 4})` }, b.label)
          : S('text', { x: lx, y: ly, fill: C.dim, 'font-size': 14, 'text-anchor': 'middle' }, b.label));
        if (b.note && !many) svg.append(S('text', { x: lx, y: ly + 18 + (i % 2) * 16, fill: b.color || C.fg, 'font-size': 14, 'font-weight': 600, 'text-anchor': 'middle' }, b.note));
      }
    });
    if (p.hline) {
      svg.append(S('line', { x1: x0 + M.l, x2: x0 + pw - M.r, y1: Y(p.hline.y), y2: Y(p.hline.y), stroke: C.accent, 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }));
      const right = p.hline.side === 'right';
      (Array.isArray(p.hline.label) ? p.hline.label : [p.hline.label]).forEach((t, j) =>
        svg.append(S('text', { x: right ? x0 + pw - M.r + 10 : x0 + pw - M.r, y: right ? Y(p.hline.y) + 5 + 18 * j : Y(p.hline.y) - 6 - 18 * j, fill: C.accent, 'font-size': 15, 'text-anchor': right ? 'start' : 'end' }, t)));
    }
  });
  el.append(svg);
}
