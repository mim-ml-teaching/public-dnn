// Architecture diagrams as a row of 3D boxes ("volumes" H×W×C), dense layers as thin bars, and labelled
// arrows for the operations in between. Box sizes grow with the square root of the size (spatial size →
// height and depth, channels → thickness), so 224 and 7, or 3 and 2048 channels, fit on one slide.
// config: {
//   items: [ { type: "vol", h: 32, w: 32, c: 3, label: "32×32×3", sub: "input", color: "#5ab0ff",
//              kernel: { k: 5, label: "5×5×3" } },
//            { type: "op", text: "conv 5×5", sub: "6 filters" },
//            { type: "fc", n: 120, label: "120" } ],
//   width: 1180, height: 300 (the drawing is scaled to fit), hs: 9, cs: 2.2, gap: 12, maxScale: 2.2
// }
import { C } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};

export function mount(el, cfg) {
  el.classList.add('widget');
  const hs = cfg.hs || 9, cs = cfg.cs || 2.2, gap = cfg.gap ?? 12, Wmax = cfg.width || 1180, Hmax = cfg.height || 300;
  const sp = s => hs * Math.sqrt(s);                 // spatial size → pixels
  const ch = c => Math.max(3, cs * Math.sqrt(c));    // channels → thickness
  const items = cfg.items || [];
  // measure
  const dims = items.map(it => {
    if (it.type === 'vol') { const f = sp(it.h), d = sp(it.w) * 0.42; return { w: ch(it.c) + d, top: f / 2 + d, bot: f / 2 }; }
    if (it.type === 'fc') { const f = it.size || Math.min(220, 6 * Math.sqrt(it.n)); return { w: 14 + 6, top: f / 2 + 6, bot: f / 2 }; }
    if (it.type === 'op') return { w: it.width || 74, top: 0, bot: 0 };
    return { w: it.width || 20, top: 0, bot: 0 };
  });
  const above = Math.max(30, ...dims.map(d => d.top)) + 34, below = Math.max(30, ...dims.map(d => d.bot)) + 54;
  const M = 46, totalW = dims.reduce((a, d) => a + d.w + gap, 0) - gap + 2 * M, totalH = above + below;
  const sc = Math.min(cfg.maxScale || 2.2, Wmax / totalW, Hmax / totalH);   // fit the box, up or down
  const svg = S('svg', { width: totalW * sc, height: totalH * sc, viewBox: `0 0 ${totalW} ${totalH}`, style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const yc = above;
  let x = M;
  const text = (x0, y0, t, color, size = 15, weight = 400, anchor = 'middle') =>
    svg.append(S('text', { x: x0, y: y0, fill: color, 'font-size': size, 'font-weight': weight, 'text-anchor': anchor }, t));
  items.forEach((it, i) => {
    const d = dims[i];
    if (it.type === 'vol' || it.type === 'fc') {
      const color = it.color || (it.type === 'fc' ? C.purple : C.blue);
      let f, t, dd;
      if (it.type === 'vol') { f = sp(it.h); t = ch(it.c); dd = sp(it.w) * 0.42; } else { f = it.size || Math.min(220, 6 * Math.sqrt(it.n)); t = 14; dd = 6; }
      const x0 = x, y0 = yc - f / 2;
      const poly = (pts, fill, op) => svg.append(S('polygon', { points: pts.map(p => p.join(',')).join(' '), fill, 'fill-opacity': op, stroke: color, 'stroke-width': it.highlight ? 2.6 : 1.3 }));
      poly([[x0, y0], [x0 + dd, y0 - dd], [x0 + t + dd, y0 - dd], [x0 + t, y0]], color, 0.45);           // top
      poly([[x0 + t, y0], [x0 + t + dd, y0 - dd], [x0 + t + dd, y0 + f - dd], [x0 + t, y0 + f]], color, 0.18); // side
      poly([[x0, y0], [x0 + t, y0], [x0 + t, y0 + f], [x0, y0 + f]], color, 0.3);                          // front
      if (it.kernel) {
        const kf = sp(it.kernel.k), kd = kf * 0.42, kx = x0, ky = y0 + f * 0.18;
        const kc = C.accent;
        const kp = (pts, op) => svg.append(S('polygon', { points: pts.map(p => p.join(',')).join(' '), fill: kc, 'fill-opacity': op, stroke: kc, 'stroke-width': 1.6 }));
        kp([[kx, ky], [kx + kd, ky - kd], [kx + t + kd, ky - kd], [kx + t, ky]], 0.6);
        kp([[kx + t, ky], [kx + t + kd, ky - kd], [kx + t + kd, ky + kf - kd], [kx + t, ky + kf]], 0.35);
        kp([[kx, ky], [kx + t, ky], [kx + t, ky + kf], [kx, ky + kf]], 0.5);
        if (it.kernel.label) text(kx - 6, ky + kf / 2 + 5, it.kernel.label, kc, 14, 600, 'end');
      }
      const lx = x0 + (t + dd) / 2;
      if (it.label) text(lx, yc + f / 2 + 22, it.label, C.fg, it.labelSize || 15, 600);
      if (it.sub) text(lx, yc + f / 2 + 40, it.sub, C.dim, 13);
      if (it.top) text(lx, y0 - dd - 10, it.top, color, 13, 600);
    } else if (it.type === 'op') {
      const x1 = x + 4, x2 = x + d.w - 4;
      svg.append(S('line', { x1, y1: yc, x2: x2 - 6, y2: yc, stroke: '#5a606c', 'stroke-width': 2 }));
      svg.append(S('polygon', { points: `${x2},${yc} ${x2 - 9},${yc - 5} ${x2 - 9},${yc + 5}`, fill: '#5a606c' }));
      if (it.text) text((x1 + x2) / 2, yc - 10, it.text, C.accent, it.size || 14, 600);
      if (it.sub) text((x1 + x2) / 2, yc + 22, it.sub, C.dim, 12);
    } else if (it.type === 'text') {
      text(x + d.w / 2, yc + 6, it.text, it.color || C.dim, it.size || 18, 600);
    }
    x += d.w + gap;
  });
  el.append(svg);
}
