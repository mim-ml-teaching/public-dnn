// Why NMS fails on thin, tilted, packed objects: N parallel crayons, side by side, rotated by an angle. Each gets a
// correct detection (its axis-aligned box). The crayons never touch, but at 45° their boxes are big, mostly empty
// squares that overlap a lot, so greedy NMS deletes correct detections. Sliders: angle, IoU threshold.
// config: { n: 6, angle: 45, thr: 0.5, len: 230, thick: 22, gap: 14, width: 560 }
import { C, h } from './util.js';
import { mapHex, LIGHT } from '../theme.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; return e; };
const COLS = ['#ffd84d', '#b28dff', '#ff6b6b', '#5fd38d', '#5ab0ff', '#ff9f43', '#f78fb3'];
const SCORES = [0.91, 0.87, 0.95, 0.83, 0.89, 0.85, 0.9];
const iou = (a, b) => {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = iw * ih; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i);
};

export function mount(el, cfg) {
  el.classList.add('widget');
  const n = cfg.n || 6, L = cfg.len || 230, T = cfg.thick || 22, gap = cfg.gap || 14, VW = 460, VH = 400;
  const st = { angle: cfg.angle ?? 45, thr: cfg.thr ?? 0.5, nms: cfg.nms ?? true };
  const svg = S('svg', { viewBox: `0 0 ${VW} ${VH}`, width: cfg.width || 560, height: (cfg.width || 560) * VH / VW, style: 'display:block; font-family: Inter, Arial, sans-serif' });
  const readout = h('div', { class: 'readout', style: `min-height:2.5em; color:${C.fg}; line-height:1.35` });
  const col = c => (LIGHT ? mapHex(c) : c);
  function draw() {
    svg.replaceChildren();
    const a = st.angle * Math.PI / 180, ux = Math.cos(a), uy = -Math.sin(a), px = -uy, py = ux;   // along the crayon, across
    const step = T + gap, cx0 = VW / 2, cy0 = VH / 2;
    const cr = [];
    for (let i = 0; i < n; i++) {
      const o = (i - (n - 1) / 2) * step, cx = cx0 + px * o, cy = cy0 + py * o;
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([s, t]) => [cx + ux * s * L / 2 + px * t * T / 2, cy + uy * s * L / 2 + py * t * T / 2]);
      const xs = corners.map(c => c[0]), ys = corners.map(c => c[1]);
      cr.push({ cx, cy, corners, b: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], s: SCORES[i % SCORES.length], c: COLS[i % COLS.length] });
    }
    // greedy NMS
    const order = cr.map((_, i) => i).sort((i, j) => cr[j].s - cr[i].s), keep = new Array(n).fill(null);
    for (const i of order) { if (keep[i] !== null) continue; keep[i] = true; for (const j of order) if (keep[j] === null && iou(cr[i].b, cr[j].b) > st.thr) keep[j] = false; }
    cr.forEach((c, i) => {
      const gone = st.nms && !keep[i];
      // the crayon: body + tip
      const tip = [c.cx + ux * (L / 2 + 16), c.cy + uy * (L / 2 + 16)];
      const [p0, p1, p2, p3] = c.corners;
      svg.append(S('polygon', { points: [p3, p0, p1, tip, p2].map(p => p.join(',')).join(' '), fill: col(c.c), 'fill-opacity': gone ? 0.35 : 0.9, stroke: '#222', 'stroke-width': 1 }));
    });
    cr.forEach((c, i) => {
      const gone = st.nms && !keep[i], [x0, y0, x1, y1] = c.b;
      svg.append(S('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: 'none', stroke: gone ? col(C.red) : col(c.c), 'stroke-width': gone ? 1.5 : 2, 'stroke-dasharray': gone ? '6 4' : 'none', opacity: gone ? 0.8 : 0.95 }));
    });
    const nb = iou(cr[0].b, cr[1].b), kept = keep.filter(Boolean).length;
    readout.textContent = `${st.angle}°: neighbours' IoU ${nb.toFixed(2)}, ` + (st.nms ? `NMS keeps ${kept} of ${n}` : `${n} correct boxes`);
  }
  const angle = h('input', { type: 'range', min: 0, max: 90, step: 5, value: st.angle, oninput: () => { st.angle = +angle.value; draw(); } });
  const thr = h('input', { type: 'range', min: 0.3, max: 0.95, step: 0.05, value: st.thr, oninput: () => { st.thr = +thr.value; thv.textContent = st.thr.toFixed(2); draw(); } });
  const thv = h('span', { class: 'readout', style: 'display:inline-block; width:3em' }, st.thr.toFixed(2));
  const bBefore = h('button', { onclick: () => { st.nms = false; bBefore.classList.add('active'); bAfter.classList.remove('active'); draw(); } }, 'before NMS');
  const bAfter = h('button', { onclick: () => { st.nms = true; bAfter.classList.add('active'); bBefore.classList.remove('active'); draw(); } }, 'after NMS');
  (st.nms ? bAfter : bBefore).classList.add('active');
  el.append(svg, readout, h('div', { class: 'wctl interactive-only' }, bBefore, bAfter, h('label', { style: 'margin-left:0.8em' }, 'angle'), angle,
    h('label', { style: 'margin-left:0.8em' }, 'IoU threshold'), thr, thv));
  draw();
}
