// Intersection over union of two boxes: the ground-truth box (green) is fixed, the predicted box (amber) can be
// dragged (inside) or resized (bottom-right corner). Presets give the poor / good / excellent examples.
// config: { width: 560, height: 330, pred: [x, y, w, h], gt: [x, y, w, h] }
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
  const W = cfg.width || 420, H = cfg.height || 280;
  const gt = cfg.gt || [150, 50, 150, 190];
  const P = { poor: [210, 120, 150, 170], good: [170, 70, 150, 190], excellent: [154, 54, 150, 190] };
  let pr = (cfg.pred || P.good).slice();
  const svg = S('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: `display:block; flex:none; background:${C.bg2}; border:1px solid ${C.line}; border-radius:8px; touch-action:none; font-family: Inter, Arial, sans-serif` });
  const out = h('div', { style: 'font-size:1.15em; margin-left:0.8em; width:10.5em; flex:none' });
  const ctl = h('div', { class: 'wctl interactive-only' }, ...Object.keys(P).map(k => h('button', { onclick: () => { pr = P[k].slice(); draw(); } }, k)));
  el.append(h('div', { style: 'display:flex; align-items:center' }, svg, out), ctl);
  let drag = null;
  function draw() {
    svg.replaceChildren();
    const [gx, gy, gw, gh] = gt, [px, py, pw, ph] = pr;
    const ix1 = Math.max(gx, px), iy1 = Math.max(gy, py), ix2 = Math.min(gx + gw, px + pw), iy2 = Math.min(gy + gh, py + ph);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1), uni = gw * gh + pw * ph - inter, v = inter / uni;
    svg.append(S('rect', { x: gx, y: gy, width: gw, height: gh, fill: C.green, 'fill-opacity': 0.12, stroke: C.green, 'stroke-width': 3 }));
    svg.append(S('rect', { x: px, y: py, width: pw, height: ph, fill: C.accent, 'fill-opacity': 0.12, stroke: C.accent, 'stroke-width': 3, style: 'cursor:move' }));
    if (inter > 0) svg.append(S('rect', { x: ix1, y: iy1, width: ix2 - ix1, height: iy2 - iy1, fill: C.blue, 'fill-opacity': 0.45, 'pointer-events': 'none' }));
    svg.append(S('rect', { x: px + pw - 7, y: py + ph - 7, width: 14, height: 14, fill: C.accent, style: 'cursor:nwse-resize', 'data-h': 1 }));
    svg.append(S('text', { x: gx, y: gy - 8, fill: C.green, 'font-size': 16, 'font-weight': 600 }, 'ground truth'));
    svg.append(S('text', { x: px + pw, y: py - 8, fill: C.accent, 'font-size': 16, 'font-weight': 600, 'text-anchor': 'end' }, 'prediction'));
    const verdict = v >= 0.9 ? 'excellent' : v >= 0.7 ? 'good' : v >= 0.5 ? 'counts as a hit (≥ 0.5)' : 'a miss (< 0.5)';
    out.innerHTML = '';
    out.append(
      h('div', { class: 'readout' }, h('span', { style: `color:${C.blue}` }, 'intersection'), ` ${Math.round(inter)}`),
      h('div', { class: 'readout' }, 'union ', `${Math.round(uni)}`),
      h('div', { style: 'font-size:1.5em; font-weight:700; margin-top:0.2em; color:' + C.accent }, `IoU = ${v.toFixed(2)}`),
      h('div', { class: 'small' }, verdict));
  }
  const pt = e => { const r = svg.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; };
  svg.addEventListener('pointerdown', e => {
    const [x, y] = pt(e), [px, py, pw, ph] = pr;
    if (Math.abs(x - px - pw) < 14 && Math.abs(y - py - ph) < 14) drag = { mode: 'size', x, y, p: pr.slice() };
    else if (x > px && x < px + pw && y > py && y < py + ph) drag = { mode: 'move', x, y, p: pr.slice() };
    if (drag) svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', e => {
    if (!drag) return;
    const [x, y] = pt(e), dx = x - drag.x, dy = y - drag.y, p = drag.p;
    pr = drag.mode === 'move' ? [p[0] + dx, p[1] + dy, p[2], p[3]] : [p[0], p[1], Math.max(20, p[2] + dx), Math.max(20, p[3] + dy)];
    draw();
  });
  svg.addEventListener('pointerup', () => { drag = null; });
  draw();
}
