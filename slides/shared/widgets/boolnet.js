// Two layers of perceptrons compute any Boolean function: the OR-of-ANDs construction on
// f(x1, x2, x3) = "exactly one input is 1". Click a truth-table row to feed it to the network.
//   hidden unit for pattern p:  [ Σ_i s_i x_i ≥ (number of 1s in p) ],  s_i = +1 if p_i = 1 else −1
//   output:                     [ h_1 + h_2 + h_3 ≥ 1 ]
// config: { row: 4 }
import { C, h } from './util.js';

const PATTERNS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];           // inputs where f = 1
const step = z => (z >= 0 ? 1 : 0);
const hidden = (p, x) => step(p.reduce((s, pi, i) => s + (pi ? 1 : -1) * x[i], 0) - p.reduce((a, b) => a + b, 0));

export function mount(el, cfg) {
  const st = { row: cfg.row ?? 4 };                          // index into the 8 rows (4 = 1,0,0)
  const rows = Array.from({ length: 8 }, (_, k) => [(k >> 2) & 1, (k >> 1) & 1, k & 1]);
  const f = x => (x[0] + x[1] + x[2] === 1 ? 1 : 0);
  const table = h('table', { style: 'font-size:1.15em; border-collapse:collapse' });
  const ns = 'http://www.w3.org/2000/svg';
  const S = (tag, a = {}, text) => { const e = document.createElementNS(ns, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };
  const W = 600, H = 330;
  const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, width: 700, height: Math.round(700 * H / W), style: 'font-family: Inter, Arial, sans-serif' });

  function draw() {
    const x = rows[st.row];
    // truth table
    table.replaceChildren(h('tr', {}, ...['x₁', 'x₂', 'x₃', 'f'].map(t => h('th', { style: 'padding:0.15em 0.7em; text-align:center' }, t))));
    rows.forEach((r, k) => {
      const tr = h('tr', { style: `cursor:pointer; ${k === st.row ? 'outline:2px solid #5ab0ff;' : ''} ${f(r) ? 'background:rgba(242,177,52,0.15);' : ''}` },
        ...r.map(v => h('td', { style: 'padding:0.1em 0.7em; text-align:center' }, String(v))),
        h('td', { style: `padding:0.1em 0.7em; text-align:center; font-weight:700; color:${f(r) ? C.accent : C.dim}` }, String(f(r))));
      tr.addEventListener('click', () => { st.row = k; draw(); });
      table.append(tr);
    });
    // network
    svg.replaceChildren();
    const IX = 60, HX = 300, OX = 520, iy = [70, 165, 260], hy = [70, 165, 260], oy = 165;
    const hv = PATTERNS.map(p => hidden(p, x)), y = step(hv.reduce((a, b) => a + b, 0) - 1);
    PATTERNS.forEach((p, j) => p.forEach((pi, i) => {
      const w = pi ? 1 : -1;
      svg.append(S('line', { x1: IX + 22, y1: iy[i], x2: HX - 34, y2: hy[j], stroke: w > 0 ? '#5fd38d' : '#ff6b6b', 'stroke-width': x[i] ? 2.6 : 1.2, opacity: x[i] ? 1 : 0.45 }));
    }));
    hv.forEach((v, j) => svg.append(S('line', { x1: HX + 34, y1: hy[j], x2: OX - 26, y2: oy, stroke: '#9aa1ae', 'stroke-width': v ? 2.6 : 1.2, opacity: v ? 1 : 0.45 })));
    const node = (cx, cy, txt, sub, on, color, w = 44) => {
      svg.append(S('rect', { x: cx - w / 2, y: cy - 22, width: w, height: 44, rx: 10, fill: on ? color : '#272b34', 'fill-opacity': on ? 0.35 : 1, stroke: color, 'stroke-width': 2 }));
      svg.append(S('text', { x: cx, y: cy + 6, fill: C.fg, 'font-size': 17, 'font-weight': 600, 'text-anchor': 'middle' }, txt));
      if (sub) svg.append(S('text', { x: cx, y: cy + 38, fill: C.dim, 'font-size': 13, 'text-anchor': 'middle' }, sub));
    };
    x.forEach((v, i) => node(IX, iy[i], String(v), `x${'₁₂₃'[i]}`, v, '#5ab0ff'));
    PATTERNS.forEach((p, j) => node(HX, hy[j], `h${'₁₂₃'[j]} = ${hv[j]}`, `fires only on ${p.join('')}  (≥ 1)`, hv[j], '#b28dff', 68));
    node(OX, oy, `y = ${y}`, 'OR  (h₁+h₂+h₃ ≥ 1)', y, '#f2b134', 70);
  }
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:1.2em; align-items:flex-start' },
    h('div', {}, table, h('div', { class: 'dim', style: 'margin-top:0.4em' }, 'click a row'),
      h('div', { style: 'margin-top:0.8em; display:flex; flex-direction:column; gap:0.3em; font-size:1.05em' },
        h('div', { style: 'display:flex; align-items:center; gap:0.5em' }, h('span', { style: 'display:inline-block; width:28px; height:0; border-top:3px solid #5fd38d' }), 'weight +1'),
        h('div', { style: 'display:flex; align-items:center; gap:0.5em' }, h('span', { style: 'display:inline-block; width:28px; height:0; border-top:3px solid #ff6b6b' }), 'weight −1'),
        h('div', { class: 'dim', style: 'font-size:0.9em' }, 'thick: input is 1'))), svg));
  draw();
}
