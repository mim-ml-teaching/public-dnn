// Why local + shared weights give translation equivariance, in 1D (sequence → sequence).
// Two panels: "local + shared" (one kernel w₁ w₂ w₃ used at every position = a convolution) and "local, own
// weights" (every output position has its own three weights). Each panel shows the input with a small pattern
// and its output, then the same input shifted by s and its output. Lines from the pattern to the outputs are
// coloured by weight identity: in the shared panel the shifted pattern meets the same three colours, so the
// output is the old output shifted by s; in the unshared panel it meets different weights → different numbers.
// config: { n: 10, pattern: [1,3,2], pos: 1, shift: 3, kernel: [-1,2,1], cell: 44, width: 1180 }
import { C, h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const sgn = v => (v < 0 ? `−${-v}` : `${v}`);

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

export function mount(el, cfg) {
  el.classList.add('widget');
  const n = cfg.n || 10, pat = cfg.pattern || [1, 3, 2], w = cfg.kernel || [-1, 2, 1];
  const cs = cfg.cell || 44, W = cfg.width || 1180;
  const st = { pos: cfg.pos ?? 1, shift: cfg.shift ?? 3 };
  // unshared: a fixed random triple of small non-zero integers per output position
  const rand = rng(7), vals = [-2, -1, 1, 2, 3];
  const U = Array.from({ length: n }, () => [0, 1, 2].map(() => vals[Math.floor(rand() * vals.length)]));
  const sharedCol = [C.red, C.green, C.purple];
  const hue = (o, d) => `hsl(${(o * 3 + d) * 37 % 360}, 70%, 64%)`;

  const svg = S('svg', { style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const ctl = h('div', { class: 'wctl interactive-only' });
  el.append(svg, ctl);

  const input = p => Array.from({ length: n }, (_, i) => (i >= p && i < p + pat.length ? pat[i - p] : 0));
  const out = (x, shared) => Array.from({ length: n }, (_, o) => {
    let s = 0;
    for (let d = 0; d < 3; d++) { const i = o + d - 1; if (i >= 0 && i < n) s += (shared ? w[d] : U[o][d]) * x[i]; }
    return s;
  });

  function draw() {
    svg.replaceChildren();
    const pw = W / 2, rowGap = 74, blockGap = 46;
    const H = 30 + 2 * (2 * cs + rowGap) + blockGap + 64;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    [true, false].forEach((shared, k) => {
      const x0 = k * pw + (pw - n * cs) / 2 + 30;
      const cx = i => x0 + i * cs + cs / 2;
      svg.append(S('text', { x: k * pw + pw / 2 + 30, y: 20, fill: C.accent, 'font-size': 20, 'font-weight': 600, 'text-anchor': 'middle' },
        shared ? 'local + shared (convolution)' : 'local, own weights at every position'));
      const outs = [];
      [st.pos, st.pos + st.shift].forEach((p, b) => {
        const yOut = 34 + b * (2 * cs + rowGap + blockGap), yIn = yOut + cs + rowGap;
        const x = input(p), y = out(x, shared);
        outs.push(y);
        // lines from non-zero inputs to the outputs that read them
        for (let o = 0; o < n; o++) for (let d = 0; d < 3; d++) {
          const i = o + d - 1;
          if (i < 0 || i >= n) continue;
          const active = x[i] !== 0;
          svg.append(S('line', { x1: cx(i), y1: yIn, x2: cx(o), y2: yOut + cs,
            stroke: active ? (shared ? sharedCol[d] : hue(o, d)) : '#3a3f4b', 'stroke-width': active ? 2.6 : 1, opacity: active ? 1 : 0.7 }));
        }
        const row = (vals, y0, color, label) => {
          svg.append(S('text', { x: x0 - 12, y: y0 + cs / 2 + 5, fill: C.dim, 'font-size': 15, 'text-anchor': 'end' }, label));
          vals.forEach((v, i) => {
            svg.append(S('rect', { x: x0 + i * cs, y: y0, width: cs, height: cs, fill: v ? color : '#1e2128', 'fill-opacity': v ? 0.28 : 1, stroke: v ? color : '#3a3f4b', 'stroke-width': v ? 2 : 1 }));
            svg.append(S('text', { x: x0 + i * cs + cs / 2, y: y0 + cs / 2 + 6, fill: v ? C.fg : C.dim, 'font-size': 17, 'text-anchor': 'middle', 'font-weight': v ? 700 : 400 }, sgn(v)));
          });
        };
        row(y, yOut, C.blue, b ? 'output′' : 'output');
        row(x, yIn, C.accent, b ? 'input′' : 'input');
        if (b) svg.append(S('text', { x: x0 + n * cs / 2, y: yOut - 12, fill: C.dim, 'font-size': 15, 'text-anchor': 'middle' }, `input′ = input shifted by ${st.shift}`));
      });
      // verdict: is output′ the output shifted by s?
      const sh = Array.from({ length: n }, (_, i) => (i - st.shift >= 0 ? outs[0][i - st.shift] : 0));
      const ok = sh.every((v, i) => v === outs[1][i]);
      svg.append(S('text', { x: k * pw + pw / 2 + 30, y: H - 10, fill: ok ? C.green : C.red, 'font-size': 18, 'font-weight': 600, 'text-anchor': 'middle' },
        ok ? `✓ output′ = output shifted by ${st.shift}` : `✗ output′ ≠ output shifted by ${st.shift}`));
    });
    // legend: the shared weights
    w.forEach((v, d) => svg.append(S('text', { x: W / 4 + 30 + (d - 1) * 90, y: H - 34, fill: sharedCol[d], 'font-size': 16, 'font-weight': 700, 'text-anchor': 'middle' }, `w${'₁₂₃'[d]} = ${sgn(v)}`)));
    svg.append(S('text', { x: 3 * W / 4 + 30, y: H - 34, fill: C.dim, 'font-size': 15, 'text-anchor': 'middle' }, `${3 * n - 2} different weights (one colour each)`));
    if (W > 0) svg.append(S('line', { x1: W / 2 + 30, y1: 30, x2: W / 2 + 30, y2: H - 40, stroke: '#3a3f4b', 'stroke-width': 1 }));
  }

  const maxShift = n - pat.length - st.pos;
  const b = (t, f) => h('button', { onclick: () => { f(); draw(); } }, t);
  ctl.append(h('label', {}, 'shift '), b('◀', () => { st.shift = Math.max(1, st.shift - 1); }), b('▶', () => { st.shift = Math.min(maxShift, st.shift + 1); }));
  draw();
}
