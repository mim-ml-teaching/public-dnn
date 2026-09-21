// Batch normalization: which values share one mean and variance.
// mlp: true  → two panels: an MLP neuron (one number per example: μ, σ² over N values) and one conv channel
//              (one H×W map per example: μ, σ² over N·H·W values).
// channels: C → a table, one row per channel, one column per image of the batch; each row (a channel over all
//              images and positions) is one group with its own μ_c, σ²_c, γ_c, β_c. Hover a row to highlight it.
// config: { mlp: false, channels: 3, n: 4, size: 4, cell: 17 }
import { C } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const s = (tag, attrs = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const COLS = [C.red, C.green, C.blue, C.purple];

function rng(seed) { let x = seed >>> 0; return () => ((x = (x * 1664525 + 1013904223) >>> 0) / 4294967296); }

export function mount(el, cfg) {
  el.classList.add('widget');
  const N = cfg.n || 4, K = cfg.size || 4, cell = cfg.cell || 17, G = K * cell, gap = 14, rand = rng(5);
  const grid = (g, x, y, color, k = K) => {
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++)
      g.append(s('rect', { x: x + j * cell, y: y + i * cell, width: cell - 1.5, height: cell - 1.5, rx: 2, fill: color, 'fill-opacity': (0.2 + 0.7 * rand()).toFixed(2) }));
    g.append(s('rect', { x: x - 1, y: y - 1, width: k * cell + 0.5, height: k * cell + 0.5, rx: 3, fill: 'none', stroke: color, 'stroke-width': 1.5 }));
  };
  const brace = (g, x0, x1, y, color) => g.append(s('path', { d: `M${x0} ${y} q0 8 8 8 L${(x0 + x1) / 2 - 6} ${y + 8} q6 0 6 8 q0 -8 6 -8 L${x1 - 8} ${y + 8} q8 0 8 -8`, fill: 'none', stroke: color, 'stroke-width': 2 }));
  const txt = (g, x, y, t, a = {}) => g.append(s('text', { x, y, fill: C.fg, 'font-size': 16, 'font-family': 'Inter, Arial', 'text-anchor': 'middle', ...a }, t));

  let W, H; const root = s('g');
  if (cfg.mlp) {
    // left: MLP neuron j, one value per example
    const lx = 10, top = 40, sc = 30;
    txt(root, lx + (N * (sc + gap) - gap) / 2, 20, 'MLP: neuron j', { fill: C.dim });
    for (let n = 0; n < N; n++) {
      const x = lx + n * (sc + gap);
      root.append(s('rect', { x, y: top + (G - sc) / 2, width: sc, height: sc, rx: 3, fill: C.accent, 'fill-opacity': (0.3 + 0.6 * rand()).toFixed(2), stroke: C.accent, 'stroke-width': 1.5 }));
      txt(root, x + sc / 2, top + G + 22, `x${n + 1}`, { 'font-size': 13, fill: C.dim });
    }
    const lw = N * (sc + gap) - gap;
    brace(root, lx, lx + lw, top + G + 32, C.accent);
    txt(root, lx + lw / 2, top + G + 68, 'one μ, σ²:', { fill: C.accent });
    txt(root, lx + lw / 2, top + G + 90, `N = ${N} values`, { fill: C.accent });
    // right: one conv channel, one H×W map per example
    const rx = lx + lw + 90;
    txt(root, rx + (N * (G + gap) - gap) / 2, 20, 'convnet: one channel', { fill: C.dim });
    for (let n = 0; n < N; n++) {
      grid(root, rx + n * (G + gap), top, C.blue);
      txt(root, rx + n * (G + gap) + G / 2, top + G + 22, `image ${n + 1}`, { 'font-size': 13, fill: C.dim });
    }
    const rw = N * (G + gap) - gap;
    brace(root, rx, rx + rw, top + G + 32, C.blue);
    txt(root, rx + rw / 2, top + G + 68, 'one μ, σ²:', { fill: C.blue });
    txt(root, rx + rw / 2, top + G + 90, `N·H·W = ${N}·${K}·${K} = ${N * K * K} values`, { fill: C.blue });
    root.append(s('line', { x1: rx - 45, y1: 10, x2: rx - 45, y2: top + G + 90, stroke: C.line, 'stroke-width': 1.5, 'stroke-dasharray': '4 4' }));
    W = rx + rw + 10; H = top + G + 100;
  } else {
    const Cn = cfg.channels || 3, lw = 100, top = 34, rowH = G + 18, rows = [];
    for (let n = 0; n < N; n++) txt(root, lw + n * (G + gap) + G / 2, 20, `image ${n + 1}`, { 'font-size': 14, fill: C.dim });
    const rx = lw + N * (G + gap) + 4;
    for (let c = 0; c < Cn; c++) {
      const g = s('g', { style: 'cursor:pointer; transition:opacity 0.15s' }), y = top + c * rowH, col = COLS[c % COLS.length];
      g.append(s('rect', { x: 0, y: y - 6, width: rx + 210, height: G + 12, rx: 8, fill: col, 'fill-opacity': 0.06 }));
      txt(g, lw - 14, y + G / 2 + 6, `channel ${c + 1}`, { 'text-anchor': 'end', fill: col, 'font-size': 15 });
      for (let n = 0; n < N; n++) grid(g, lw + n * (G + gap), y, col);
      g.append(s('path', { d: `M${rx} ${y} q10 0 10 10 L${rx + 10} ${y + G / 2 - 8} q0 8 8 8 q-8 0 -8 8 L${rx + 10} ${y + G - 10} q0 10 -10 10`, fill: 'none', stroke: col, 'stroke-width': 2 }));
      txt(g, rx + 26, y + G / 2 - 4, `μ${'₁₂₃₄'[c]}, σ²${'₁₂₃₄'[c]}  (${N * K * K} values)`, { 'text-anchor': 'start', fill: col, 'font-size': 15 });
      txt(g, rx + 26, y + G / 2 + 16, `γ${'₁₂₃₄'[c]}, β${'₁₂₃₄'[c]}`, { 'text-anchor': 'start', fill: col, 'font-size': 15 });
      rows.push(g); root.append(g);
    }
    rows.forEach((g, c) => {
      g.addEventListener('mouseenter', () => rows.forEach((o, k) => { o.style.opacity = k === c ? 1 : 0.25; }));
      g.addEventListener('mouseleave', () => rows.forEach(o => { o.style.opacity = 1; }));
    });
    W = rx + 215; H = top + Cn * rowH;
  }
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, style: `width:100%; max-width:${cfg.maxWidth || W * 1.25}px; display:block` });
  svg.append(root);
  el.append(svg);
}
