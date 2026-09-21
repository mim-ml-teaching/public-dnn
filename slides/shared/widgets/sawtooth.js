// Depth folds space: the tent map t(x) = 2·relu(x) − 4·relu(x − ½) on [0, 1] uses 2 ReLU units.
// Composing it k times (k layers, 2k units) gives a sawtooth with 2^k linear pieces; a single hidden
// layer needs about 2^k units for the same function.
// config: { depth: 3, width, height }
import { C, h, isPrint } from './util.js';

const relu = z => Math.max(0, z);
const tent = x => 2 * relu(x) - 4 * relu(x - 0.5);

export function mount(el, cfg) {
  const W = cfg.width || 600, H = cfg.height || 330;
  const state = { k: cfg.depth ?? 3 };
  const canvas = h('canvas', { width: W, height: H });
  const g = canvas.getContext('2d');
  const strip = h('canvas', { width: W, height: 60 });
  const gs = strip.getContext('2d');
  const readout = h('div', { style: 'line-height:1.6' });

  function draw() {
    const pad = 28, sx = x => pad + x * (W - 2 * pad), sy = y => H - pad - y * (H - 2 * pad);
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#2d313a'; g.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
    const comp = (x, k) => { for (let i = 0; i < k; i++) x = tent(x); return x; };
    const M = 2400;
    const plot = (k, color, width) => {
      g.strokeStyle = color; g.lineWidth = width; g.beginPath();
      for (let i = 0; i <= M; i++) { const x = i / M; i ? g.lineTo(sx(x), sy(comp(x, k))) : g.moveTo(sx(x), sy(comp(x, k))); }
      g.stroke();
    };
    plot(state.k, C.accent, 2.5);
    g.font = '13px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText(`depth ${state.k}: ${2 ** state.k} linear pieces`, pad, 18);

    // network strip: k layers of 2 units
    gs.fillStyle = C.bg; gs.fillRect(0, 0, W, 60);
    const nodes = [[1]].concat(Array.from({ length: state.k }, () => [0, 1])).concat([[1]]);
    const cx = i => 30 + i * (W - 60) / (nodes.length - 1);
    const cy = (n, j) => (n.length === 1 ? 30 : 16 + j * 28);
    gs.strokeStyle = '#4a505c'; gs.lineWidth = 1;
    for (let i = 0; i + 1 < nodes.length; i++)
      nodes[i].forEach((_, a) => nodes[i + 1].forEach((_, b) => { gs.beginPath(); gs.moveTo(cx(i), cy(nodes[i], a)); gs.lineTo(cx(i + 1), cy(nodes[i + 1], b)); gs.stroke(); }));
    nodes.forEach((n, i) => n.forEach((_, j) => {
      gs.fillStyle = i === 0 || i === nodes.length - 1 ? C.dim : C.purple;
      gs.beginPath(); gs.arc(cx(i), cy(n, j), 6, 0, 7); gs.fill();
    }));

    readout.innerHTML =
      `<div>layers: <b>${state.k}</b></div><div>ReLU units: <b>${2 * state.k}</b></div>` +
      `<div>linear pieces: <b style="color:${C.accent}">${2 ** state.k}</b></div>` +
      `<div class="dim" style="margin-top:0.4em">one hidden layer needs ≈ ${2 ** state.k} units for this function</div>`;
  }

  const slider = h('input', { type: 'range', min: 1, max: 7, step: 1, value: state.k });
  slider.addEventListener('input', () => { state.k = +slider.value; draw(); });
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' },
    h('div', {}, canvas, strip),
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:13em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only' }, h('label', {}, 'depth '), slider))));
  if (isPrint()) state.k = cfg.depth ?? 4;
  draw();
}
