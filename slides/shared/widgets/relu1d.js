// One hidden layer of ReLUs in 1D: f(x) = c + Σ_k a_k · relu(w_k x + b_k) is piecewise linear with a
// "joint" at x = −b_k / w_k for every unit. "fit" trains all parameters with gradient descent (Adam)
// on the target curve, live in the browser.
// config: { units: 6, target: "wave" | "step" | "bump", width, height }
import { C, h, fmt, isPrint } from './util.js';

const TARGETS = {
  wave: x => Math.sin(3.2 * x) + 0.35 * x,
  bump: x => Math.exp(-12 * x * x) * 1.4 - 0.2,
  step: x => (x > 0.15 ? 0.9 : -0.6),
};

function rng(seed) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

export function mount(el, cfg) {
  const W = cfg.width || 620, H = cfg.height || 380;
  const state = { n: cfg.units ?? 6, target: cfg.target || 'wave', seed: 3, running: false, steps: 0 };
  let P;                                            // parameters {w, b, a, c}
  const N = 160, XS = Array.from({ length: N }, (_, i) => -1 + 2 * i / (N - 1));
  const canvas = h('canvas', { width: W, height: H });
  const g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.6' });

  function init() {
    const r = rng(state.seed * 97 + state.n);
    P = { w: [], b: [], a: [], c: 0 };
    for (let k = 0; k < state.n; k++) {
      const w = (r() < 0.5 ? -1 : 1) * (1 + 2 * r()), joint = -0.95 + 1.9 * r();
      P.w.push(w); P.b.push(-w * joint); P.a.push((r() - 0.5) * 0.6);
    }
    P.c = 0; state.steps = 0;
    adam = { m: null, v: null, t: 0 };
  }
  const relu = z => (z > 0 ? z : 0);
  const unit = (k, x) => P.a[k] * relu(P.w[k] * x + P.b[k]);
  const f = x => P.c + P.w.reduce((s, _, k) => s + unit(k, x), 0);
  const loss = () => XS.reduce((s, x) => s + (f(x) - TARGETS[state.target](x)) ** 2, 0) / N;

  // gradient of the mean squared error, done by hand (the chain rule, as in the lecture)
  let adam;
  function step(lr = 0.02) {
    const n = state.n, gw = new Array(n).fill(0), gb = new Array(n).fill(0), ga = new Array(n).fill(0);
    let gc = 0;
    for (const x of XS) {
      const e = 2 * (f(x) - TARGETS[state.target](x)) / N;
      gc += e;
      for (let k = 0; k < n; k++) {
        const z = P.w[k] * x + P.b[k];
        if (z <= 0) continue;
        ga[k] += e * z; gw[k] += e * P.a[k] * x; gb[k] += e * P.a[k];
      }
    }
    const grad = [...gw, ...gb, ...ga, gc], theta = [...P.w, ...P.b, ...P.a, P.c];
    if (!adam.m) { adam.m = grad.map(() => 0); adam.v = grad.map(() => 0); }
    adam.t++;
    const b1 = 0.9, b2 = 0.999;
    for (let i = 0; i < theta.length; i++) {
      adam.m[i] = b1 * adam.m[i] + (1 - b1) * grad[i];
      adam.v[i] = b2 * adam.v[i] + (1 - b2) * grad[i] ** 2;
      theta[i] -= lr * (adam.m[i] / (1 - b1 ** adam.t)) / (Math.sqrt(adam.v[i] / (1 - b2 ** adam.t)) + 1e-8);
    }
    P.w = theta.slice(0, n); P.b = theta.slice(n, 2 * n); P.a = theta.slice(2 * n, 3 * n); P.c = theta[3 * n];
    state.steps++;
  }

  function draw() {
    const pad = 30, sx = x => pad + (x + 1) / 2 * (W - 2 * pad), sy = y => H / 2 - y * (H / 2 - pad) / 1.7;
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#2d313a'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, sy(0)); g.lineTo(W - pad, sy(0)); g.stroke();
    const curve = (fn, color, width, dash = []) => {
      g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash); g.beginPath();
      XS.forEach((x, i) => (i ? g.lineTo(sx(x), sy(fn(x))) : g.moveTo(sx(x), sy(fn(x))))); g.stroke(); g.setLineDash([]);
    };
    for (let k = 0; k < state.n; k++) curve(x => unit(k, x), 'rgba(178,141,255,0.35)', 1.2);
    curve(TARGETS[state.target], C.dim, 2, [6, 5]);
    curve(f, C.accent, 3);
    // joints
    for (let k = 0; k < state.n; k++) {
      const j = -P.b[k] / P.w[k];
      if (j < -1 || j > 1) continue;
      g.fillStyle = C.purple; g.beginPath(); g.arc(sx(j), sy(f(j)), 4.5, 0, 7); g.fill();
    }
    g.font = '600 15px Inter, Arial'; g.fillStyle = C.blue; g.textAlign = 'right';
    g.fillText('input x →', W - pad, sy(0) - 8); g.textAlign = 'left';
    g.fillStyle = C.accent; g.fillText('output f(x)', pad, 20);
    g.font = '13px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText('dashed: target · yellow: network · purple: single units and joints', pad, H - 8);
    readout.innerHTML =
      `<div>hidden units: <b>${state.n}</b></div><div>parameters: <b>${3 * state.n + 1}</b></div>` +
      `<div>linear pieces: <b>≤ ${state.n + 1}</b></div>` +
      `<div>loss (MSE): <b>${fmt(loss(), 4)}</b></div><div class="dim">training steps: ${state.steps}</div>`;
  }

  let raf;
  function run() {
    state.running = !state.running; fitBtn.textContent = state.running ? '⏸ pause' : '▶ fit';
    const tick = () => {
      if (!state.running || !canvas.isConnected) return;
      for (let i = 0; i < 8; i++) step();
      draw();
      if (state.steps < 4000) raf = requestAnimationFrame(tick); else { state.running = false; fitBtn.textContent = '▶ fit'; }
    };
    if (state.running) raf = requestAnimationFrame(tick);
  }
  const fitBtn = h('button', { onclick: run }, '▶ fit');
  const slider = h('input', { type: 'range', min: 1, max: 30, step: 1, value: state.n });
  slider.addEventListener('input', () => { state.n = +slider.value; init(); draw(); });
  const tsel = Object.keys(TARGETS).map(t => h('button', { onclick: () => { state.target = t; tsel.forEach(b => b.classList.toggle('active', b.textContent === t)); state.steps = 0; adam = { m: null, v: null, t: 0 }; draw(); } }, t));
  tsel.forEach(b => b.classList.toggle('active', b.textContent === state.target));

  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:13em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'flex-direction:column; align-items:flex-start; gap:0.35em' },
        h('div', {}, h('label', {}, 'units '), slider),
        h('div', { style: 'display:flex; gap:0.3em' }, fitBtn, h('button', { onclick: () => { state.seed++; init(); draw(); } }, 'reset')),
        h('div', { style: 'display:flex; gap:0.3em' }, tsel)))));
  init();
  if (isPrint() || cfg.prefit) for (let i = 0; i < 3000; i++) step();
  draw();
}
