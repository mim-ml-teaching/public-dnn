// Euler's method on the rotation field u(x, y) = (−y, x): the exact flow carries every point along a circle;
// Euler steps follow the tangent and drift outwards, the more the larger the step. A small patch of points is
// carried along by the exact flow to show that a flow moves whole regions, not just one point.
// config: { n: 32, width: 420, height: 420 }
import { C, h, fmt, isPrint } from './util.js';

export function mount(el, cfg) {
  const W = cfg.width || 420, H = cfg.height || 420, R = 3.2, T = 2 * Math.PI, r0 = 1.6;
  const st = { n: cfg.n ?? 32, k: null, playing: false };
  const sx = x => (x + R) / (2 * R) * W, sy = y => H - (y + R) / (2 * R) * H;
  const canvas = h('canvas', { width: W, height: H }), g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.5' });
  const eulerPath = n => { const p = [[r0, 0]]; let [x, y] = p[0]; const hh = T / n; for (let i = 0; i < n; i++) { [x, y] = [x - hh * y, y + hh * x]; p.push([x, y]); } return p; };
  const patch = []; for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) patch.push([r0 + 0.12 * i, 0.12 * j]);
  const rot = ([x, y], a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];

  function draw() {
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    // the field
    const m = 11, step = 2 * R / m;
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
      const x = -R + (i + 0.5) * step, y = -R + (j + 0.5) * step, L = Math.hypot(x, y) || 1, s = Math.min(0.45, 0.12 * L) / L;
      g.strokeStyle = '#4a505c'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(sx(x), sy(y)); g.lineTo(sx(x - y * s), sy(y + x * s)); g.stroke();
      g.fillStyle = '#4a505c'; g.beginPath(); g.arc(sx(x - y * s), sy(y + x * s), 1.8, 0, 7); g.fill();
    }
    // exact flow: the circle, and the patch at four times
    g.strokeStyle = '#9aa1ae'; g.setLineDash([5, 5]); g.lineWidth = 1.5; g.beginPath(); g.arc(sx(0), sy(0), r0 / (2 * R) * W, 0, 7); g.stroke(); g.setLineDash([]);
    [0, 0.25, 0.5, 0.75].forEach((f, i) => { g.fillStyle = i ? 'rgba(95,211,141,0.45)' : C.green; patch.forEach(p => { const [x, y] = rot(p, f * T); g.beginPath(); g.arc(sx(x), sy(y), 2.5, 0, 7); g.fill(); }); });
    // Euler
    const path = eulerPath(st.n), k = st.k ?? st.n, shown = path.slice(0, k + 1);
    g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath(); shown.forEach(([x, y], i) => (i ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y)))); g.stroke();
    g.fillStyle = C.accent; shown.forEach(([x, y]) => { g.beginPath(); g.arc(sx(x), sy(y), st.n > 32 ? 2 : 4, 0, 7); g.fill(); });
    const last = shown.at(-1); g.fillStyle = C.fg; g.beginPath(); g.arc(sx(last[0]), sy(last[1]), 5, 0, 7); g.fill();
    g.font = '13px Inter, Arial'; g.fillStyle = C.dim; g.fillText('u(x, y) = (−y, x)', 10, 18);
    readout.innerHTML = `<div><b>n = ${st.n}</b> Euler steps over one full turn: h = 2π/${st.n} = ${fmt(T / st.n, 2)}</div>` +
      `<div><span style="color:${C.green}">●</span> a patch carried by the exact flow (every point circles at the same speed) · dashed: the exact orbit</div>` +
      `<div><span style="color:${C.accent}">●</span> Euler path · distance from the origin: ${fmt(r0, 1)} → <b>${fmt(Math.hypot(...path.at(-1)), 2)}</b> after one turn (exact: ${fmt(r0, 1)})</div>`;
  }
  function play() {
    st.playing = !st.playing; playBtn.textContent = st.playing ? '⏸ pause' : '▶ step by step';
    if (st.k === null || st.k >= st.n) st.k = 0;
    const tick = () => { if (!st.playing || !canvas.isConnected) return; st.k++; draw(); if (st.k < st.n) setTimeout(tick, Math.max(60, 900 / Math.sqrt(st.n))); else { st.playing = false; st.k = null; playBtn.textContent = '▶ step by step'; } };
    if (st.playing) tick();
  }
  const playBtn = h('button', { onclick: play }, '▶ step by step');
  const btns = [8, 16, 32, 64, 256].map(n => h('button', { class: n === st.n ? 'active' : '', onclick: e => { st.n = n; st.k = null; st.playing = false; playBtn.textContent = '▶ step by step'; btns.forEach(b => b.classList.toggle('active', b === e.target)); draw(); } }, String(n)));
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:17em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'display:flex; flex-direction:column; gap:0.4em; align-items:flex-start' },
        h('div', { style: 'display:flex; gap:0.4em; align-items:center; flex-wrap:wrap' }, h('label', {}, 'steps'), ...btns), playBtn))));
  draw();
}
