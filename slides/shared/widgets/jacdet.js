// How a map changes volume locally. A grid on [−L, L]² is carried by φ_τ = (1 − τ)·id + τ·φ (slider / play); each
// cell is coloured by image area / original area (blue: shrinks, red: expands), which for a small cell is |det Dφ_τ|.
// One cell A is highlighted (click to move it): the two arrows are the images of its two sides — for a small cell the
// columns of the Jacobian at its corner times the side length — and the readout is the area of the parallelogram
// they span relative to the area of A. The map is separable, φ(x, y) = (f(x) + g(y), h(x) + k(y)), so the image of a
// grid cell is exactly that parallelogram: arrows, cell outline and colour always agree.
// config: { width: 560, height: 370, tau: 0, print_tau: 1, cell: [6, 2], loop: false (true: plays in a loop by itself) }
import { C, h, isPrint, arrow, loopPlayer } from './util.js';

const L = 1.5, N = 8;                         // grid on [−L, L]², N×N cells
// stretch along x (growing with x) and along y (shrinking with y), a shear of x by y that varies with y, and a small
// shear of y by x, plus a translation so that points visibly travel: |det Dφ| runs from ≈ 0.11 (top left) to ≈ 2.6 (bottom right)
const f = x => x + 0.21 * x * x, g = y => 0.4 * y + 0.1 * Math.sin(2 * y), hx = x => 0.12 * x, k = y => y - 0.21 * y * y;
const phi = (x, y) => [f(x) + g(y) - 0.2, hx(x) + k(y) + 0.35];
const RMIN = 0.1, RMAX = 3;   // range of the colour scale
const col = r => r >= 1 ? `rgba(255,107,107,${(Math.min(1, (r - 1) / (RMAX - 1)) * 0.7).toFixed(3)})` : `rgba(90,176,255,${(Math.min(1, (1 - r) / (1 - RMIN)) * 0.7).toFixed(3)})`;
const VX = -1.9, VY = 1.62;   // view box: x from VX rightwards, y from VY downwards (holds the grid and its image)

export function mount(el, cfg) {
  const W = cfg.width || 560, H = cfg.height || 370, S = 83, X0 = 10, Y0 = 20, RX = 388;   // RX: where the readout starts
  const sx = x => X0 + (x - VX) * S, sy = y => Y0 + (VY - y) * S;
  const st = { tau: cfg.tau ?? 0, cell: (cfg.cell || [6, 2]).slice() };
  const hc = 2 * L / N;
  const canvas = h('canvas', { width: W, height: H, style: 'cursor:pointer' }), g = canvas.getContext('2d');
  const map = (x, y, tau) => { const [u, v] = phi(x, y); return [x + tau * (u - x), y + tau * (v - y)]; };
  const area = P => { let a = 0; for (let i = 0; i < P.length; i++) { const [x0, y0] = P[i], [x1, y1] = P[(i + 1) % P.length]; a += x0 * y1 - x1 * y0; } return Math.abs(a) / 2; };
  const corner = (i, j) => [-L + i * hc, -L + j * hc];
  let V;   // images of the grid vertices at the current τ
  const vertices = tau => { V = []; for (let i = 0; i <= N; i++) { V.push([]); for (let j = 0; j <= N; j++) V[i].push(map(...corner(i, j), tau)); } };
  const poly = (i, j) => [V[i][j], V[i + 1][j], V[i + 1][j + 1], V[i][j + 1]];
  const path = P => { g.beginPath(); P.forEach(([x, y], k) => (k ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y)))); g.closePath(); };

  function draw() {
    const tau = st.tau;
    vertices(tau);
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const P = poly(i, j);
      path(P); g.fillStyle = col(area(P) / (hc * hc)); g.fill(); g.strokeStyle = 'rgba(154,161,174,0.55)'; g.lineWidth = 1; g.stroke();
    }
    // the highlighted cell A: its image is the parallelogram spanned by the images of its two sides (the arrows)
    const [ci, cj] = st.cell, P = poly(ci, cj), p0 = V[ci][cj], x0 = corner(ci, cj);
    if (tau > 0.02) {   // where A and its corner x were, and the way the corner travelled
      g.setLineDash([4, 3]); g.strokeStyle = 'rgba(242,177,52,0.6)'; g.lineWidth = 1.5;
      path([x0, [x0[0] + hc, x0[1]], [x0[0] + hc, x0[1] + hc], [x0[0], x0[1] + hc]]); g.stroke();
      g.strokeStyle = C.fg; g.beginPath(); g.moveTo(sx(x0[0]), sy(x0[1])); g.lineTo(sx(p0[0]), sy(p0[1])); g.stroke(); g.setLineDash([]);
      g.fillStyle = 'rgba(231,233,238,0.5)'; g.beginPath(); g.arc(sx(x0[0]), sy(x0[1]), 3, 0, 7); g.fill();
    }
    path(P); g.fillStyle = 'rgba(242,177,52,0.5)'; g.fill(); g.strokeStyle = C.accent; g.lineWidth = 2.5; g.stroke();
    const u = [V[ci + 1][cj][0] - p0[0], V[ci + 1][cj][1] - p0[1]], v = [V[ci][cj + 1][0] - p0[0], V[ci][cj + 1][1] - p0[1]];
    const det = Math.abs(u[0] * v[1] - u[1] * v[0]) / (hc * hc);
    arrow(g, sx(p0[0]), sy(p0[1]), sx(p0[0] + u[0]), sy(p0[1] + u[1]), C.red, 2.5);
    arrow(g, sx(p0[0]), sy(p0[1]), sx(p0[0] + v[0]), sy(p0[1] + v[1]), C.green, 2.5);
    const cx = P.reduce((s, p) => s + p[0], 0) / 4, top = Math.max(...P.map(p => p[1]));
    g.font = 'italic 15px Inter, Arial'; g.fillStyle = C.fg; g.textAlign = 'center'; g.fillText(tau < 0.02 ? 'A' : 'φ(A)', sx(cx), sy(top) - 6);
    // the corner x of A, carried to φ(x): the origin of the two arrows
    g.fillStyle = C.bg; g.beginPath(); g.arc(sx(p0[0]), sy(p0[1]), 6, 0, 7); g.fill(); g.fillStyle = C.fg; g.beginPath(); g.arc(sx(p0[0]), sy(p0[1]), 4, 0, 7); g.fill();
    g.textAlign = 'right'; g.fillText(tau < 0.02 ? 'x' : 'φ(x)', sx(p0[0]) - 8, sy(p0[1]) + 15);
    // readout
    g.textAlign = 'left'; g.font = '13px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText('area change at the cell A', RX, 30);
    g.font = 'bold 30px Inter, Arial'; g.fillStyle = C.accent; g.fillText(`× ${det.toFixed(2)}`, RX, 64);
    g.font = '13px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText('= area of φ(A), the', RX, 88); g.fillText('parallelogram spanned', RX, 104); g.fillText('by the two arrows,', RX, 120); g.fillText('divided by the area of A', RX, 136);
    g.fillText('colour of a cell:', RX, 186); g.fillText('its area change', RX, 202);
    const lw = 150, lx = RX, ly = 212;
    for (let i = 0; i < lw; i++) { g.fillStyle = col(RMIN + (RMAX - RMIN) * i / (lw - 1)); g.fillRect(lx + i, ly, 1.5, 12); }
    g.fillStyle = C.dim; g.font = '12px Inter, Arial'; g.textAlign = 'left'; g.fillText(`× ${RMIN}`, lx, ly + 26); g.textAlign = 'center'; g.fillText('× 1', lx + lw * ((1 - RMIN) / (RMAX - RMIN)), ly + 26); g.textAlign = 'right'; g.fillText(`× ${RMAX}`, lx + lw, ly + 26);
    g.textAlign = 'left'; g.fillText('shrinks', lx, ly + 42); g.textAlign = 'right'; g.fillText('expands', lx + lw, ly + 42);
    g.textAlign = 'left'; g.fillStyle = C.dim; g.font = '13px Inter, Arial'; g.fillText('the dot: the corner x of A', RX, 290); g.fillText('is carried to φ(x)', RX, 306);
    if (!isPrint()) { g.textAlign = 'left'; g.fillStyle = C.dim; g.fillText('click a cell to move A', RX, H - 14); }
  }
  // φ is switched on over 3 s; by default once per click, with cfg.loop in a loop by itself (hold at τ = 1, again from 0)
  const setBtn = () => { playBtn.textContent = loop.playing ? '⏸ pause' : st.tau >= 1 ? '▶ replay' : '▶ play'; };
  const loop = loopPlayer({ dur: 3000, once: !cfg.loop, onEnd: () => setBtn(), alive: () => canvas.isConnected, step: tau => { st.tau = tau; slider.value = tau; draw(); } });
  const play = () => { if (loop.playing) loop.stop(); else loop.start(st.tau >= 1 ? 0 : st.tau); setBtn(); };
  const reset = () => { loop.stop(); st.tau = cfg.tau ?? 0; slider.value = st.tau; st.cell = (cfg.cell || [6, 2]).slice(); if (cfg.loop && !isPrint()) loop.start(st.tau); setBtn(); };
  canvas.addEventListener('click', ev => {
    const r = canvas.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * H / r.height;
    const x = (px - X0) / S + VX, y = VY - (py - Y0) / S;
    let best = null, bd = 1e9;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const P = poly(i, j), cx = P.reduce((s, p) => s + p[0], 0) / 4, cy = P.reduce((s, p) => s + p[1], 0) / 4, d = (cx - x) ** 2 + (cy - y) ** 2;
      if (d < bd) { bd = d; best = [i, j]; }
    }
    if (best && bd < 0.5) { st.cell = best; draw(); }
  });
  const playBtn = h('button', { onclick: play, style: 'min-width:6.5em' }, '▶ play');
  const slider = h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: st.tau, style: 'width:11em', oninput: e => { loop.stop(); st.tau = +e.target.value; setBtn(); draw(); } });
  el.classList.add('widget');
  el.append(canvas, h('div', { class: 'wctl interactive-only', style: 'gap:0.6em; margin-top:0.3em' }, h('label', {}, 'switch φ on: τ'), slider, playBtn));
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!canvas.isConnected) return; if (ev.currentSlide?.contains(el)) { reset(); draw(); } else if (loop.playing) { loop.stop(); setBtn(); } });
  if (isPrint() && cfg.print_tau !== undefined) { st.tau = cfg.print_tau; slider.value = st.tau; }
  draw();
  if (!isPrint() && el.closest('section')?.classList.contains('present')) reset();   // mounted on the current slide: start now (loop mode)
}
