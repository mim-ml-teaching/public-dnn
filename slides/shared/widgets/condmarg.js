// Conditional vs marginal velocity fields, computed exactly (no network), in the setting of the Cambridge MLG post
// "An introduction to Flow Matching": two source clusters on the left, two data clusters on the right, linear
// conditional paths x_t = (1 − t) x0 + t x1, so that u_t(x | x1) = (x1 − x) / (1 − t) and the posterior over the data
// points is in closed form.
// mode 'fan':   particles follow the marginal field; at every moment the thin arrows are the conditional velocities
//               u_t(x | x1) towards the data points, with the posterior p(x1 | x_t = x) as opacity, and the bold arrow
//               is their average, u_t(x).
// mode 'paths': left, the straight conditional paths of random pairs (x0, x1): they cross; right, the marginal paths of
//               the same starting points: they do not. The clouds agree at every t: same p_t, different motions.
// mode 'loss':  the two Monte Carlo estimators of the loss at one time t. Left: samples x_t ~ p_t with the exact target
//               u_t(x_t), each an inner average over x1 ~ posterior (the fan is shown at one of them). Right: pairs (x0, x1)
//               sampled all over the place, one conditional arrow at each x_t; their local averages near the same points
//               converge to the targets on the left as the arrows accumulate.
// config: { mode: 'fan' | 'paths' | 'loss', n: 160 (data points), particles: 2 (fan) | 140 (paths), tflow: 0.5 (loss),
//           width, height, seed: 3, loop: true, t: 0, print_t: 0.55 }
import { C, h, isPrint, loopPlayer, arrow } from './util.js';
import { rng, gauss } from './flowlib.js';

const SRC = [[-3, 1.3], [-3, -1.3]], DAT = [[3, 1.3], [3, -1.3]], SD = 0.38, SMIN = 0.04;   // cluster centres and spread
const XR0 = 4.4, YR0 = 2.3, TMAX = 0.97, DT = 0.005, LAM = 0.2;   // smallest view half-sizes, end of the animation, Euler step, arrow scale
const BLUE = '#5ab0ff', RED = '#ff6b6b';
const NMAX = 320, NPROBE = 6, KW = 0.32;   // loss mode: arrows sampled, probe points, kernel width of the local average

export function mount(el, cfg) {
  const MODE = cfg.mode || 'fan', W = cfg.width || 640, H = cfg.height || 400, N = cfg.n || 160, seed0 = cfg.seed || 3;
  const st = { t: cfg.t ?? 0, seed: seed0, tflow: cfg.tflow ?? 0.5 };
  const PANEL = MODE === 'fan' ? 1 : 2, PW = PANEL === 2 ? (W - 16) / 2 : W;
  const YR = Math.max(YR0, XR0 * H / PW), XR = YR * PW / H;   // isotropic view that holds both cluster rows whatever the panel's aspect
  const canvas = h('canvas', { width: W, height: H, style: `width:${W}px; height:${H}px` }), g = canvas.getContext('2d');
  const sx = (x, p = 0) => p * (PW + 16) + (x + XR) / (2 * XR) * PW, sy = y => H - (y + YR) / (2 * YR) * H;
  // the clouds
  let data, src, parts, trajM, trajC, pairs, lossPairs = [], probes = [], probeU = [], heat = [];   // heat: the exact density p_t on a grid, per time step (paths mode)
  const HC = 6, HNT = 40, hnx = Math.ceil(PW / HC), hny = Math.ceil(H / HC);
  const draw1 = (r, centres) => { const c = centres[Math.floor(r() * 2)]; return [c[0] + SD * gauss(r), c[1] + SD * gauss(r)]; };
  // posterior weights of the data points at (t, x), and the marginal velocity
  const post = (t, x) => {
    const v = (1 - t) * (1 - t) * SD * SD + SMIN * SMIN, w = new Float64Array(N); let mx = -Infinity;
    for (let i = 0; i < N; i++) { let s = 0; for (const mu of SRC) { const dx = x[0] - ((1 - t) * mu[0] + t * data[i][0]), dy = x[1] - ((1 - t) * mu[1] + t * data[i][1]); s += Math.exp(-(dx * dx + dy * dy) / (2 * v)); } w[i] = s; mx = Math.max(mx, s); }
    let tot = 0; for (let i = 0; i < N; i++) tot += w[i];
    for (let i = 0; i < N; i++) w[i] /= tot;
    return w;
  };
  const condU = (t, x, x1) => [(x1[0] - x[0]) / Math.max(1 - t, 0.03), (x1[1] - x[1]) / Math.max(1 - t, 0.03)];
  const margU = (t, x, w = post(t, x)) => { let ux = 0, uy = 0; for (let i = 0; i < N; i++) { const u = condU(t, x, data[i]); ux += w[i] * u[0]; uy += w[i] * u[1]; } return [ux, uy]; };
  const integrate = x0 => { const path = [x0.slice()]; let x = x0.slice(); for (let t = 0; t < TMAX - 1e-9; t += DT) { const u = margU(t, x); x = [x[0] + DT * u[0], x[1] + DT * u[1]]; path.push(x); } return path; };
  function setup() {
    const r = rng(st.seed);
    data = Array.from({ length: N }, () => draw1(r, DAT)); src = Array.from({ length: 120 }, () => draw1(r, SRC));
    if (MODE === 'fan') { const k = cfg.particles || 2; parts = Array.from({ length: k }, (_, i) => { const c = SRC[i % 2]; return [c[0] + SD * gauss(r), c[1] + SD * gauss(r)]; }); trajM = parts.map(integrate); }
    else if (MODE === 'paths') { const k = cfg.particles || 140; parts = Array.from({ length: k }, () => draw1(r, SRC)); pairs = parts.map(() => data[Math.floor(r() * N)]); trajM = parts.map(integrate); trajC = parts.map((x0, j) => [x0, pairs[j]]); heatmaps(); }
    else setupLoss(r);
  }
  function setupLoss(r) {   // the conditional estimator's pairs, and probe points x_t ~ p_t (the first few pairs' x_t) with their exact targets
    const tt = st.tflow;
    lossPairs = Array.from({ length: NMAX }, () => { const x0 = draw1(r, SRC), x1 = data[Math.floor(r() * N)]; return { x0, x1, xt: [x0[0] + tt * (x1[0] - x0[0]), x0[1] + tt * (x1[1] - x0[1])], u: [x1[0] - x0[0], x1[1] - x0[1]] }; });
    probes = lossPairs.slice(0, NPROBE).map(q => q.xt);
    probeU = probes.map(x => margU(tt, x));
  }
  const at = (path, t) => path[Math.min(path.length - 1, Math.round(t / DT))];
  // the exact p_t of the construction: the mixture over data points and source components of N((1 − t) μ + t x1, (1 − t)² SD²)
  function heatmaps() {
    heat = [];
    for (let k = 0; k <= HNT; k++) {
      const t = k / HNT * TMAX, v = (1 - t) * (1 - t) * SD * SD + SMIN * SMIN, grid = new Float32Array(hnx * hny); let mx = 0;
      for (let j = 0; j < hny; j++) for (let i = 0; i < hnx; i++) {
        const x = -XR + (i + 0.5) * HC / PW * 2 * XR, y = YR - (j + 0.5) * HC / H * 2 * YR; let d = 0;
        for (const q of data) for (const mu of SRC) { const dx = x - ((1 - t) * mu[0] + t * q[0]), dy = y - ((1 - t) * mu[1] + t * q[1]); d += Math.exp(-(dx * dx + dy * dy) / (2 * v)); }
        grid[j * hnx + i] = d; mx = Math.max(mx, d);
      }
      for (let i = 0; i < grid.length; i++) grid[i] /= mx;
      heat.push(grid);
    }
  }
  const colEnd = (y, a = 1) => (y > 0 ? `rgba(90,176,255,${a})` : `rgba(255,107,107,${a})`);

  function drawFan() {
    const t = Math.min(st.t, TMAX);
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    g.fillStyle = 'rgba(154,161,174,0.35)'; for (const p of src) { g.beginPath(); g.arc(sx(p[0]), sy(p[1]), 2, 0, 7); g.fill(); }
    const ws = parts.map((_, j) => post(t, at(trajM[j], t)));
    const wmax = ws.map(w => Math.max(...w));
    data.forEach((p, i) => { const a = Math.max(...ws.map((w, j) => w[i] / wmax[j])); g.fillStyle = `rgba(231,233,238,${(0.12 + 0.7 * a).toFixed(3)})`; g.beginPath(); g.arc(sx(p[0]), sy(p[1]), 2.6, 0, 7); g.fill(); });
    parts.forEach((_, j) => {
      const col = j % 2 === 0 ? BLUE : RED, rgb = j % 2 === 0 ? '90,176,255' : '255,107,107', x = at(trajM[j], t), w = ws[j];
      // the trajectory so far
      g.strokeStyle = `rgba(${rgb},0.9)`; g.lineWidth = 1.5; g.beginPath(); trajM[j].slice(0, Math.round(t / DT) + 1).forEach((p, k) => (k ? g.lineTo(sx(p[0]), sy(p[1])) : g.moveTo(sx(p[0]), sy(p[1])))); g.stroke();
      // the conditional velocities towards every data point, opacity = posterior weight
      for (let i = 0; i < N; i++) {
        const a = w[i] / wmax[j]; if (a < 0.02) continue;
        const u = condU(t, x, data[i]), L = Math.hypot(...u), d = Math.hypot(data[i][0] - x[0], data[i][1] - x[1]), len = Math.min(LAM * L, 0.92 * d), f = len / (L + 1e-9);
        g.strokeStyle = `rgba(${rgb},${(0.85 * a).toFixed(3)})`; g.lineWidth = 1; g.beginPath(); g.moveTo(sx(x[0]), sy(x[1])); g.lineTo(sx(x[0] + u[0] * f), sy(x[1] + u[1] * f)); g.stroke();
      }
      // their average: the marginal velocity
      const u = margU(t, x, w), L = Math.hypot(...u), len = Math.min(LAM * L, 2.2), f = len / (L + 1e-9);
      arrow(g, sx(x[0]), sy(x[1]), sx(x[0] + u[0] * f), sy(x[1] + u[1] * f), col, 3);
      g.fillStyle = col; g.beginPath(); g.arc(sx(x[0]), sy(x[1]), 4.5, 0, 7); g.fill();
    });
    g.font = 'bold 15px Inter, Arial'; g.textAlign = 'left'; g.fillStyle = C.fg; g.fillText(`u_t(x)  vs  u_t(x | x₁)  at  t = ${t.toFixed(2)}`, 12, 24);
    g.font = '12px Inter, Arial'; g.fillStyle = C.dim; g.fillText('thin arrows: u_t(x | x₁) for every data point x₁, opacity = posterior p(x₁ | x_t = x)', 12, 44); g.fillText('bold arrow: their average, u_t(x) · the particle follows u_t', 12, 60);
  }
  function drawPaths() {
    const t = Math.min(st.t, TMAX), k = Math.round(t / DT);
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    g.fillStyle = C.bg3; g.fillRect(PW, 0, 16, H);
    const grid = heat[Math.round(t / TMAX * HNT)];
    for (let p = 0; p < 2; p++) {
      if (grid) for (let j = 0; j < hny; j++) for (let i = 0; i < hnx; i++) { const a = grid[j * hnx + i]; if (a > 0.03) { g.fillStyle = `rgba(200,210,230,${(Math.pow(a, 0.7) * 0.4).toFixed(3)})`; g.fillRect(p * (PW + 16) + i * HC, j * HC, HC, HC); } }
      g.fillStyle = 'rgba(154,161,174,0.25)'; for (const q of data) { g.beginPath(); g.arc(sx(q[0], p), sy(q[1]), 2, 0, 7); g.fill(); }
      parts.forEach((x0, j) => {
        const pos = p === 0 ? [x0[0] + t * (pairs[j][0] - x0[0]), x0[1] + t * (pairs[j][1] - x0[1])] : at(trajM[j], t);
        const yEnd = p === 0 ? pairs[j][1] : trajM[j][trajM[j].length - 1][1];
        g.strokeStyle = colEnd(yEnd, 0.22); g.lineWidth = 1; g.beginPath();
        if (p === 0) { g.moveTo(sx(x0[0], p), sy(x0[1])); g.lineTo(sx(pos[0], p), sy(pos[1])); }
        else trajM[j].slice(0, k + 1).forEach((q, i) => (i ? g.lineTo(sx(q[0], p), sy(q[1])) : g.moveTo(sx(q[0], p), sy(q[1]))));
        g.stroke();
        g.fillStyle = colEnd(yEnd, 0.95); g.beginPath(); g.arc(sx(pos[0], p), sy(pos[1]), 2.8, 0, 7); g.fill();
      });
    }
    g.font = 'bold 14px Inter, Arial'; g.textAlign = 'left'; g.fillStyle = C.fg;
    g.fillText('conditional paths: straight lines x₀ → x₁, random pairs', 10, 22); g.fillText('marginal paths: the same x₀, following u_t', PW + 26, 22);
    g.font = '12px Inter, Arial'; g.fillStyle = C.dim; g.fillText('they cross', 10, 40); g.fillText('they never cross', PW + 26, 40);
    g.textAlign = 'center'; g.fillStyle = C.accent; g.font = 'bold 13px Inter, Arial'; g.fillText(`t = ${t.toFixed(2)} · shaded: the exact density p_t, the same in both panels; the dots are samples of it`, W / 2, H - 10);
  }
  function drawLoss() {
    const tt = st.tflow, n = Math.round(Math.min(1, st.t / TMAX) * NMAX), hi = probes[0], wHi = post(tt, hi), wmax = Math.max(...wHi);
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H); g.fillStyle = C.bg3; g.fillRect(PW, 0, 16, H);
    for (let p = 0; p < 2; p++) { g.fillStyle = 'rgba(154,161,174,0.22)'; for (const q of src) { g.beginPath(); g.arc(sx(q[0], p), sy(q[1]), 1.8, 0, 7); g.fill(); } for (const q of data) { g.beginPath(); g.arc(sx(q[0], p), sy(q[1]), 1.8, 0, 7); g.fill(); } }
    const arrowU = (p, x, u, col, lw) => { const L = Math.hypot(...u), len = Math.min(LAM * L, 2.2), f = len / (L + 1e-9); arrow(g, sx(x[0], p), sy(x[1]), sx(x[0] + u[0] * f, p), sy(x[1] + u[1] * f), col, lw); };
    // left: the targets of the unconditional loss, exact posterior averages; the inner fan at the first probe
    for (let i = 0; i < N; i++) { const a = wHi[i] / wmax; if (a < 0.02) continue; const u = condU(tt, hi, data[i]), L = Math.hypot(...u), d = Math.hypot(data[i][0] - hi[0], data[i][1] - hi[1]), f = Math.min(LAM * L, 0.92 * d) / (L + 1e-9);
      g.strokeStyle = `rgba(90,176,255,${(0.8 * a).toFixed(3)})`; g.lineWidth = 1; g.beginPath(); g.moveTo(sx(hi[0], 0), sy(hi[1])); g.lineTo(sx(hi[0] + u[0] * f, 0), sy(hi[1] + u[1] * f)); g.stroke(); }
    probes.forEach((x, k) => { arrowU(0, x, probeU[k], C.accent, 2.5); g.fillStyle = C.fg; g.beginPath(); g.arc(sx(x[0], 0), sy(x[1]), 3.5, 0, 7); g.fill(); });
    // right: n conditional arrows all over the place, and their local averages near the same probe points
    for (let j = 0; j < n; j++) { const q = lossPairs[j], L = Math.hypot(...q.u), f = Math.min(LAM * L, 2.2) / (L + 1e-9); g.strokeStyle = colEnd(q.x1[1], 0.35); g.lineWidth = 1; g.beginPath(); g.moveTo(sx(q.xt[0], 1), sy(q.xt[1])); g.lineTo(sx(q.xt[0] + q.u[0] * f, 1), sy(q.xt[1] + q.u[1] * f)); g.stroke(); }
    probes.forEach(x => {
      let sw = 0, ux = 0, uy = 0; for (let j = 0; j < n; j++) { const q = lossPairs[j], d2 = (q.xt[0] - x[0]) ** 2 + (q.xt[1] - x[1]) ** 2, w = Math.exp(-d2 / (2 * KW * KW)); sw += w; ux += w * q.u[0]; uy += w * q.u[1]; }
      if (sw > 0.8) arrowU(1, x, [ux / sw, uy / sw], C.accent, 2.5);
      g.fillStyle = C.fg; g.beginPath(); g.arc(sx(x[0], 1), sy(x[1]), 3.5, 0, 7); g.fill();
    });
    g.font = 'bold 14px Inter, Arial'; g.textAlign = 'left'; g.fillStyle = C.fg;
    g.fillText(`ℒ_FM at t = ${tt.toFixed(2)}: sample x_t ~ p_t, target u_t(x_t)`, 10, 22); g.fillText(`ℒ_CFM at t = ${tt.toFixed(2)}: sample pairs (x₀, x₁), one arrow each`, PW + 26, 22);
    g.font = '12px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText('every target is an inner average over x₁ ~ posterior (fan): needs p(x₁ | x_t)', 10, 40); g.fillText(`n = ${n} arrows · bold: their average near each point → u_t(x_t)`, PW + 26, 40);
  }
  const draw = () => (MODE === 'fan' ? drawFan() : MODE === 'paths' ? drawPaths() : drawLoss());
  const loop = loopPlayer({ dur: 7000, hold: 1800, hold0: 500, alive: () => canvas.isConnected, step: t => { st.t = t * TMAX; slider.value = st.t; draw(); } });
  const setBtn = () => { playBtn.textContent = loop.playing ? '⏸' : '▶'; };
  const playBtn = h('button', { style: 'min-width:2.4em', onclick: () => { if (loop.playing) loop.stop(); else loop.start(st.t / TMAX); setBtn(); } }, '▶');
  const slider = h('input', { type: 'range', min: 0, max: TMAX, step: 0.005, value: st.t, style: 'width:10em', oninput: e => { loop.stop(); setBtn(); st.t = +e.target.value; draw(); } });
  const newBtn = h('button', { onclick: () => { st.seed++; setup(); draw(); } }, MODE === 'fan' ? '↺ new x₀' : '↺ new samples');
  const tfSlider = h('input', { type: 'range', min: 0.05, max: 0.95, step: 0.05, value: st.tflow, style: 'width:7em', oninput: e => { st.tflow = +e.target.value; setup(); draw(); } });
  el.classList.add('widget');
  const ctl = MODE === 'loss' ? [h('label', {}, 'samples'), slider, playBtn, h('label', { style: 'margin-left:0.6em' }, 'time t'), tfSlider, newBtn] : [h('label', {}, 'time t'), slider, playBtn, newBtn];
  el.append(canvas, h('div', { class: 'wctl interactive-only', style: 'gap:0.5em; margin-top:0.3em; justify-content:center' }, ...ctl));
  setup();
  const reset = () => { loop.stop(); st.t = cfg.t ?? 0; slider.value = st.t; if (cfg.loop !== false && !isPrint()) loop.start(0); setBtn(); draw(); };
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!canvas.isConnected) return; if (ev.currentSlide?.contains(el)) reset(); else { loop.stop(); setBtn(); } });
  if (isPrint()) { st.t = cfg.print_t ?? 0.55; slider.value = st.t; draw(); return; }
  draw();
  if (el.closest('section')?.classList.contains('present')) reset();
}
