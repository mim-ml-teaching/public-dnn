// Fitting a mixture of Gaussians to 2-D data, the general framework as a loop: 1 · samples from the unknown
// distribution appear, 2 · the model: K standard Gaussians (unit covariance) at random data points, 3 · the fit: EM
// steps adapt weights, means and covariances (animated), the mixture density shown as a heat map. Loops by itself with a
// new start each time; the buttons take over (manual mode) and "loop" hands back. The readout is the loss of the
// following slides, −(1/N) Σ log p_θ(x_i).
// config: { data: 'moons' | 'spiral' | 'rings', n: 300, k: 6, width: 1140, height: 400, ry: 2.15 (half-height of the view in data units), seed: 7, loop: true,
//           static: false (true: fitted once at mount, no animation, no controls: a picture of the fit) }
import { C, h, isPrint } from './util.js';
import { rng, TARGETS, sample } from './flowlib.js';

const PAL = ['#f2b134', '#5ab0ff', '#5fd38d', '#ff6b6b', '#b28dff', '#ffd34d', '#4dd0e1', '#ff8a65', '#aed581', '#f48fb1', '#9aa1ae', '#ce93d8'];
const CELL = 4, STEP_MS = 350, MAX_IT = 80, FIT_MAX_MS = 14000;
const T_REVEAL = 1500, T_HOLD1 = 900, T_ADD = 550, T_HOLD2 = 1200, T_HOLD3 = 2500;   // the loop's timing (T_ADD: per Gaussian added in step 2)

export function mount(el, cfg) {
  const W = cfg.width || 1140, H = cfg.height || 400, N = cfg.n || 300, RY = cfg.ry || 2.15, RX = RY * W / H, STATIC = cfg.static === true;   // isotropic view [−RX, RX] × [−RY, RY]
  const st = { k: cfg.k || 6, stage: 0, iter: 0, playing: false, auto: false, seed: cfg.seed || 7, reveal: 1, shown: 0, done: false };   // shown: Gaussians placed so far (step 2)
  const canvas = h('canvas', { width: W, height: H, style: `width:${W}px; height:${H}px` }), g = canvas.getContext('2d');
  const sx = x => (x + RX) / (2 * RX) * W, sy = y => H - (y + RY) / (2 * RY) * H;
  const data = sample(TARGETS[cfg.data || 'moons'], N, rng(cfg.seed || 7));
  let comps = [], from = null, target = null, t0 = 0;   // comps: drawn parameters; from → target: the EM step being animated
  const init = () => { const r = rng(st.seed + 100 * st.k); comps = Array.from({ length: st.k }, () => { const p = data[Math.floor(r() * N)]; return { w: 1 / st.k, m: [p[0], p[1]], S: [1, 0, 1] }; }); from = target = null; st.iter = 0; st.shown = st.k; st.done = false; };
  // the mixture: S = [a, b, d] is the covariance [[a, b], [b, d]]
  const gauss = (x, c) => { const [a, b, d] = c.S, det = a * d - b * b, dx = x[0] - c.m[0], dy = x[1] - c.m[1]; return Math.exp(-(d * dx * dx - 2 * b * dx * dy + a * dy * dy) / (2 * det)) / (2 * Math.PI * Math.sqrt(det)); };
  const mix = (x, cs) => cs.reduce((s, c) => s + c.w * gauss(x, c), 0);
  const loss = cs => -data.reduce((s, x) => s + Math.log(mix(x, cs) + 1e-300), 0) / N;
  function emStep(cs) {
    const resp = data.map(x => { const r = cs.map(c => c.w * gauss(x, c)), s = r.reduce((a, b) => a + b, 0) + 1e-300; return r.map(v => v / s); });
    return cs.map((c, k) => {
      let nk = 0, mx = 0, my = 0; data.forEach((x, i) => { nk += resp[i][k]; mx += resp[i][k] * x[0]; my += resp[i][k] * x[1]; });
      nk = Math.max(nk, 1e-6); mx /= nk; my /= nk;
      let a = 0, b = 0, d = 0; data.forEach((x, i) => { const dx = x[0] - mx, dy = x[1] - my; a += resp[i][k] * dx * dx; b += resp[i][k] * dx * dy; d += resp[i][k] * dy * dy; });
      return { w: nk / N, m: [mx, my], S: [a / nk + 2e-3, b / nk, d / nk + 2e-3] };
    });
  }
  const lerp = (A, B, u) => A.map((c, k) => ({ w: c.w + (B[k].w - c.w) * u, m: [c.m[0] + (B[k].m[0] - c.m[0]) * u, c.m[1] + (B[k].m[1] - c.m[1]) * u], S: c.S.map((v, i) => v + (B[k].S[i] - v) * u) }));

  function draw() {
    const vis = st.stage === 2 ? comps.slice(0, st.shown) : comps;   // in step 2 the Gaussians come one by one
    const visN = vis.map(c => ({ ...c, w: c.w * st.k / Math.max(1, vis.length) }));   // weights renormalised, for the loss
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    if (st.stage >= 2 && vis.length) {   // the mixture density as a heat map
      const nx = Math.ceil(W / CELL), ny = Math.ceil(H / CELL), vals = new Float32Array(nx * ny); let vmax = 1e-9;
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const v = mix([-RX + (i + 0.5) * CELL / W * 2 * RX, RY - (j + 0.5) * CELL / H * 2 * RY], vis); vals[j * nx + i] = v; vmax = Math.max(vmax, v); }
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const a = Math.min(0.85, Math.pow(vals[j * nx + i] / vmax, 0.6)); if (a > 0.02) { g.fillStyle = `rgba(242,177,52,${(a * 0.8).toFixed(3)})`; g.fillRect(i * CELL, j * CELL, CELL, CELL); } }
    }
    if (st.stage >= 1) { g.fillStyle = C.blue; const n = Math.round(st.reveal * N); for (let i = 0; i < n; i++) { g.beginPath(); g.arc(sx(data[i][0]), sy(data[i][1]), 2.6, 0, 7); g.fill(); } }
    if (st.stage >= 2) vis.forEach((c, k) => {   // 1σ (filled) and 2σ (outline) ellipses of each component
      const [a, b, d] = c.S, tr = (a + d) / 2, dd = Math.sqrt(Math.max(0, tr * tr - (a * d - b * b))), l1 = tr + dd, l2 = Math.max(1e-6, tr - dd), ang = Math.atan2(l1 - a, b || 1e-9);
      const rx = Math.sqrt(l1) * W / (2 * RX), ry = Math.sqrt(l2) * H / (2 * RY);
      for (const [s, lw, fill] of [[1, 1.5, true], [2, 1.2, false]]) {
        g.beginPath(); g.ellipse(sx(c.m[0]), sy(c.m[1]), s * rx, s * ry, -ang, 0, 2 * Math.PI);
        if (fill) { g.fillStyle = PAL[k % PAL.length] + '33'; g.fill(); } g.strokeStyle = PAL[k % PAL.length]; g.lineWidth = lw; g.stroke();
      }
      g.fillStyle = PAL[k % PAL.length]; g.beginPath(); g.arc(sx(c.m[0]), sy(c.m[1]), 3.5, 0, 7); g.fill();
    });
    // the step, spoken by the animation
    g.font = 'bold 20px Inter, Arial'; g.textAlign = 'left'; g.fillStyle = C.fg;
    g.fillText(STATIC ? `K = ${st.k} Gaussians, fitted` : ['', '1 · Collect samples from the unknown distribution', '2 · Initialise a Gaussian mixture', '3 · Fit'][st.stage], 16, 30);
    g.font = '14px Inter, Arial'; g.fillStyle = C.dim;
    if (st.stage === 1) g.fillText(`${Math.round(st.reveal * N)} samples`, 16, 52);
    if (st.stage === 2) g.fillText(`Gaussian ${Math.max(1, st.shown)} of ${st.k}: unit covariance, a random position`, 16, 52);
    if (st.stage === 3 && !STATIC) g.fillText(`EM step ${st.iter}${st.done ? ' · done' : ''}`, 16, 52);
    if (st.stage >= 2 && vis.length) { g.fillStyle = C.accent; g.fillText(`−(1/N) Σ log p_θ(x_i) = ${loss(visN).toFixed(3)}`, 16, STATIC ? 52 : 72); }
  }
  // one frame of the EM animation; true when the fit is finished
  function advanceFit(now) {
    if (!target) { from = comps; target = emStep(from); t0 = now; if (Math.abs(loss(target) - loss(from)) < 1e-5 || st.iter >= MAX_IT) { target = null; return true; } }
    const u = Math.min(1, (now - t0) / STEP_MS); comps = lerp(from, target, u);
    if (u >= 1) { comps = target; from = target; target = null; st.iter++; }
    return false;
  }
  // manual fit (button)
  function fitTick(now) { if (!st.playing || !canvas.isConnected) return; if (advanceFit(now)) { st.done = true; st.playing = false; setBtn(); draw(); return; } draw(); requestAnimationFrame(fitTick); }
  // the loop: 1 → 2 → 3 → (new start) 1 → …
  let phase = 0, pt0 = 0, doneAt = 0;
  const enter = (ph, now) => { phase = ph; pt0 = now; st.stage = ph; if (ph === 1) { st.reveal = 0; } if (ph === 2) { init(); st.shown = 0; } if (ph === 3) { st.shown = st.k; doneAt = 0; } };
  function autoTick(now) {
    if (!st.auto || !canvas.isConnected) return;
    const dt = now - pt0;
    if (phase === 1) { st.reveal = Math.min(1, dt / T_REVEAL); if (dt > T_REVEAL + T_HOLD1) enter(2, now); }
    else if (phase === 2) { st.shown = Math.min(st.k, Math.floor(dt / T_ADD) + 1); if (dt > st.k * T_ADD + T_HOLD2) enter(3, now); }
    else if (phase === 3) {
      if (!st.done) { if (advanceFit(now) || dt > FIT_MAX_MS) { st.done = true; doneAt = now; } }
      else if (now - doneAt > T_HOLD3) { st.seed++; enter(1, now); }
    }
    draw(); requestAnimationFrame(autoTick);
  }
  const startAuto = () => { st.playing = false; st.auto = true; st.seed = cfg.seed || 7; enter(1, performance.now()); setBtn(); requestAnimationFrame(autoTick); };
  const manual = () => { st.auto = false; st.playing = false; if (target) { comps = target; st.iter++; target = null; } st.reveal = 1; st.shown = st.k; setBtn(); };
  const setBtn = () => { fitBtn.textContent = st.playing ? '⏸' : st.done ? '✓ fitted' : '▶ fit'; loopBtn.classList.toggle('active', st.auto); };
  const fitBtn = h('button', { style: 'min-width:4.6em', onclick: () => { if (st.playing) { manual(); return; } manual(); if (st.stage < 2) { st.stage = 2; init(); } st.stage = 3; if (!st.done) { st.playing = true; setBtn(); requestAnimationFrame(fitTick); } } }, '▶ fit');
  const stepBtn = h('button', { onclick: () => { manual(); if (st.stage < 2) { st.stage = 2; init(); } st.stage = 3; if (!st.done) { comps = emStep(comps); st.iter++; } draw(); } }, 'step');
  const resetBtn = h('button', { onclick: () => { manual(); st.seed++; st.stage = 2; init(); draw(); } }, '↺ new start');
  const loopBtn = h('button', { onclick: () => { if (!st.auto) startAuto(); } }, '⟳ loop');
  const kOut = h('span', { class: 'dim', style: 'display:inline-block; width:1.4em; font-variant-numeric:tabular-nums' }, String(st.k));
  const kSlider = h('input', { type: 'range', min: 2, max: 12, step: 1, value: st.k, style: 'width:6em', oninput: e => { st.k = +e.target.value; kOut.textContent = String(st.k); manual(); st.stage = 2; init(); draw(); } });
  el.classList.add('widget');
  const fitNow = () => { st.stage = 3; init(); for (let i = 0; i < MAX_IT; i++) { const nx = emStep(comps); const stop = Math.abs(loss(nx) - loss(comps)) < 1e-5; comps = nx; st.iter++; if (stop) break; } st.done = true; draw(); };
  if (STATIC) { el.append(canvas); fitNow(); return; }
  el.append(canvas, h('div', { class: 'wctl interactive-only', style: 'gap:0.5em; margin-top:0.3em; justify-content:center' }, h('label', {}, 'K'), kSlider, kOut, fitBtn, stepBtn, resetBtn, loopBtn));
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!canvas.isConnected) return; if (ev.currentSlide?.contains(el)) { if (cfg.loop !== false) startAuto(); else { manual(); st.stage = 0; draw(); } } else manual(); });
  if (isPrint()) { fitNow(); return; }
  draw();
  if (cfg.loop !== false && el.closest('section')?.classList.contains('present')) startAuto();   // mounted on the current slide: start now
}
