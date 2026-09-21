// 2D flow matching, live. A velocity field v_θ(t, x) (an MLP pretrained in Node by experiments/flow2d_train.mjs, or
// trained here in the browser) moves samples of a simple source distribution onto a toy data distribution.
// Modes: flow (particles + field over time), euler (how many steps?), pairs (the conditional training targets),
//        marginal (conditional arrows at one point, their posterior average vs. the network), train (watch it learn).
// config: { source: "gauss" | "gauss8", target: "moons" | "spiral" | "rings", mode, modes: [..] (buttons),
//           t, steps, pairs, field: true, data: true (show the data points), print_t, print_data, width, height, autoplay,
//           loop: true (flow mode: plays by itself on arrival and starts over after a short hold;
//                       euler mode: a loop with captions drawn on the canvas: first the learned (curved) paths, then one
//                       straight Euler step, then 2, 4, 8, 64 steps, the particles hopping step by step),
//           story: true (euler mode: the canvas in the middle, big captions beside it, the controls below),
//           click: true (euler mode: the stages advance with the arrow keys: elements <span class="fragment" data-stage="1..5"> on the
//                        slide are the clicks; each plays its stage once and holds; going back shows the previous stage) }
import { LIGHT } from '../theme.js';
import { C, h, fmt, isPrint, arrow, speedColor } from './util.js';
import { rng, SOURCES, TARGETS, sample, makeNet, cfmStep, velocity, euler, midpoint, importNet, nParams } from './flowlib.js';

const COMBOS = { 'gauss→moons': 'Gaussian → two moons', 'gauss8→moons': '8 Gaussians → two moons', 'gauss→spiral': 'Gaussian → spiral', 'gauss→rings': 'Gaussian → two rings' };
const MODES = { flow: 'flow', euler: 'Euler steps', pairs: 'training pairs', marginal: 'average velocity', train: 'train' };
const ACC = 'rgba(242,177,52,0.55)', DIM = '#5a606c', TRAJ = 60;

export async function mount(el, cfg) {
  const W = cfg.width || 440, H = cfg.height || 440, R = 3.5;
  const st = { source: cfg.source || 'gauss', target: cfg.target || 'moons', mode: cfg.mode || 'flow', t: cfg.t ?? 0,
    nSteps: cfg.steps ?? 8, eu: 1, euAuto: false, euKind: 'euler', pairs: cfg.pairs ?? 8, showField: cfg.field ?? true, showData: cfg.data ?? true, showNet: cfg.shownet ?? false, seed: 1,
    point: (cfg.point || [0.4, 0.9]).slice(), playing: false, training: false, step: 0, loss: [], lossEma: null };
  const sx = x => (x + R) / (2 * R) * W, sy = y => H - (y + R) / (2 * R) * H;
  const canvas = h('canvas', { width: W, height: H, style: 'cursor:crosshair' }), g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.5' });
  const lossC = h('canvas', { width: 240, height: 90 });
  let net, trainNet, data, src, traj, eulerCache = {}, refAll = null, pairSet, vref = 1;

  async function loadNet() {
    const key = `${st.source}→${st.target}`;
    const json = await (await fetch(`../../assets/lecture05/data/flow_${st.source}_${st.target}.json`)).json();
    net = importNet(json); traj = null; eulerCache = {}; refAll = null;
    // reference speed for the arrow colours: the largest speed on the grid over the whole time interval
    vref = 0;
    for (let k = 0; k <= 8; k++) for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) { const [vx, vy] = velocity(net, k / 8, -R + (i + 0.5) * 2 * R / 13, -R + (j + 0.5) * 2 * R / 13); vref = Math.max(vref, Math.hypot(vx, vy)); }
    return key;
  }
  function makeData() {
    const r = rng(3);
    data = sample(TARGETS[st.target], 600, r);
    src = sample(SOURCES[st.source], 400, r);
    traj = null; eulerCache = {}; refAll = null; pairSet = null;
  }
  const trajectories = () => (traj ||= src.map(p => euler(net, p, TRAJ)));
  const nearest = pts => pts.reduce((s, x) => s + Math.sqrt(data.reduce((m, p) => Math.min(m, (p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2), 1e9)), 0) / pts.length;

  // ---- drawing helpers ----
  function clear() { g.fillStyle = C.bg2; g.fillRect(0, 0, W, H); }
  function dots(pts, color, rad) { g.fillStyle = color; for (const [x, y] of pts) { g.beginPath(); g.arc(sx(x), sy(y), rad, 0, 7); g.fill(); } }
  // arrows coloured by speed (blue slow, red fast), lengths clipped to the grid spacing
  function field(t, model = net) {
    const n = 13, step = 2 * R / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = -R + (i + 0.5) * step, y = -R + (j + 0.5) * step, [vx, vy] = velocity(model, t, x, y), L = Math.hypot(vx, vy);
      const s = Math.min(step * 0.85, L * 0.16) / (L + 1e-9);
      arrow(g, sx(x), sy(y), sx(x + vx * s), sy(y + vy * s), speedColor(L / vref), 1.4);
    }
    const lw = 60, x0 = W - lw - 34, y0 = H - 12;
    for (let i = 0; i < lw; i++) { g.fillStyle = speedColor(i / (lw - 1)); g.fillRect(x0 + i, y0, 1.5, 6); }
    g.font = '11px Inter, Arial'; g.fillStyle = C.dim; g.textAlign = 'right'; g.fillText('slow', x0 - 4, y0 + 6); g.textAlign = 'left'; g.fillText('fast', x0 + lw + 4, y0 + 6);
  }
  function polyline(pts, color, width, dash) {
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.beginPath();
    pts.forEach(([x, y], i) => (i ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y)))); g.stroke(); g.setLineDash([]);
  }
  const combo = () => COMBOS[`${st.source}→${st.target}`];

  // ---- modes ----
  function drawFlow() {
    clear(); if (st.showData) dots(data, ACC, 2.5);
    if (st.showField) field(st.t);
    const k = Math.round(st.t * TRAJ);
    dots(trajectories().map(p => p[k]), C.blue, 3);
    readout.innerHTML = `<div>${combo()}</div><div>time <b>t = ${fmt(st.t, 2)}</b></div>` +
      `<div class="dim"><span style="color:${C.blue}">●</span> ${st.t === 0 ? 'samples of the noise distribution' : 'the same samples, moved along the learned field'}${st.showData ? ` · <span style="color:${C.accent}">●</span> data` : ''}${st.showField ? ' · arrows: v<sub>θ</sub>(t, x)' : ''}</div>`;
  }
  // a path followed up to the fraction u of its steps: whole segments, then part of the next one (Euler hops are straight)
  const partial = (path, u) => { u = Math.max(0, Math.min(1, u)); const n = path.length - 1, k = Math.min(n, Math.floor(u * n)), pts = path.slice(0, k + 1); if (k < n) { const s = u * n - k, a = path[k], b = path[k + 1]; pts.push([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])]); } return pts; };
  // caption band at the top of the canvas (the story is told on the animation)
  const capEl = cfg.story ? h('div', { style: 'flex:none; width:560px; line-height:1.3' }) : null;
  function caption(l1, l2, l3) {
    if (capEl) { capEl.innerHTML = `<div style="font-size:1.6em; font-weight:700; color:${C.blue}; margin-bottom:0.35em">${l1}</div><div style="font-size:1.2em">${l2 || ''}</div>`; return; }   // no third line: the details distract
    g.fillStyle = 'rgba(30,33,40,0.88)'; g.fillRect(0, 0, W, l3 ? 66 : 50);
    g.textAlign = 'left'; g.fillStyle = C.fg; g.font = 'bold 15px Inter, Arial'; g.fillText(l1, 10, 21);
    g.fillStyle = C.dim; g.font = '12px Inter, Arial'; if (l2) g.fillText(l2, 10, 40); if (l3) g.fillText(l3, 10, 56);
  }
  function drawEuler() {
    clear(); dots(data, ACC, 2.5);
    const n = st.nSteps, u = st.eu, show = src.slice(0, 14);
    refAll ||= src.map(p => midpoint(net, p, 64));
    const ref = refAll.slice(0, 14);
    if (st.euKind === 'flow') {   // the learned flow itself: every point follows its curved path
      dots(refAll.map(p => partial(p, u).at(-1)), 'rgba(90,176,255,0.45)', 2.5);
      ref.forEach(p => polyline(partial(p, u), C.fg, 1.5));
      dots(ref.map(p => partial(p, u).at(-1)), C.blue, 4);
      caption('The learned marginal flow','complicated bended paths', `t = ${fmt(u, 2)}`);
      readout.innerHTML = `<div>${combo()}</div>`;
      return;
    }
    const c = eulerCache[n] ||= { paths: show.map(p => euler(net, p, n)), all: src.map(p => euler(net, p, n)) };
    c.finals ||= c.all.map(p => p.at(-1));
    ref.forEach(p => polyline(p, DIM, 1, [4, 4]));
    dots(c.all.map(p => partial(p, u).at(-1)), 'rgba(90,176,255,0.45)', 2.5);
    c.paths.forEach(p => { const q = partial(p, u); polyline(q, C.fg, 1.5); dots(q.slice(0, -1), C.fg, 2.5); });
    dots(c.paths.map(p => partial(p, u).at(-1)), C.blue, 4);
    const k = Math.min(n, Math.floor(u * n + 1e-9)), dist = fmt(nearest(c.finals), 3);
    if (n === 1) caption('Single step inference', 'straight along the velocity at t = 0', `mean distance to the data: ${dist}`);
    else caption(`${n} integration steps ${u < 1 ? ` · step ${k + 1}` : ''}`, n <= 4 ? 'few steps cut the bends: samples land off the data' : 'more steps follow the curves (dashed: the exact flow)', `mean distance to the data: ${dist}`);
    readout.innerHTML = `<div>${combo()}</div>`;
  }
  function drawPairs() {
    clear(); dots(src, DIM, 2); dots(data, ACC, 2.5);
    if (!pairSet) { const r = rng(st.seed * 17); pairSet = Array.from({ length: 30 }, () => [SOURCES[st.source](r), TARGETS[st.target](r)]); }
    const P = pairSet.slice(0, st.pairs), t = st.t;
    let err = 0;
    P.forEach(([a, b]) => polyline([a, b], 'rgba(154,161,174,0.5)', 1));
    P.forEach(([a, b]) => {
      const x = (1 - t) * a[0] + t * b[0], y = (1 - t) * a[1] + t * b[1], ux = b[0] - a[0], uy = b[1] - a[1];
      arrow(g, sx(x), sy(y), sx(x + 0.22 * ux), sy(y + 0.22 * uy), C.accent, 2.5);
      if (st.showNet) { const [vx, vy] = velocity(net, t, x, y); arrow(g, sx(x), sy(y), sx(x + 0.22 * vx), sy(y + 0.22 * vy), C.blue, 2); err += Math.hypot(vx - ux, vy - uy); }
      g.fillStyle = C.fg; g.beginPath(); g.arc(sx(x), sy(y), 3.5, 0, 7); g.fill();
    });
    readout.innerHTML = `<div>${combo()}</div><div><b>${st.pairs}</b> pair${st.pairs > 1 ? 's' : ''} (x<sub>0</sub>, x<sub>1</sub>) · time <b>t = ${fmt(t, 2)}</b></div>` +
      `<div class="dim"><span style="color:${C.fg}">●</span> x<sub>t</sub> = (1−t) x<sub>0</sub> + t x<sub>1</sub> · <span style="color:${C.accent}">→</span> target velocity x<sub>1</sub> − x<sub>0</sub>` +
      (st.showNet ? ` · <span style="color:${C.blue}">→</span> network v<sub>θ</sub>(t, x<sub>t</sub>)</div><div>mean |v<sub>θ</sub> − (x<sub>1</sub> − x<sub>0</sub>)| = <b>${fmt(err / P.length, 2)}</b></div>` : '</div>');
  }
  function drawMarginal() {
    clear(); dots(data, ACC, 2.5);
    const t = Math.min(st.t, 0.9), [px, py] = st.point, s2 = (1 - t) ** 2;
    const w = data.map(([a, b]) => Math.exp(-((px - t * a) ** 2 + (py - t * b) ** 2) / (2 * s2)));
    const Z = w.reduce((s, v) => s + v, 0) || 1, wmax = Math.max(...w) || 1;
    let ux = 0, uy = 0;
    data.forEach(([a, b], i) => { const p = w[i] / Z; ux += p * (a - px) / (1 - t); uy += p * (b - py) / (1 - t); });
    data.forEach(([a, b], i) => { const o = w[i] / wmax; if (o > 0.02) polyline([[px, py], [a, b]], `rgba(154,161,174,${0.8 * o})`, 1); });
    const [vx, vy] = velocity(net, t, px, py);
    arrow(g, sx(px), sy(py), sx(px + 0.22 * ux), sy(py + 0.22 * uy), C.accent, 4);
    arrow(g, sx(px), sy(py), sx(px + 0.22 * vx), sy(py + 0.22 * vy), C.blue, 2);
    g.fillStyle = C.fg; g.beginPath(); g.arc(sx(px), sy(py), 5, 0, 7); g.fill();
    const eff = 1 / data.reduce((s, _, i) => s + (w[i] / Z) ** 2, 0);
    readout.innerHTML = `<div>${combo()}</div><div>point <b>x = (${fmt(px, 1)}, ${fmt(py, 1)})</b> at time <b>t = ${fmt(t, 2)}</b> (drag it)</div>` +
      `<div class="dim">lines: conditional directions towards data points x<sub>1</sub>, opacity = posterior weight p(x<sub>1</sub> | x<sub>t</sub> = x) · about ${Math.round(eff)} data points matter here</div>` +
      `<div><span style="color:${C.accent}">→</span> their weighted average u<sub>t</sub>(x) · <span style="color:${C.blue}">→</span> the network v<sub>θ</sub>(t, x): differ by <b>${fmt(Math.hypot(ux - vx, uy - vy), 2)}</b></div>`;
  }
  function drawTrain() {
    const model = st.pretrained ? net : trainNet;
    clear(); dots(data, ACC, 2.5);
    if (st.showField) field(0.5, model);
    dots(src.slice(0, 250).map(p => euler(model, p, 12).at(-1)), C.blue, 3);
    const LW = lossC.width, LH = lossC.height, gl = lossC.getContext('2d');
    gl.fillStyle = C.bg2; gl.fillRect(0, 0, LW, LH);
    const L = st.loss;
    if (L.length > 1) {
      const lo = Math.min(...L) * 0.9, hi = L[0] * 1.05;
      gl.strokeStyle = '#3987e5'; gl.lineWidth = 2; gl.beginPath();
      L.forEach((v, i) => { const x = 8 + i / (L.length - 1) * (LW - 16), y = LH - 18 - (v - lo) / (hi - lo + 1e-9) * (LH - 30); i ? gl.lineTo(x, y) : gl.moveTo(x, y); });
      gl.stroke();
    }
    gl.font = '11px Inter, Arial'; gl.fillStyle = C.dim; gl.fillText('conditional flow-matching loss (running average)', 8, LH - 4);
    readout.innerHTML = `<div>${combo()}</div><div>MLP 3 → 64 → 64 → 64 → 2, <b>${nParams(model)}</b> parameters</div>` +
      (st.pretrained ? `<div>pretrained (6,000 steps)</div>` : `<div>Adam steps: <b>${st.step}</b> · loss <b>${st.lossEma === null ? '–' : fmt(st.lossEma, 2)}</b></div>`) +
      `<div class="dim"><span style="color:${C.blue}">●</span> samples generated by the current network (12 Euler steps) · <span style="color:${C.accent}">●</span> data</div>`;
  }
  const DRAW = { flow: drawFlow, euler: drawEuler, pairs: drawPairs, marginal: drawMarginal, train: drawTrain };
  const draw = () => DRAW[st.mode]();

  // ---- animation: flow playback and live training ----
  function play() {
    if (st.mode !== 'flow') return;
    st.playing = !st.playing; playBtn.textContent = st.playing ? '⏸ pause' : '▶ play';
    if (st.t >= 1) st.t = 0;
    let last = performance.now();
    const tick = now => {
      if (!st.playing || !canvas.isConnected) return;
      st.t = Math.min(1, st.t + (now - last) / 3500); last = now; tSlider.value = st.t; draw();
      if (st.t < 1) requestAnimationFrame(tick);
      else if (cfg.loop) setTimeout(() => { if (st.playing && canvas.isConnected) { st.t = 0; last = performance.now(); requestAnimationFrame(tick); } }, 1500);   // hold, then again
      else { st.playing = false; playBtn.textContent = '▶ replay'; }
    };
    if (st.playing) requestAnimationFrame(tick);
  }
  function resetTrain() { trainNet = makeNet([3, 64, 64, 64, 2], rng(st.seed * 7 + 1)); st.opt = { lr: 2e-3, step: 0 }; st.r = rng(st.seed * 11 + 5); st.step = 0; st.loss = []; st.lossEma = null; st.pretrained = false; }
  function train() {
    st.training = !st.training; trainBtn.textContent = st.training ? '⏸ pause' : '▶ train';
    let frame = 0;
    const tick = () => {
      if (!st.training || !canvas.isConnected) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 28) {
        const l = cfmStep(trainNet, st.opt, st.r, SOURCES[st.source], TARGETS[st.target], 128);
        st.lossEma = st.lossEma === null ? l : 0.97 * st.lossEma + 0.03 * l; st.step++;
        if (st.step % 25 === 0) st.loss.push(st.lossEma);
        if (st.step === 2500) st.opt.lr = 7e-4;
      }
      if (++frame % 3 === 0) draw();
      if (st.step < 6000) requestAnimationFrame(tick); else { st.training = false; trainBtn.textContent = '▶ train'; draw(); }
    };
    if (st.training) requestAnimationFrame(tick);
  }

  // ---- the Euler loop: 1, 2, 4, 8, 64 steps, the particles hopping along their straight steps, a hold after each ----
  // stages: the learned flow, then one straight step, then more and more steps
  const EU_SEQ = [{ kind: 'flow', n: 64, run: 3200, hold: 1300 }, { kind: 'euler', n: 1, run: 1500, hold: 2000 }, { kind: 'euler', n: 2, run: 1500, hold: 1300 }, { kind: 'euler', n: 4, run: 1500, hold: 1300 }, { kind: 'euler', n: 8, run: 1500, hold: 1300 }, { kind: 'euler', n: 64, run: 1800, hold: 1800 }];
  let euStage = 0, euT0 = 0, euAnimRaf = 0;
  const CLICK = cfg.click === true;
  const enterStage = (i, now) => { euStage = i; euT0 = now; st.eu = 0; st.euKind = EU_SEQ[i].kind; setSteps(EU_SEQ[i].n); };
  function euTick(now) {
    if (!st.euAuto || st.euPaused || !canvas.isConnected || st.mode !== 'euler') return;
    const sg = EU_SEQ[euStage], dt = now - euT0;
    if (dt < sg.run) st.eu = Math.max(0, Math.min(1, dt / sg.run));
    else if (dt < sg.run + sg.hold) st.eu = 1;
    else enterStage((euStage + 1) % EU_SEQ.length, now);
    draw(); requestAnimationFrame(euTick);
  }
  const setSteps = n => { st.nSteps = n; stepBtns.forEach(b => b.classList.toggle('active', b.textContent === String(n))); };
  const startEu = () => { st.euAuto = true; st.euPaused = false; pauseBtn.textContent = '⏸'; enterStage(0, performance.now()); loopBtn.classList.add('active'); requestAnimationFrame(euTick); };
  const stopEu = () => { cancelAnimationFrame(euAnimRaf); st.euAuto = false; st.euPaused = false; st.eu = 1; st.euKind = 'euler'; pauseBtn.textContent = '▶'; loopBtn.classList.remove('active'); };
  // click mode: stage i plays once (animate) or is shown finished; the arrow keys step through the fragments of the slide
  function playStage(i, animate = true) {
    cancelAnimationFrame(euAnimRaf); euStage = i; st.euKind = EU_SEQ[i].kind; setSteps(EU_SEQ[i].n);
    if (!animate) { st.eu = 1; draw(); return; }
    st.eu = 0; const t0 = performance.now(), run = EU_SEQ[i].run;
    const tick = now => { st.eu = Math.max(0, Math.min(1, (now - t0) / run)); draw(); if (st.eu < 1 && canvas.isConnected && euStage === i) euAnimRaf = requestAnimationFrame(tick); };
    euAnimRaf = requestAnimationFrame(tick);
  }
  const sec = el.closest('section');
  const stageFromDom = () => { let n = 0; sec?.querySelectorAll('.fragment[data-stage].visible').forEach(f => { n = Math.max(n, +f.dataset.stage); }); return n; };
  const replayBtn = h('button', { onclick: () => playStage(euStage, true) }, '↻ replay');
  // pause freezes the current frame; play resumes from the same moment (or starts the loop if it was stopped)
  let euPausedAt = 0;
  const pauseBtn = h('button', { style: 'min-width:2.4em', onclick: () => {
    if (!st.euAuto) { startEu(); return; }
    if (st.euPaused) { st.euPaused = false; euT0 += performance.now() - euPausedAt; pauseBtn.textContent = '⏸'; requestAnimationFrame(euTick); }
    else { st.euPaused = true; euPausedAt = performance.now(); pauseBtn.textContent = '▶'; }
  } }, '⏸');
  const loopBtn = h('button', { onclick: () => { if (!st.euAuto) { startEu(); } } }, '⟳ loop');
  // ---- controls ----
  const playBtn = h('button', { onclick: play }, '▶ play');
  const trainBtn = h('button', { onclick: train }, '▶ train');
  const tSlider = h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: st.t, style: 'flex:1; min-width:5em; width:auto', oninput: e => { st.t = +e.target.value; draw(); } });
  const stepBtns = [1, 2, 4, 8, 16, 64].map(n => h('button', { class: n === st.nSteps ? 'active' : '', onclick: e => { stopEu(); st.nSteps = n; stepBtns.forEach(b => b.classList.toggle('active', b === e.target)); draw(); } }, String(n)));
  const pairBtns = [1, 4, 8, 30].map(n => h('button', { class: n === st.pairs ? 'active' : '', onclick: e => { st.pairs = n; pairBtns.forEach(b => b.classList.toggle('active', b === e.target)); draw(); } }, String(n)));
  const check = (key, label) => h('label', { style: 'display:flex; gap:0.3em; align-items:center' }, h('input', { type: 'checkbox', ...(st[key] ? { checked: '' } : {}), onchange: e => { st[key] = e.target.checked; draw(); } }), label);
  const dataSel = h('select', { style: 'font:inherit; background:#272b34; color:#e7e9ee; border:1px solid #3a3f4b; border-radius:6px; max-width:100%' },
    Object.entries(COMBOS).map(([k, v]) => { const o = h('option', { value: k }, v); if (k === `${st.source}→${st.target}`) o.selected = true; return o; }));
  dataSel.addEventListener('change', async () => {
    [st.source, st.target] = dataSel.value.split('→');
    st.playing = false; playBtn.textContent = '▶ play'; st.t = cfg.t ?? 0; tSlider.value = st.t;   // a new case starts over from t = 0
    st.training = false; trainBtn.textContent = '▶ train';
    await loadNet(); makeData(); resetTrain(); draw();
  });
  const row = (...kids) => h('div', { style: 'display:flex; gap:0.5em; align-items:center; flex-wrap:wrap' }, ...kids);
  const resampleBtn = h('button', { onclick: () => { st.seed++; pairSet = null; draw(); } }, 'resample');
  const resetBtn = h('button', { onclick: () => { st.training = false; trainBtn.textContent = '▶ train'; st.seed++; resetTrain(); draw(); } }, 'reset');
  // rows are built on demand: the slider and buttons are single DOM nodes and must sit in the current mode's row only
  const rows = {
    flow: () => [row(h('label', {}, 'time t'), tSlider), row(playBtn, check('showData', 'show the data'), check('showField', 'show the field'))],
    euler: () => [row(h('label', {}, 'steps'), ...stepBtns, ...(CLICK ? [replayBtn] : [pauseBtn, loopBtn]))],
    pairs: () => [row(h('label', {}, 'time t'), tSlider), row(h('label', {}, 'pairs'), ...pairBtns, resampleBtn), row(check('showNet', 'compare with the network'))],
    marginal: () => [row(h('label', {}, 'time t'), tSlider)],
    train: () => [row(trainBtn, resetBtn), row(check('showField', 'show the field at t = ½')), lossC],
  };
  const ctl = h('div', { class: 'wctl interactive-only', style: 'display:flex; flex-direction:column; gap:0.4em; align-items:stretch' });
  const modeBtns = (cfg.modes || []).map(m => h('button', { class: m === st.mode ? 'active' : '', onclick: e => { st.mode = m; st.playing = false; playBtn.textContent = '▶ play'; modeBtns.forEach(b => b.classList.toggle('active', b === e.target)); layout(); draw(); } }, MODES[m]));
  function layout() { ctl.replaceChildren(row(h('label', {}, 'data'), dataSel), ...rows[st.mode]()); }
  // drag the probe point (marginal mode)
  const toWorld = ev => { const r = canvas.getBoundingClientRect(); return [(ev.clientX - r.left) / r.width * 2 * R - R, R - (ev.clientY - r.top) / r.height * 2 * R]; };
  let dragging = false;
  canvas.addEventListener('pointerdown', ev => { if (st.mode !== 'marginal') return; dragging = true; st.point = toWorld(ev); draw(); });
  canvas.addEventListener('pointermove', ev => { if (dragging) { st.point = toWorld(ev); draw(); } });
  window.addEventListener('pointerup', () => { dragging = false; });

  el.classList.add('widget');
  if (cfg.story) {   // the animation in the middle, big captions beside it, the controls underneath
    ctl.style.cssText = 'display:flex; flex-direction:row; flex-wrap:wrap; gap:0.6em 1.2em; align-items:center; justify-content:center; margin-top:0.4em';
    el.append(h('div', { style: 'display:flex; gap:1.6em; align-items:center; justify-content:center' }, canvas, capEl), ctl);
  } else el.append(
    modeBtns.length ? h('div', { class: 'wctl interactive-only', style: 'margin:0 0 0.3em' }, modeBtns) : '',
    h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
      h('div', { style: `display:flex; flex-direction:column; gap:0.5em; width:${cfg.panel || 17}em; flex:none` }, readout, ctl)));
  await loadNet(); makeData(); resetTrain(); layout();
  if (isPrint() && st.mode === 'train') st.pretrained = true;
  if (isPrint() && cfg.print_t !== undefined) { st.t = cfg.print_t; tSlider.value = st.t; }
  if (isPrint() && cfg.print_data !== undefined) st.showData = cfg.print_data;
  draw();
  // arriving at the slide again: start over from t = 0 (the playback state would otherwise persist)
  el.closest('.reveal')?.addEventListener('slidechanged', ev => {
    if (!canvas.isConnected) return;
    if (CLICK) { if (ev.currentSlide?.contains(el)) { const n = stageFromDom(); playStage(n, n === 0); } else cancelAnimationFrame(euAnimRaf); return; }
    if (st.mode === 'euler') { if (ev.currentSlide?.contains(el)) { if (cfg.loop) startEu(); } else stopEu(); return; }
    if (st.mode !== 'flow') return;
    if (!ev.currentSlide?.contains(el)) { st.playing = false; playBtn.textContent = '▶ play'; return; }   // left the slide: stop
    st.playing = false; playBtn.textContent = '▶ play'; st.t = cfg.t ?? 0; tSlider.value = st.t; draw();
    if (cfg.loop) play();
  });
  if (CLICK) {
    const deckEl = el.closest('.reveal');
    deckEl?.addEventListener('fragmentshown', ev => { const f = ev.fragment; if (canvas.isConnected && f?.dataset.stage && sec?.contains(f)) playStage(+f.dataset.stage, true); });
    deckEl?.addEventListener('fragmenthidden', ev => { const f = ev.fragment; if (canvas.isConnected && f?.dataset.stage && sec?.contains(f)) playStage(+f.dataset.stage - 1, false); });
    if (isPrint()) playStage(EU_SEQ.length - 1, false);   // the PDF shows the last stage
    else if (sec?.classList.contains('present')) { const n = stageFromDom(); playStage(n, n === 0); }
    else playStage(0, false);
  }
  if ((cfg.autoplay || cfg.loop) && !isPrint() && el.closest('section')?.classList.contains('present')) setTimeout(() => { if (st.mode === 'flow') play(); else if (st.mode === 'train') train(); else if (st.mode === 'euler') startEu(); }, 400);
}
