// 1D change of variables: X ~ N(0, 1), Y = φ(X) = X + s·sin(t·X). Bottom, hanging from the x axis: the density of X.
// Middle: where φ sends a comb of equally spaced points (lines bunch up where φ compresses). Top: the density of Y,
// p_Y(φ(x)) = p_X(x) / φ'(x). Invertible (monotone) iff |s·t| < 1. "play" switches the deformation on gradually.
// config: { s: 0.1, t: 5, width: 1000, height: 400 }
import { C, h, fmt, isPrint } from './util.js';

export function mount(el, cfg) {
  const W = cfg.width || 1000, H = cfg.height || 400, X0 = -4, X1 = 4, ML = 48, MR = 24;
  const st = { s: cfg.s ?? 0.1, t: cfg.t ?? 5, tau: 1, playing: false };
  const phi = x => x + st.tau * st.s * Math.sin(st.t * x);
  const dphi = x => 1 + st.tau * st.s * st.t * Math.cos(st.t * x);
  const pX = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  const sx = x => ML + (x - X0) / (X1 - X0) * (W - ML - MR);
  const yY = 0.44 * H, yX = 0.64 * H, SC = 0.5 * H;   // the y axis (top) and the x axis (bottom); density scale (pixels per unit density)
  const canvas = h('canvas', { width: W, height: H }), g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.5' });

  function draw() {
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    g.font = '13px Inter, Arial';
    // axes with ticks
    for (const [y, lab] of [[yY, 'y = φ(x)'], [yX, 'x']]) {
      g.strokeStyle = '#5a606c'; g.lineWidth = 1; g.beginPath(); g.moveTo(ML, y); g.lineTo(W - MR, y); g.stroke();
      g.fillStyle = C.dim; g.textAlign = 'right'; g.fillText(lab, W - MR - 2, y - 5);
      for (let v = -3; v <= 3; v++) { g.beginPath(); g.moveTo(sx(v), y - 3); g.lineTo(sx(v), y + 3); g.stroke(); }
    }
    g.textAlign = 'center'; g.fillStyle = C.dim;
    for (let v = -3; v <= 3; v++) g.fillText(String(v), sx(v), yX - 7);
    // density of X, hanging down from the x axis
    g.beginPath(); g.moveTo(sx(X0), yX);
    for (let x = X0; x <= X1; x += 0.02) g.lineTo(sx(x), yX + pX(x) * SC);
    g.lineTo(sx(X1), yX); g.closePath(); g.fillStyle = 'rgba(90,176,255,0.18)'; g.fill(); g.strokeStyle = C.blue; g.lineWidth = 2; g.stroke();
    // comb: where φ sends equally spaced points, from the x axis up to the y axis
    g.strokeStyle = 'rgba(154,161,174,0.55)'; g.lineWidth = 1;
    for (let x = -3.6; x <= 3.6; x += 0.15) { g.beginPath(); g.moveTo(sx(x), yX); g.lineTo(sx(phi(x)), yY); g.stroke(); }
    // density of Y, upwards from the y axis: the point (φ(x), p_X(x)/φ'(x))
    g.beginPath(); g.moveTo(sx(phi(X0)), yY);
    for (let x = X0; x <= X1; x += 0.005) g.lineTo(sx(phi(x)), yY - pX(x) / Math.max(dphi(x), 0.05) * SC);
    g.lineTo(sx(phi(X1)), yY); g.closePath(); g.fillStyle = 'rgba(242,177,52,0.18)'; g.fill(); g.strokeStyle = C.accent; g.lineWidth = 2; g.stroke();
    g.textAlign = 'left'; g.fillStyle = C.accent; g.fillText('p_Y: density of Y = φ(X)', ML + 6, 16);
    g.fillStyle = C.blue; g.fillText('p_X: density of X ~ N(0, 1)', ML + 6, H - 8);
    const st_ = st.tau * st.s * st.t;
    readout.innerHTML = `<div>φ(x) = x + s · sin(t · x) with <b>s = ${fmt(st.s, 2)}</b>, <b>t = ${fmt(st.t, 1)}</b>${st.tau < 1 ? ` · deformation switched on to <b>${Math.round(st.tau * 100)}%</b>` : ''}</div>` +
      `<div class="dim">φ′(x) = 1 + s·t·cos(t·x) ∈ [${fmt(1 - st_, 2)}, ${fmt(1 + st_, 2)}] ${st.s * st.t >= 1 ? '<span style="color:' + C.red + '">· not invertible: |s·t| ≥ 1</span>' : '· invertible since |s·t| < 1'}</div>`;
  }
  function play() {
    st.playing = !st.playing; playBtn.textContent = st.playing ? '⏸ pause' : '▶ play';
    if (st.tau >= 1) st.tau = 0;
    let last = performance.now();
    const tick = now => {
      if (!st.playing || !canvas.isConnected) return;
      st.tau = Math.min(1, st.tau + (now - last) / 2500); last = now; draw();
      if (st.tau < 1) requestAnimationFrame(tick); else { st.playing = false; playBtn.textContent = '▶ replay'; }
    };
    if (st.playing) requestAnimationFrame(tick);
  }
  const playBtn = h('button', { onclick: play }, '▶ play');
  const slider = (key, min, max, step) => h('input', { type: 'range', min, max, step, value: st[key], oninput: e => { st[key] = +e.target.value; st.tau = 1; draw(); } });
  el.classList.add('widget');
  // controls first, with a fixed width, so that the changing readout text never moves the sliders
  readout.style.flex = '1'; readout.style.minWidth = '0';
  el.append(canvas, h('div', { style: 'display:flex; gap:1.5em; align-items:flex-start; margin-top:0.3em' },
    h('div', { class: 'wctl interactive-only', style: 'display:grid; grid-template-columns:7em auto; gap:0.3em 0.5em; align-items:center; flex:none; margin:0' },
      h('label', {}, 's (strength)'), slider('s', 0, 0.6, 0.01), h('label', {}, 't (frequency)'), slider('t', 0.5, 8, 0.1), playBtn, h('span', { class: 'dim' }, 'switch φ on gradually')),
    readout));
  draw();
}
