// Gaussian conditional probability paths in 1D as a ridge plot over time t: p_t(x | x1) = N(μ_t(x1), σ_t²) for the
// linear ("optimal transport") path or the variance-preserving diffusion path; and the marginal path
// p_t(x) = average over data points x1 of p_t(x | x1), which goes from N(0, 1) at t = 0 to the data at t = 1.
// config: { mode: "cond" | "marginal", modes: [...], path: "ot" | "vp", x1: -2.5, data: [...], rows: 6, width, height,
//           animate: true (one distribution climbs the time axis, leaving faint copies; loops by itself; print_t: 1),
//           legend: 'below' | 'right' (where the readout and the controls go) }
import { C, h, fmt, isPrint, loopPlayer } from './util.js';

const PATHS = { ot: 'linear (optimal transport)', vp: 'diffusion (variance preserving)' };
const PAL = ['#f2b134', '#5fd38d', '#ff6b6b', '#b28dff', '#ffd34d', '#4dd0e1', '#ff8a65', '#aed581'];   // one colour per data point (animated mixture)

export function mount(el, cfg) {
  const W = cfg.width || 560, H = cfg.height || 400, X0 = -5, X1 = 5, MR = 16, MB = 30, ANIM = cfg.animate === true, ML = ANIM ? 74 : 56, MT = ANIM ? 96 : 62;   // MT: headroom for the density at t = 1
  const st = { mode: cfg.mode || 'cond', path: cfg.path || 'ot', x1: cfg.x1 ?? -2.5, data: cfg.data || [-3.1, -2.4, -0.3, 1.6, 2.2, 2.9], rows: cfg.rows || 6, smin: 0.2, t: cfg.t ?? 0 };
  const sx = x => ML + (x - X0) / (X1 - X0) * (W - ML - MR);
  const alpha = s => Math.exp(-0.25 * s * s * 19.9 - 0.5 * s * 0.1);   // VP schedule: β(s) from 0.1 to 20
  const musig = (t, x1) => {
    if (st.path === 'ot') return [t * x1, 1 - (1 - st.smin) * t];
    const a = alpha(1 - t); return [a * x1, Math.max(st.smin, Math.sqrt(Math.max(0, 1 - a * a)))];
  };
  const N = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
  const dens = (t, x) => st.mode === 'cond' ? N(x, ...musig(t, st.x1)) : st.data.reduce((s, x1) => s + N(x, ...musig(t, x1)), 0) / st.data.length;
  const canvas = h('canvas', { width: W, height: H }), g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.5' });

  function curve(t, y, fill, stroke, lw, SC) {
    g.beginPath(); g.moveTo(sx(X0), y);
    for (let x = X0; x <= X1; x += 0.02) g.lineTo(sx(x), y - Math.min(dens(t, x), 2.6) * SC);
    g.lineTo(sx(X1), y); g.closePath();
    g.fillStyle = fill; g.fill(); g.strokeStyle = stroke; g.lineWidth = lw; g.stroke();
  }
  function draw() {
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    const gap = (H - MT - MB) / (st.rows - 1), SC = Math.min(gap * 0.95 / 2.0, (MT - 8) / 2.0);
    // the moving curve gets its own, larger scale: as tall as the headroom allows for the tallest density on the path
    const peak = Math.max(...[0, 0.5, 1].flatMap(t => Array.from({ length: 101 }, (_, i) => dens(t, X0 + (X1 - X0) * i / 100))));
    const SCl = ANIM ? Math.min(2.6 * gap, (MT - 10) / peak) : SC;
    g.font = '13px Inter, Arial';
    for (let k = 0; k < st.rows; k++) {
      const t = k / (st.rows - 1), y = H - MB - k * gap;
      g.strokeStyle = '#3a3f4b'; g.lineWidth = 1; g.beginPath(); g.moveTo(ML, y); g.lineTo(W - MR, y); g.stroke();
      if (!ANIM) { g.fillStyle = C.dim; g.textAlign = 'right'; g.fillText(`t = ${fmt(t, 1)}`, ML - 6, y + 4); }
      else { g.strokeStyle = '#5a606c'; g.beginPath(); g.moveTo(ML - 4, y); g.lineTo(ML + 2, y); g.stroke(); }   // tick on the time axis
      if (ANIM && t > st.t + 1e-9) continue;   // animation: only the rows the distribution has already passed
      curve(t, y, ANIM ? 'rgba(90,176,255,0.07)' : 'rgba(90,176,255,0.2)', ANIM ? 'rgba(90,176,255,0.35)' : C.blue, ANIM ? 1.2 : 1.8, SC);
    }
    if (ANIM) {   // the distribution at the current time, climbing the time axis
      const y = H - MB - st.t * (H - MT - MB);
      g.strokeStyle = '#5a606c'; g.lineWidth = 1; g.beginPath(); g.moveTo(ML - 4, H - MB); g.lineTo(ML - 4, MT); g.stroke();
      curve(st.t, y, 'rgba(90,176,255,0.3)', C.blue, 2.6, SCl);
      if (st.mode === 'marginal') st.data.forEach((x1, i) => {   // the parts of the mixture: one simple path per data point, weight 1/K
        const [m, sg] = musig(st.t, x1), col = PAL[i % PAL.length];
        g.beginPath(); g.moveTo(sx(X0), y);
        for (let x = X0; x <= X1; x += 0.02) g.lineTo(sx(x), y - Math.min(N(x, m, sg) / st.data.length, 2.6) * SCl);
        g.strokeStyle = col; g.lineWidth = 1.4; g.stroke();
        g.fillStyle = col; g.beginPath(); g.arc(sx(m), y, 3.5, 0, 7); g.fill();
      });
      g.fillStyle = C.accent; g.textAlign = 'right'; g.font = 'bold 14px Inter, Arial'; g.fillText(`t = ${st.t.toFixed(2)}`, ML - 10, y + 5); g.font = '13px Inter, Arial';
    }
    g.textAlign = 'center'; g.fillStyle = C.dim;
    for (let v = -4; v <= 4; v += 2) g.fillText(String(v), sx(v), H - MB + 16);
    g.fillText('x', W - MR, H - MB + 16);
    if (st.mode === 'cond') {   // the mean μ_t(x1) across time, and x1 itself on the top row
      g.strokeStyle = C.red; g.setLineDash([4, 4]); g.lineWidth = 1.5; g.beginPath();
      const tEnd = ANIM ? st.t : 1;   // the mean's path, up to the current time when animating
      for (let k = 0; k <= 40; k++) { const t = k / 40 * tEnd, y = H - MB - t * (H - MT - MB); k ? g.lineTo(sx(musig(t, st.x1)[0]), y) : g.moveTo(sx(musig(t, st.x1)[0]), y); }
      g.stroke(); g.setLineDash([]);
      if (ANIM) { g.fillStyle = C.red; g.beginPath(); g.arc(sx(musig(st.t, st.x1)[0]), H - MB - st.t * (H - MT - MB), 4, 0, 7); g.fill(); }
      g.fillStyle = C.accent; g.beginPath(); g.arc(sx(st.x1), MT, 5, 0, 7); g.fill();
      g.textAlign = 'left'; g.fillText('x₁', sx(st.x1) + 8, MT + 4);
    } else {
      st.data.forEach((x1, i) => { g.fillStyle = ANIM ? PAL[i % PAL.length] : C.accent; g.beginPath(); g.arc(sx(x1), MT, 4.5, 0, 7); g.fill(); });
      g.fillStyle = C.accent;
      g.textAlign = 'left'; g.fillText('data points x₁', sx(Math.max(...st.data)) + 10, MT + 4);
    }
    const f = st.path === 'ot' ? `μ<sub>t</sub>(x₁) = t · x₁, &nbsp; σ<sub>t</sub> = 1 − (1 − σ<sub>min</sub>) t` : `μ<sub>t</sub>(x₁) = α<sub>1−t</sub> · x₁, &nbsp; σ<sub>t</sub> = √(1 − α²<sub>1−t</sub>)`;
    readout.innerHTML = (st.mode === 'cond'
      ? `<div>p<sub>t</sub>(x | x₁) = N(μ<sub>t</sub>(x₁), σ<sub>t</sub>²) with <b>x₁ = ${fmt(st.x1, 1)}</b></div>`
      : `<div>p<sub>t</sub>(x) = average of p<sub>t</sub>(x | x₁) over the <b>${st.data.length}</b> data points</div>`) +
      `<div class="dim">${PATHS[st.path]} path: ${f}</div>`;
  }
  const sel = h('select', { style: 'font:inherit; background:#272b34; color:#e7e9ee; border:1px solid #3a3f4b; border-radius:6px' },
    Object.entries(PATHS).map(([k, v]) => { const o = h('option', { value: k }, v); if (k === st.path) o.selected = true; return o; }));
  sel.addEventListener('change', () => { st.path = sel.value; draw(); });
  const x1s = h('input', { type: 'range', min: -4, max: 4, step: 0.1, value: st.x1, oninput: e => { st.x1 = +e.target.value; draw(); } });
  const modeBtns = (cfg.modes || []).map(m => h('button', { class: m === st.mode ? 'active' : '', onclick: e => { st.mode = m; modeBtns.forEach(b => b.classList.toggle('active', b === e.target)); x1row.forEach(x => { x.style.display = m === 'cond' ? '' : 'none'; }); draw(); } }, m === 'cond' ? 'one data point' : 'all data points'));
  const x1row = [h('label', {}, 'x₁'), x1s];
  if (st.mode !== 'cond') x1row.forEach(x => { x.style.display = 'none'; });
  // animation: 0 → 1 in 5 s, a pause at t = 1, again; the controls take over (pause, slider)
  const loop = ANIM ? loopPlayer({ dur: 5000, hold: 1500, hold0: 500, alive: () => canvas.isConnected, step: t => { st.t = t; tSlider.value = t; draw(); } }) : null;
  const setBtn = () => { if (ANIM) playBtn.textContent = loop.playing ? '⏸' : '▶'; };
  const playBtn = h('button', { style: 'min-width:2.4em', onclick: () => { if (loop.playing) loop.stop(); else loop.start(st.t); setBtn(); } }, '▶');
  const tSlider = h('input', { type: 'range', min: 0, max: 1, step: 0.005, value: st.t, style: 'width:9em', oninput: e => { loop.stop(); setBtn(); st.t = +e.target.value; draw(); } });
  const timeRow = ANIM ? [h('label', {}, 'time t'), h('div', { style: 'display:flex; gap:0.4em; align-items:center' }, tSlider, playBtn)] : [];
  el.classList.add('widget');
  const wctl = h('div', { class: 'wctl interactive-only', style: 'display:grid; grid-template-columns:auto auto; gap:0.3em 0.5em; align-items:center; flex:none' },
    ...timeRow, h('label', {}, 'path'), sel, ...x1row, ...(modeBtns.length ? [h('label', {}, 'show'), h('div', { style: 'display:flex; gap:0.25em' }, modeBtns)] : []));
  if (cfg.legend === 'right') el.append(h('div', { style: 'display:flex; gap:1.2em; align-items:flex-start' }, canvas, h('div', { style: 'display:flex; flex-direction:column; gap:0.9em; align-items:flex-start; padding-top:0.3em' }, readout, wctl)));
  else el.append(canvas, h('div', { style: 'display:flex; gap:1.2em; align-items:flex-start; margin-top:0.3em' }, readout, wctl));
  if (isPrint()) { st.t = cfg.print_t ?? 1; tSlider.value = st.t; }
  draw();
  if (ANIM && !isPrint()) {
    const reset = () => { loop.stop(); st.t = cfg.t ?? 0; tSlider.value = st.t; if (cfg.loop !== false) loop.start(st.t); setBtn(); draw(); };
    el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!canvas.isConnected) return; if (ev.currentSlide?.contains(el)) reset(); else { loop.stop(); setBtn(); } });
    if (el.closest('section')?.classList.contains('present')) reset();   // mounted on the current slide: start now
  }
}
