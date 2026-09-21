// Three panels sharing one time t, each placed next to (or above) its formula on the slide:
// 1. samples carried by the flow (φ_t), 2. the velocity field u_t (arrows coloured by speed: blue slow, red fast),
// 3. the samples at t = 1 next to the data (panels: 2 leaves this one out). Uses the pretrained fields of flow2d.
// config: { source, target, size: 165, layout: 'column' | 'row', gap: 24 (px, row layout), panels: 3 | 2, t: 0, print_t }
import { C, h, fmt, isPrint, arrow, speedColor, loopPlayer } from './util.js';
import { rng, SOURCES, TARGETS, sample, velocity, euler, importNet } from './flowlib.js';

const COMBOS = { 'gauss→moons': 'Gaussian → two moons', 'gauss8→moons': '8 Gaussians → two moons', 'gauss→spiral': 'Gaussian → spiral', 'gauss→rings': 'Gaussian → two rings' };
const ACC = 'rgba(242,177,52,0.55)', TRAJ = 60, R = 3.5;

export async function mount(el, cfg) {
  const S = cfg.size || 165, ROW = cfg.layout === 'row', GAP = cfg.gap ?? 24, NP = cfg.panels === 2 ? 2 : 3;
  const st = { source: cfg.source || 'gauss8', target: cfg.target || 'moons', t: cfg.t ?? 0 };
  const sx = x => (x + R) / (2 * R) * S, sy = y => S - (y + R) / (2 * R) * S;
  const cvs = [0, 1, 2].slice(0, NP).map(() => h('canvas', { width: S, height: S, style: `width:${S}px; height:${S}px; flex:none` })), gs = cvs.map(c => c.getContext('2d'));
  let net, data, src, traj, vref;
  const GRID = 9, step = 2 * R / GRID, gx = i => -R + (i + 0.5) * step;
  async function loadNet() {
    net = importNet(await (await fetch(`../../assets/lecture05/data/flow_${st.source}_${st.target}.json`)).json()); traj = null;
    // reference speed for the colours: the largest speed on the grid over the whole time interval (so colours are comparable across t)
    vref = 0;
    for (let k = 0; k <= 8; k++) for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) { const [vx, vy] = velocity(net, k / 8, gx(i), gx(j)); vref = Math.max(vref, Math.hypot(vx, vy)); }
  }
  function makeData() { const r = rng(3); data = sample(TARGETS[st.target], 500, r); src = sample(SOURCES[st.source], 300, r); traj = null; }
  const trajectories = () => (traj ||= src.map(p => euler(net, p, TRAJ)));
  const clear = (g, label) => { g.fillStyle = C.bg2; g.fillRect(0, 0, S, S); g.font = `${S > 250 ? 14 : 12}px Inter, Arial`; g.fillStyle = C.dim; g.fillText(label, 6, S > 250 ? 17 : 14); };
  const dots = (g, pts, color, rad) => { g.fillStyle = color; for (const [x, y] of pts) { g.beginPath(); g.arc(sx(x), sy(y), rad, 0, 7); g.fill(); } };
  function field(g, t) {
    for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
      const x = gx(i), y = gx(j), [vx, vy] = velocity(net, t, x, y), L = Math.hypot(vx, vy);
      const s = Math.min(step * 0.85, L * 0.16) / (L + 1e-9);
      arrow(g, sx(x), sy(y), sx(x + vx * s), sy(y + vy * s), speedColor(L / vref), S > 250 ? 2 : 1.4);
    }
    // legend: slow → fast
    const lw = S > 250 ? 70 : 44, x0 = S - lw - 34, y0 = S - 12;
    for (let i = 0; i < lw; i++) { g.fillStyle = speedColor(i / (lw - 1)); g.fillRect(x0 + i, y0, 1.5, 6); }
    g.font = '11px Inter, Arial'; g.fillStyle = C.dim; g.textAlign = 'right'; g.fillText('slow', x0 - 4, y0 + 6); g.textAlign = 'left'; g.fillText('fast', x0 + lw + 4, y0 + 6);
  }
  function draw() {
    const k = Math.round(st.t * TRAJ), T = trajectories(), rad = S > 250 ? 3 : 2.2, radD = S > 250 ? 2 : 1.5;
    clear(gs[0], `samples at t = ${fmt(st.t, 2)}`); dots(gs[0], data, ACC, radD); dots(gs[0], T.map(p => p[k]), C.blue, rad);
    clear(gs[1], `velocity field at t = ${fmt(st.t, 2)}`); field(gs[1], st.t);
    if (NP > 2) { clear(gs[2], 'samples at t = 1 · data'); dots(gs[2], data, ACC, radD); dots(gs[2], T.map(p => p[TRAJ]), C.blue, rad); }
  }
  // plays in a loop by itself (cfg.loop: false → only on demand): 0 → 1 in 3.5 s, a pause at the data, again from 0
  const loop = loopPlayer({ dur: 3500, alive: () => cvs[0].isConnected, step: t => { st.t = t; slider.value = t; draw(); } });
  const setBtn = () => { playBtn.textContent = loop.playing ? '⏸' : '▶'; };
  const play = () => { if (loop.playing) loop.stop(); else loop.start(st.t); setBtn(); };
  const reset = (autoplay = cfg.loop !== false) => { loop.stop(); st.t = cfg.t ?? 0; slider.value = st.t; if (autoplay && !isPrint()) loop.start(st.t); setBtn(); };
  const playBtn = h('button', { onclick: play, style: 'min-width:2.4em' }, '▶');
  const slider = h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: st.t, style: ROW ? 'width:16em' : 'width:7em', oninput: e => { loop.stop(); setBtn(); st.t = +e.target.value; draw(); } });
  const sel = h('select', { style: 'font:inherit; background:#272b34; color:#e7e9ee; border:1px solid #3a3f4b; border-radius:6px; max-width:100%; font-size:0.85em' },
    Object.entries(COMBOS).map(([k, v]) => { const o = h('option', { value: k }, v); if (k === `${st.source}→${st.target}`) o.selected = true; return o; }));
  sel.addEventListener('change', async () => { const wasPlaying = loop.playing; reset(false); [st.source, st.target] = sel.value.split('→'); await loadNet(); makeData(); draw(); if (wasPlaying) { loop.start(0); setBtn(); } });
  el.classList.add('widget');
  if (ROW) {
    el.append(h('div', { style: `display:flex; gap:${GAP}px; justify-content:center` }, ...cvs),
      h('div', { class: 'wctl interactive-only', style: 'justify-content:center; gap:0.8em; margin-top:0.3em' },
        h('div', { style: 'display:flex; gap:0.4em; align-items:center' }, h('label', {}, 'time t'), slider, playBtn), sel));
  } else {
    el.style.width = `${S}px`;
    el.append(h('div', { style: 'display:flex; flex-direction:column; gap:8px; align-items:flex-start' }, ...cvs),
      h('div', { class: 'wctl interactive-only', style: `flex-direction:column; align-items:stretch; gap:0.3em; margin-top:0.4em; width:${S}px` },
        h('div', { style: 'display:flex; gap:0.4em; align-items:center' }, h('label', {}, 't'), slider, playBtn), sel));
  }
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!cvs[0].isConnected) return; if (ev.currentSlide?.contains(el)) { reset(); draw(); } else if (loop.playing) { loop.stop(); setBtn(); } });
  await loadNet(); makeData();
  if (isPrint() && cfg.print_t !== undefined) { st.t = cfg.print_t; slider.value = st.t; }
  draw();
  if (!isPrint() && el.closest('section')?.classList.contains('present')) reset();   // mounted on the current slide: start now
}
