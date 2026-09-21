// Gradient descent vs SGD on a 2-parameter problem: fit y ≈ w·x + b to noisy data.
// Left: contour plot of the training loss L(w, b) with the optimizer's path. GD follows −∇L exactly;
// SGD uses the gradient of a random minibatch (an unbiased but noisy estimate).
// config: { batch: 4, lr: 0.1, width, height }
import { LIGHT } from '../theme.js';
import { C, h, fmt, isPrint } from './util.js';

function rng(seed) { let s = seed; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }

export function mount(el, cfg) {
  const W = cfg.width || 520, H = cfg.height || 400;
  const r0 = rng(11);
  // data: anisotropic so the valley is elongated (x has a large mean → w and b are correlated)
  const DATA = Array.from({ length: 64 }, () => { const x = 1.5 + 1.2 * (r0() - 0.5) * 2; return [x, 0.8 * x - 0.6 + 0.35 * (r0() - 0.5) * 2]; });
  const st = { lr: cfg.lr ?? 0.1, batch: cfg.batch ?? 4, mode: cfg.mode || 'both', running: false };
  const W0 = [-1.2, 1.6];                                // start (w, b)
  let paths;
  const range = { w: [-1.6, 2.2], b: [-2.6, 2.2] };
  const canvas = h('canvas', { width: W, height: H });
  const g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.55' });

  const loss = ([w, b], set = DATA) => set.reduce((s, [x, y]) => s + (w * x + b - y) ** 2, 0) / set.length;
  const grad = ([w, b], set) => set.reduce(([gw, gb], [x, y]) => { const e = 2 * (w * x + b - y) / set.length; return [gw + e * x, gb + e]; }, [0, 0]);

  // contour background, computed once
  const bgC = document.createElement('canvas'); bgC.width = W; bgC.height = H;
  (function contours() {
    const bg = bgC.getContext('2d'), img = bg.createImageData(W, H);
    const lmin = loss([0.8, -0.6]);
    for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
      const w = range.w[0] + px / W * (range.w[1] - range.w[0]), b = range.b[1] - py / H * (range.b[1] - range.b[0]);
      const v = Math.log(loss([w, b]) - lmin + 1e-3), band = (v * 2.2) - Math.floor(v * 2.2), k = 4 * (py * W + px);
      const f = Math.max(0, 1 - (v + 3) / 7), line = band < 0.06 ? 28 : 0;
      const shade = LIGHT ? 250 - 22 * f - 1.4 * line : 30 + 18 * f + line;
      img.data[k] = shade; img.data[k + 1] = shade + (LIGHT ? 1 : 3); img.data[k + 2] = shade + (LIGHT ? 4 : 10); img.data[k + 3] = 255;
    }
    bg.putImageData(img, 0, 0);
  })();
  const toC = ([w, b]) => [(w - range.w[0]) / (range.w[1] - range.w[0]) * W, (range.b[1] - b) / (range.b[1] - range.b[0]) * H];

  function reset() { paths = { gd: [W0.slice()], sgd: [W0.slice()] }; st.seed = 5; st.r = rng(17); draw(); }
  function step() {
    const gd = paths.gd.at(-1), sg = paths.sgd.at(-1);
    const gg = grad(gd, DATA);
    paths.gd.push([gd[0] - st.lr * gg[0], gd[1] - st.lr * gg[1]]);
    const mb = Array.from({ length: st.batch }, () => DATA[Math.floor(st.r() * DATA.length)]);
    const gs = grad(sg, mb);
    paths.sgd.push([sg[0] - st.lr * gs[0], sg[1] - st.lr * gs[1]]);
  }
  function draw() {
    g.drawImage(bgC, 0, 0);
    const path = (pts, color) => {
      g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
      pts.forEach((p, i) => { const [x, y] = toC(p); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
      const [x, y] = toC(pts.at(-1)); g.fillStyle = color; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill();
    };
    if (st.mode !== 'sgd') path(paths.gd, '#3987e5');
    if (st.mode !== 'gd') path(paths.sgd, '#c98500');
    const [sx, sy] = toC(W0); g.fillStyle = C.fg; g.beginPath(); g.arc(sx, sy, 4, 0, 7); g.fill();
    const [mx, my] = toC([0.8, -0.6]); g.strokeStyle = C.fg; g.lineWidth = 1.5; g.beginPath(); g.moveTo(mx - 6, my); g.lineTo(mx + 6, my); g.moveTo(mx, my - 6); g.lineTo(mx, my + 6); g.stroke();
    g.font = '13px Inter, Arial'; g.fillStyle = C.dim;
    g.fillText('w →', W - 40, H - 8); g.fillText('b ↑', 8, 16); g.fillText('start', sx + 7, sy - 6);
    readout.innerHTML =
      `<div><span style="color:#3987e5">●</span> GD (all ${DATA.length} points): loss <b>${fmt(loss(paths.gd.at(-1)), 3)}</b></div>` +
      `<div><span style="color:#c98500">●</span> SGD (batch ${st.batch}): loss <b>${fmt(loss(paths.sgd.at(-1)), 3)}</b></div>` +
      `<div class="dim">steps: ${paths.gd.length - 1} · cost per step: GD ${DATA.length} vs SGD ${st.batch} gradient terms</div>`;
  }
  let raf;
  function toggle() {
    st.running = !st.running; play.textContent = st.running ? '⏸ pause' : '▶ run';
    const tick = () => { if (!st.running || !canvas.isConnected) return; step(); draw(); if (paths.gd.length < 300) raf = setTimeout(() => requestAnimationFrame(tick), 60); else { st.running = false; play.textContent = '▶ run'; } };
    if (st.running) tick();
  }
  const play = h('button', { onclick: toggle }, '▶ run');
  const sel = (key, opts, label) => {
    const s = h('select', { style: 'font:inherit; background:#272b34; color:#e7e9ee; border:1px solid #3a3f4b; border-radius:6px' },
      opts.map(o => { const e = h('option', { value: o }, String(o)); if (o === st[key]) e.selected = true; return e; }));
    s.addEventListener('change', () => { st[key] = +s.value; reset(); });
    return [h('label', {}, label), s];
  };
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:16em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'display:grid; grid-template-columns:auto auto; gap:0.3em 0.5em; align-items:center' },
        ...sel('lr', [0.02, 0.05, 0.1, 0.2, 0.25], 'learning rate'), ...sel('batch', [1, 4, 16], 'SGD batch'),
        play, h('button', { onclick: () => { st.running = false; play.textContent = '▶ run'; reset(); } }, 'reset')))));
  reset();
  if (isPrint() || cfg.presteps) { for (let i = 0; i < (cfg.presteps || 60); i++) step(); draw(); }
}
