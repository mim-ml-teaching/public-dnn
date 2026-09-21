// A small MLP trained live with minibatch SGD on a 2D binary classification problem.
// Decision regions (heat map), train points (filled) and test points (rings), train/test loss curves.
// Backprop is written out by hand (it is lecture 2's job to automate it).
// config: { dataset: "spiral" | "circles" | "xor" | "moons", layers: 2, width: 12, lr: 0.1, batch: 16,
//           noise: 0.1, autoplay: false, width_px, height_px }
import { LIGHT } from '../theme.js';
import { C, h, fmt, isPrint } from './util.js';

function rng(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 16807) % 2147483647) / 2147483647); }
function gauss(r) { const u = Math.max(1e-9, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

const DATA = {
  spiral(n, noise, r) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const lab = i % 2, t = (i / n) * 1.0 * 3.2 + 0.25, a = t * 1.9 + lab * Math.PI;
      out.push([t / 3.6 * Math.cos(a) + noise * gauss(r) * 0.6, t / 3.6 * Math.sin(a) + noise * gauss(r) * 0.6, lab]);
    }
    return out;
  },
  circles(n, noise, r) {
    return Array.from({ length: n }, (_, i) => {
      const lab = i % 2, rad = lab ? 0.25 + 0.15 * r() : 0.6 + 0.25 * r(), a = 2 * Math.PI * r();
      return [rad * Math.cos(a) + noise * gauss(r) * 0.5, rad * Math.sin(a) + noise * gauss(r) * 0.5, lab];
    });
  },
  xor(n, noise, r) {
    return Array.from({ length: n }, () => {
      let x = 2 * r() - 1, y = 2 * r() - 1;
      x += Math.sign(x) * 0.05; y += Math.sign(y) * 0.05;
      return [x * 0.9 + noise * gauss(r) * 0.4, y * 0.9 + noise * gauss(r) * 0.4, x * y > 0 ? 1 : 0];
    });
  },
  moons(n, noise, r) {
    return Array.from({ length: n }, (_, i) => {
      const lab = i % 2, a = Math.PI * r();
      const x = lab ? 1 - Math.cos(a) : Math.cos(a), y = lab ? 0.5 - Math.sin(a) : Math.sin(a);
      return [(x - 0.5) * 0.75 + noise * gauss(r) * 0.6, (y - 0.25) * 0.9 + noise * gauss(r) * 0.6, lab];
    });
  },
};

export function mount(el, cfg) {
  const W = cfg.width_px || 400, H = cfg.height_px || 400;
  const st = {
    dataset: cfg.dataset || 'spiral', layers: cfg.layers ?? 2, width: cfg.width ?? 12,
    lr: cfg.lr ?? 0.1, batch: cfg.batch ?? 16, noise: cfg.noise ?? 0.1, running: false, epoch: 0, seed: 1,
  };
  let net, train, test, hist;
  const canvas = h('canvas', { width: W, height: H });
  const g = canvas.getContext('2d');
  const lossC = h('canvas', { width: 260, height: 130 });
  const gl = lossC.getContext('2d');
  const readout = h('div', { style: 'line-height:1.5' });
  const RES = 64;
  const heat = g.createImageData(RES, RES);
  const off = document.createElement('canvas'); off.width = RES; off.height = RES;

  function makeData() {
    const r = rng(st.seed * 31 + 7), all = DATA[st.dataset](400, st.noise, r);
    for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
    train = all.slice(0, 200); test = all.slice(200);
  }
  function makeNet() {
    const r = rng(st.seed * 13 + st.layers * 7 + st.width);
    const sizes = [2, ...Array(st.layers).fill(st.width), 1];
    net = sizes.slice(1).map((m, l) => {
      const n = sizes[l], s = Math.sqrt(2 / n);           // He initialization
      return { W: Array.from({ length: m }, () => Array.from({ length: n }, () => gauss(r) * s)), b: new Array(m).fill(0) };
    });
    hist = { train: [], test: [] }; st.epoch = 0;
  }
  const nParams = () => net.reduce((s, L) => s + L.W.length * (L.W[0].length + 1), 0);

  // forward pass, keeping activations for backprop; hidden: ReLU, output: logit
  function forward(x) {
    const acts = [x], zs = [];
    let a = x;
    net.forEach((L, l) => {
      const z = L.W.map((row, i) => row.reduce((s, w, j) => s + w * a[j], L.b[i]));
      zs.push(z);
      a = l < net.length - 1 ? z.map(v => (v > 0 ? v : 0)) : z;
      acts.push(a);
    });
    return { acts, zs, logit: a[0] };
  }
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const bce = (z, y) => Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));   // stable log-loss

  function sgdStep(batch) {
    const grads = net.map(L => ({ W: L.W.map(r => r.map(() => 0)), b: L.b.map(() => 0) }));
    for (const [x0, x1, y] of batch) {
      const { acts, zs, logit } = forward([x0, x1]);
      let delta = [sigmoid(logit) - y];                   // dL/dz at the output (log-loss + sigmoid)
      for (let l = net.length - 1; l >= 0; l--) {
        const a = acts[l];
        delta.forEach((d, i) => { grads[l].b[i] += d; a.forEach((aj, j) => { grads[l].W[i][j] += d * aj; }); });
        if (l > 0) delta = a.map((_, j) => (zs[l - 1][j] > 0 ? net[l].W.reduce((s, row, i) => s + row[j] * delta[i], 0) : 0));
      }
    }
    net.forEach((L, l) => {
      L.W.forEach((row, i) => row.forEach((_, j) => { row[j] -= st.lr * grads[l].W[i][j] / batch.length; }));
      L.b.forEach((_, i) => { L.b[i] -= st.lr * grads[l].b[i] / batch.length; });
    });
  }
  function epoch() {
    const r = rng(st.seed * 1000 + st.epoch), idx = train.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (let i = 0; i < idx.length; i += st.batch) sgdStep(idx.slice(i, i + st.batch).map(k => train[k]));
    st.epoch++;
    const L = set => set.reduce((s, [a, b, y]) => s + bce(forward([a, b]).logit, y), 0) / set.length;
    hist.train.push(L(train)); hist.test.push(L(test));
  }
  const acc = set => set.reduce((s, [a, b, y]) => s + ((forward([a, b]).logit > 0 ? 1 : 0) === y ? 1 : 0), 0) / set.length;

  function draw() {
    const sx = x => (x + 1.1) / 2.2 * W, sy = y => H - (y + 1.1) / 2.2 * H;
    for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
      const x = -1.1 + 2.2 * (j + 0.5) / RES, y = 1.1 - 2.2 * (i + 0.5) / RES;
      const p = sigmoid(forward([x, y]).logit), k = 4 * (i * RES + j);
      // blend between blue (class 0) and amber (class 1) through the surface colour
      const c0 = [57, 135, 229], c1 = [201, 133, 0], bg = LIGHT ? [255, 255, 255] : [30, 33, 40], t = Math.abs(p - 0.5) * 2 * 0.55;
      const c = p > 0.5 ? c1 : c0;
      heat.data[k] = bg[0] + (c[0] - bg[0]) * t; heat.data[k + 1] = bg[1] + (c[1] - bg[1]) * t; heat.data[k + 2] = bg[2] + (c[2] - bg[2]) * t; heat.data[k + 3] = 255;
    }
    off.getContext('2d').putImageData(heat, 0, 0);
    g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, W, H);
    const dot = ([x, y, lab], ring) => {
      g.beginPath(); g.arc(sx(x), sy(y), ring ? 4 : 4.5, 0, 7);
      if (ring) { g.strokeStyle = lab ? '#f2b134' : '#5ab0ff'; g.lineWidth = 1.5; g.stroke(); }
      else { g.fillStyle = lab ? '#f2b134' : '#5ab0ff'; g.fill(); g.strokeStyle = '#1e2128'; g.lineWidth = 1.5; g.stroke(); }
    };
    test.forEach(p => dot(p, true)); train.forEach(p => dot(p, false));

    // loss curves
    const LW = lossC.width, LH = lossC.height;
    gl.fillStyle = C.bg2; gl.fillRect(0, 0, LW, LH);
    const all = [...hist.train, ...hist.test], ymax = Math.max(0.75, ...all.slice(0, 1));
    const n = Math.max(20, hist.train.length);
    const lx = i => 8 + i / (n - 1) * (LW - 16), ly = v => LH - 18 - Math.min(1, v / ymax) * (LH - 30);
    const line = (arr, col) => { gl.strokeStyle = col; gl.lineWidth = 2; gl.beginPath(); arr.forEach((v, i) => (i ? gl.lineTo(lx(i), ly(v)) : gl.moveTo(lx(i), ly(v)))); gl.stroke(); };
    line(hist.train, '#3987e5'); line(hist.test, '#c98500');
    gl.font = '11px Inter, Arial'; gl.fillStyle = C.dim;
    gl.fillText('log-loss per epoch: train (blue), test (amber)', 8, LH - 4);

    readout.innerHTML =
      `<div>parameters: <b>${nParams()}</b> · epoch <b>${st.epoch}</b></div>` +
      `<div>accuracy: train <b>${Math.round(acc(train) * 100)}%</b> · test <b>${Math.round(acc(test) * 100)}%</b></div>`;
  }

  function reset() { makeData(); makeNet(); draw(); }
  let raf;
  function toggle(force) {
    st.running = force ?? !st.running; playBtn.textContent = st.running ? '⏸ pause' : '▶ train';
    const tick = () => {
      if (!st.running || !canvas.isConnected) return;
      epoch(); draw();
      if (st.epoch < 3000) raf = requestAnimationFrame(tick);
    };
    if (st.running) raf = requestAnimationFrame(tick);
  }
  const playBtn = h('button', { onclick: () => toggle() }, '▶ train');
  const selects = {};
  const sel = (key, opts, fmtv = v => v, after = reset) => {
    const s = selects[key] = h('select', { style: 'font:inherit; background:#272b34; color:#e7e9ee; border:1px solid #3a3f4b; border-radius:6px' },
      opts.map(o => { const e = h('option', { value: o }, String(fmtv(o))); if (o === st[key]) e.selected = true; return e; }));
    s.addEventListener('change', () => { st[key] = typeof st[key] === 'number' ? +s.value : s.value; toggle(false); after(); });
    return s;
  };
  // ready-made scenarios: set parameters, reset, train, and say what to look at
  const SCENARIOS = [
    { name: '1 · it learns', p: { dataset: 'spiral', layers: 2, width: 12, lr: 0.1, batch: 16, noise: 0.1 }, say: 'the boundary forms; train and test accuracy rise together' },
    { name: '2 · too small', p: { dataset: 'spiral', layers: 1, width: 2, lr: 0.1, batch: 16, noise: 0.1 }, say: 'two units = too few joints: it cannot separate the spiral' },
    { name: '3 · full-batch GD', p: { dataset: 'spiral', layers: 2, width: 12, lr: 0.1, batch: 200, noise: 0.1 }, say: 'one step per epoch: far less progress per epoch than SGD with small batches' },
  ];
  const sayEl = h('div', { style: `min-height:1.3em; color:${C.fg}` });
  const scBtns = SCENARIOS.map(sc => {
    const b = h('button', {
      onclick: () => {
        toggle(false);
        Object.assign(st, sc.p); st.seed = 1;
        Object.entries(sc.p).forEach(([k, v]) => { if (selects[k]) selects[k].value = v; });
        scBtns.forEach(x => x.classList.toggle('active', x === b));
        sayEl.innerHTML = `<b style="color:${C.accent}">${sc.name}:</b> ${sc.say}`;
        reset(); toggle(true);
      },
    }, sc.name);
    return b;
  });
  el.classList.add('widget');
  const dot = (col, ring) => h('span', { style: `display:inline-block; width:11px; height:11px; border-radius:50%; vertical-align:middle; margin-right:5px; ${ring ? `border:2px solid ${col}; box-sizing:border-box` : `background:${col}`}` });
  const sw = col => h('span', { style: `display:inline-block; width:16px; height:11px; vertical-align:middle; margin-right:5px; background:${col}; border-radius:2px` });
  const legend = h('div', { style: `display:flex; flex-direction:column; gap:0.1em; margin-top:0.35em; font-size:0.9em; width:${W}px; color:${C.dim}` },
    h('span', {}, dot('#5ab0ff'), 'class 0   ', dot('#f2b134'), 'class 1   · filled: training · ', dot('#9aa1ae', true), 'test'),
    h('span', {}, sw('rgba(57,135,229,0.6)'), sw('rgba(201,133,0,0.6)'), 'background: network\'s prediction (stronger = more confident)'));
  el.append(h('div', { class: 'wctl interactive-only', style: 'gap:0.3em; margin:0 0 0.2em' }, scBtns), sayEl,
    h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start; margin-top:0.3em' }, h('div', {}, canvas, legend),
    h('div', { style: 'display:flex; flex-direction:column; gap:0.45em; width:17em; flex:none' }, readout, lossC,
      h('div', { class: 'wctl interactive-only', style: 'display:grid; grid-template-columns:auto auto; gap:0.3em 0.5em; align-items:center' },
        h('label', {}, 'data'), sel('dataset', Object.keys(DATA)),
        h('label', {}, 'hidden layers'), sel('layers', [1, 2, 3, 4]),
        h('label', {}, 'units / layer'), sel('width', [2, 4, 8, 12, 16, 32]),
        h('label', {}, 'learning rate'), sel('lr', [0.003, 0.01, 0.03, 0.1, 0.3, 1], v => v, () => {}),
        h('label', {}, 'batch size'), sel('batch', [1, 4, 16, 64, 200], v => (v === 200 ? '200 (full GD)' : v), () => {}),
        h('label', {}, 'noise'), sel('noise', [0, 0.1, 0.2, 0.35]),
        playBtn, h('button', { onclick: () => { st.seed++; toggle(false); reset(); } }, 'reset')))));
  sayEl.innerHTML = `<span class="dim">pick a scenario above, or set the parameters by hand</span>`;
  if (cfg.scenario !== undefined) { const sc = SCENARIOS[cfg.scenario]; Object.assign(st, sc.p); Object.entries(sc.p).forEach(([k, v]) => { if (selects[k]) selects[k].value = v; }); scBtns[cfg.scenario].classList.add('active'); sayEl.innerHTML = `<b style="color:${C.accent}">${sc.name}:</b> ${sc.say}`; }
  reset();
  if (isPrint() || cfg.pretrain) { for (let i = 0; i < (cfg.pretrain || 300); i++) epoch(); draw(); }
  else if (cfg.autoplay) setTimeout(() => toggle(true), 500);
}
