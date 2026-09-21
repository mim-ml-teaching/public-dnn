// Learn a convolution kernel from labels only (no target image, no hand design).
// Data: 16×16 noisy images, each with a 5×5 "×" (label 1) or "+" (label 0) at a random position.
// Model: score = max over positions of (x ∗ K),  p(×) = σ(score + b); one k×k kernel K and a bias b, trained by
// full-batch gradient descent with momentum on the cross-entropy (+ a little weight decay). The max is a soft
// maximum τ·log mean exp(f/τ) with τ = 0.2: with a hard max the gradient reaches one position only and some random
// starts get stuck; the soft one trains from every start we tried.
// With k = 5 and seed 1 the kernel becomes the whole × (both diagonals positive, 100% on new images); other starts
// find a shifted part of it (a V, ~98%). With k = 3 it becomes the centre of the × (corners and centre positive,
// edge middles negative, ~97%) or one corner (∧, ∨, ~92%).
// Shows 8 training images with their feature maps (max marked) and p(×), the kernel, the loss curve, and the
// accuracy on 400 new images.
// config: { k: 3, n: 64, noise: 0.25, seed: 1, steps: 0, printSteps: 300 }   (the deck uses k: 5)
import { C, h, fmt, isPrint } from './util.js';
import { LIGHT } from '../theme.js';

const S = 16, SH = 5;
const SHAPES = [
  [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],   // + (label 0)
  [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],   // × (label 1)
];

function rng(seed) {
  let s = (Math.imul(seed ^ 0x9e3779b9, 2654435761) ^ (seed >>> 16)) >>> 0;
  const u = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const n = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  for (let i = 0; i < 4; i++) u();
  return { u, n };
}

function makeSet(n, noise, r) {
  const X = [], Y = [];
  for (let i = 0; i < n; i++) {
    const y = i % 2, x = new Float32Array(S * S);
    for (let q = 0; q < S * S; q++) x[q] = noise * r.n();
    const r0 = Math.floor(r.u() * (S - SH + 1)), c0 = Math.floor(r.u() * (S - SH + 1));
    for (let a = 0; a < SH; a++) for (let b = 0; b < SH; b++) x[(r0 + a) * S + c0 + b] += SHAPES[y][a][b];
    X.push(x); Y.push(y);
  }
  return { X, Y };
}

function fmap(x, K, k) {                            // valid convolution, (S−k+1)²
  const o = S - k + 1, f = new Float32Array(o * o);
  for (let i = 0; i < o; i++) for (let j = 0; j < o; j++) {
    let s = 0;
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += K[a * k + b] * x[(i + a) * S + j + b];
    f[i * o + j] = s;
  }
  return f;
}
const argmax = f => { let m = 0; for (let i = 1; i < f.length; i++) if (f[i] > f[m]) m = i; return m; };
const sig = z => 1 / (1 + Math.exp(-z));
const TAU = 0.2;
function softmax(f) {                               // returns [τ·log mean exp(f/τ), weights]
  const mx = f[argmax(f)], w = new Float64Array(f.length); let z = 0;
  for (let i = 0; i < f.length; i++) z += (w[i] = Math.exp((f[i] - mx) / TAU));
  for (let i = 0; i < f.length; i++) w[i] /= z;
  return [mx + TAU * Math.log(z / f.length), w];
}

export async function mount(el, cfg) {
  el.classList.add('widget');
  const k = cfg.k || 3, n = cfg.n || 64, noise = cfg.noise ?? 0.25, o = S - k + 1;
  const train = makeSet(n, noise, rng(1000 + (cfg.dataSeed || 1)));
  const test = makeSet(400, noise, rng(2000 + (cfg.dataSeed || 1)));
  const show = [0, 2, 4, 6, 1, 3, 5, 7];                         // 4 × "+", then 4 × "×"
  const st = { seed: cfg.seed ?? 1, running: false };
  let K, b, m, t, losses, acc;

  function setup() {
    const r = rng(st.seed);
    K = Float64Array.from({ length: k * k }, () => 0.1 * r.n()); b = 0;
    m = new Float64Array(k * k + 1); t = 0; losses = []; acc = evalTest();
  }
  function evalTest() {
    let c = 0;
    test.X.forEach((x, i) => { c += ((softmax(fmap(x, K, k))[0] + b > 0 ? 1 : 0) === test.Y[i]); });
    return c / test.X.length;
  }
  function step() {
    const g = new Float64Array(k * k + 1);
    let L = 0;
    train.X.forEach((x, i) => {
      const [sc, w] = softmax(fmap(x, K, k)), p = sig(sc + b), y = train.Y[i];
      L -= y ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12);
      const d = (p - y) / n;
      for (let a = 0; a < o * o; a++) {
        if (w[a] < 1e-6) continue;
        const ii = Math.floor(a / o), jj = a % o, dw = d * w[a];
        for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) g[u * k + v] += dw * x[(ii + u) * S + jj + v];
      }
      g[k * k] += d;
    });
    const lr = 0.1, wd = 0.02;
    for (let q = 0; q <= k * k; q++) {
      m[q] = 0.9 * m[q] + g[q] + (q < k * k ? wd * K[q] : 0);
      if (q < k * k) K[q] -= lr * m[q]; else b -= lr * m[q];
    }
    losses.push(L / n); t++;
  }

  // ---- display ----
  const T = 90;                                                  // thumbnail size (css px)
  const canv = (w, hgt, cw, chh) => { const c = h('canvas'); c.width = w; c.height = hgt; c.style.width = cw + 'px'; c.style.height = chh + 'px'; c.style.imageRendering = 'pixelated'; c.style.borderRadius = '3px'; return c; };
  const cells = show.map(i => {
    const ci = canv(S, S, T, T), cf = canv(o, o, T, T), pr = h('div', { class: 'readout', style: `width:${T}px; text-align:center; font-size:0.62em` });
    return { i, ci, cf, pr };
  });
  const kc = h('canvas'); const KS = 190; kc.width = KS; kc.height = KS; kc.style.width = KS + 'px';
  const lc = h('canvas'); lc.width = 260; lc.height = 110; lc.style.width = '260px';
  const info = h('div', { class: 'readout', style: 'width:260px; font-size:0.62em; line-height:1.4' });
  const bgv = LIGHT ? 255 : 30;

  function paintImg(c, x) {
    const g = c.getContext('2d'), im = g.createImageData(S, S);
    for (let q = 0; q < S * S; q++) {
      const v = Math.max(0, Math.min(1, (x[q] + 0.4) / 1.6)), u = LIGHT ? 255 * (1 - v) : 255 * v;
      im.data[4 * q] = im.data[4 * q + 1] = im.data[4 * q + 2] = u; im.data[4 * q + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }
  function paintMap(c, f, mx) {
    const g = c.getContext('2d'), im = g.createImageData(o, o), a = argmax(f);
    for (let q = 0; q < o * o; q++) {
      const v = Math.max(0, Math.min(1, f[q] / mx)), R = [90, 176, 255];      // C.blue, only positive responses
      for (let ch = 0; ch < 3; ch++) im.data[4 * q + ch] = bgv + (R[ch] - bgv) * v;
      im.data[4 * q + 3] = 255;
    }
    g.putImageData(im, 0, 0);
    return a;
  }
  function paintKernel() {
    const g = kc.getContext('2d'), s = KS / k, mx = Math.max(0.05, ...Array.from(K, Math.abs));
    g.clearRect(0, 0, KS, KS);
    for (let a = 0; a < k; a++) for (let c = 0; c < k; c++) {
      const w = K[a * k + c], al = Math.min(1, Math.abs(w) / mx);
      g.fillStyle = w >= 0 ? `rgba(90,176,255,${al})` : `rgba(255,107,107,${al})`;
      g.fillRect(c * s + 1, a * s + 1, s - 2, s - 2);
      g.strokeStyle = C.line; g.strokeRect(c * s + 0.5, a * s + 0.5, s - 1, s - 1);
      g.fillStyle = C.fg; g.font = `${k > 3 ? 13 : 20}px JetBrains Mono, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(fmt(w, 1), c * s + s / 2, a * s + s / 2);
    }
  }
  function paintLoss() {
    const g = lc.getContext('2d'), Wl = lc.width, Hl = lc.height, M = { l: 40, r: 6, t: 6, b: 20 };
    g.clearRect(0, 0, Wl, Hl); g.fillStyle = C.bg2; g.fillRect(0, 0, Wl, Hl);
    g.fillStyle = C.dim; g.font = '12px Inter, Arial'; g.textAlign = 'center'; g.fillText('step', (Wl + M.l) / 2, Hl - 5);
    if (!losses.length) return;
    const hi = Math.max(0.8, ...losses), nn = Math.max(losses.length, 100);
    const X = i => M.l + (Wl - M.l - M.r) * i / (nn - 1), Y = L => M.t + (Hl - M.t - M.b) * (1 - L / hi);
    g.textAlign = 'right'; g.textBaseline = 'middle';
    [0, 0.5].forEach(v => { g.strokeStyle = C.line; g.beginPath(); g.moveTo(M.l, Y(v)); g.lineTo(Wl - M.r, Y(v)); g.stroke(); g.fillText(v.toFixed(1), M.l - 4, Y(v)); });
    g.strokeStyle = C.accent; g.lineWidth = 2; g.beginPath();
    losses.forEach((L, i) => (i ? g.lineTo(X(i), Y(L)) : g.moveTo(X(i), Y(L))));
    g.stroke(); g.lineWidth = 1;
  }
  function render() {
    const maps = cells.map(c => fmap(train.X[c.i], K, k));
    const mx = Math.max(1e-3, ...maps.map(f => f[argmax(f)]));
    cells.forEach((c, q) => {
      paintImg(c.ci, train.X[c.i]);
      const a = paintMap(c.cf, maps[q], mx), p = sig(softmax(maps[q])[0] + b), y = train.Y[c.i];
      const g = c.cf.getContext('2d'); g.strokeStyle = C.accent; g.lineWidth = 0.25; g.strokeRect(a % o + 0.125, Math.floor(a / o) + 0.125, 0.75, 0.75);
      const ok = (p > 0.5 ? 1 : 0) === y;
      c.pr.textContent = p.toFixed(2);
      c.pr.style.color = t === 0 ? C.dim : ok ? C.green : C.red;
    });
    paintKernel(); paintLoss();
    info.innerHTML = `step ${t}   loss ${losses.length ? losses.at(-1).toFixed(3) : '—'}<br>accuracy on new images: ${(100 * acc).toFixed(1)}%`;
  }
  function loop() {
    if (!st.running || !el.isConnected) { st.running = false; btn.textContent = '▶ train'; return; }
    for (let i = 0; i < 2; i++) step();
    if (t % 10 === 0) acc = evalTest();
    render();
    if (t >= 400) { st.running = false; acc = evalTest(); render(); btn.textContent = '▶ train'; return; }
    requestAnimationFrame(loop);
  }
  const btn = h('button', { onclick: () => { st.running = !st.running; btn.textContent = st.running ? '❚❚ pause' : '▶ train'; if (st.running) loop(); } }, '▶ train');
  const reset = h('button', { onclick: () => { st.running = false; setup(); render(); } }, 'reset');
  const reseed = h('button', { onclick: () => { st.running = false; st.seed = Math.floor(Math.random() * 1e6); setup(); render(); } }, 'new random start');

  const lab = (txt, color) => h('div', { class: 'small', style: `width:${T}px; text-align:center; color:${color}; font-size:0.62em` }, txt);
  const col = c => h('div', { style: 'display:flex; flex-direction:column; gap:3px; align-items:center' },
    lab(`label: ${train.Y[c.i] ? '×' : '+'}`, train.Y[c.i] ? C.accent : C.dim), c.ci, c.cf, c.pr);
  const side = t2 => h('div', { class: 'small', style: 'width:5.2em; font-size:0.62em; text-align:right; line-height:1.15' }, t2);
  el.append(
    h('div', { style: 'display:flex; gap:1.4em; align-items:flex-start' },
      h('div', { style: 'display:flex; gap:6px; align-items:flex-start' },
        h('div', { style: `display:flex; flex-direction:column; gap:3px; padding-top:1.1em` },
          h('div', { style: `height:${T}px; display:flex; align-items:center` }, side('image x')),
          h('div', { style: `height:${T}px; display:flex; align-items:center` }, side(h('span', {}, 'feature map x ∗ K', h('br'), h('span', { style: 'color:' + C.accent }, '□'), ' = max'))),
          side('p(×)'),
        ),
        ...cells.map(col)),
      h('div', { style: 'display:flex; flex-direction:column; gap:0.3em' },
        h('div', { class: 'small', style: 'color:' + C.accent }, `kernel K (${k}×${k}, learned)`), kc,
        h('div', { class: 'small', style: 'margin-top:0.2em' }, 'loss (cross-entropy)'), lc, info)),
    h('div', { class: 'wctl interactive-only' }, btn, reset, reseed),
  );
  setup();
  const pre = isPrint() ? (cfg.printSteps ?? 300) : (cfg.steps || 0);
  for (let i = 0; i < pre; i++) step();
  if (pre) acc = evalTest();
  render();
}
