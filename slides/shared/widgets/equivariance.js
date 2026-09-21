// Translation equivariance: shift the input, and the convolution output shifts by the same amount; the output
// of a dense layer (a fixed random matrix, same input and output size) changes completely.
// config: { n: 20, cell: 11, dx: 0, dy: 0 }
import { C, h } from './util.js';

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

// a small asymmetric shape (the digit 7 with a bar), drawn at offset (ox, oy)
const SHAPE = ['#######', '     ##', '    ## ', '  #####', '   ##  ', '  ##   ', '  ##   '];

export function mount(el, cfg) {
  el.classList.add('widget');
  const n = cfg.n || 20, cs = cfg.cell || 11, st = { dx: cfg.dx || 0, dy: cfg.dy || 0, bg: cfg.bg || 0, timer: null };
  const K = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]];   // responds to horizontal edges (top: −, bottom: +)
  const rand = rng(3), M = Array.from({ length: n * n }, () => Float32Array.from({ length: n * n }, () => rand() - 0.5));
  const base = { x: Math.floor((n - SHAPE[0].length) / 2), y: Math.floor((n - SHAPE.length) / 2) };
  const lim = Math.min(base.x, base.y) + 3;   // far enough for the shape to touch and leave the border

  const input = () => {
    const x = new Float32Array(n * n).fill(st.bg);
    SHAPE.forEach((row, r) => [...row].forEach((ch, c) => {
      const i = base.y + st.dy + r, j = base.x + st.dx + c;
      if (ch === '#' && i >= 0 && j >= 0 && i < n && j < n) x[i * n + j] = 1;
    }));
    return x;
  };
  const conv = x => {
    const y = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
        const ii = i + a, jj = j + b;
        if (ii >= 0 && jj >= 0 && ii < n && jj < n) s += K[a + 1][b + 1] * x[ii * n + jj];
      }
      y[i * n + j] = s;
    }
    return y;
  };
  const dense = x => {
    const y = new Float32Array(n * n);
    for (let o = 0; o < n * n; o++) { let s = 0; const row = M[o]; for (let i = 0; i < n * n; i++) if (x[i]) s += row[i] * x[i]; y[o] = s; }
    return y;
  };
  const mk = () => { const c = h('canvas'); c.width = n * cs; c.height = n * cs; return c; };
  const cIn = mk(), cConv = mk(), cDense = mk();
  function paint(c, v, signed) {
    const g = c.getContext('2d'), mx = Math.max(1e-6, ...Array.from(v, Math.abs));
    g.fillStyle = C.bg2; g.fillRect(0, 0, c.width, c.height);
    for (let i = 0; i < n * n; i++) {
      const val = v[i] / mx;
      if (!val) continue;
      g.fillStyle = !signed ? `rgba(231,233,238,${val})` : val > 0 ? `rgba(90,176,255,${val})` : `rgba(255,107,107,${-val})`;
      g.fillRect((i % n) * cs, Math.floor(i / n) * cs, cs, cs);
    }
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    for (let k = 0; k <= n; k++) { g.beginPath(); g.moveTo(k * cs, 0); g.lineTo(k * cs, n * cs); g.stroke(); g.beginPath(); g.moveTo(0, k * cs); g.lineTo(n * cs, k * cs); g.stroke(); }
  }
  const readout = h('div', { class: 'readout', style: 'margin-top:0.3em' });
  function draw() {
    const x = input();
    paint(cIn, x, false); paint(cConv, conv(x), true); paint(cDense, dense(x), true);
    readout.textContent = `input shifted by (${st.dx}, ${st.dy})${st.bg ? ', grey background' : ''}`;
  }
  const move = (ddx, ddy) => { st.dx = Math.max(-lim, Math.min(lim, st.dx + ddx)); st.dy = Math.max(-lim, Math.min(lim, st.dy + ddy)); draw(); };
  const stop = () => { clearInterval(st.timer); st.timer = null; anim.textContent = '▶ animate'; };
  const anim = h('button', {
    onclick: () => {
      if (st.timer) return stop();
      anim.textContent = '❚❚ stop';
      let t = 0;
      st.timer = setInterval(() => { if (!el.isConnected) return stop(); t++; st.dx = Math.round(lim * Math.sin(t / 6)); st.dy = Math.round(lim * 0.6 * Math.sin(t / 9)); draw(); }, 120);
    },
  }, '▶ animate');
  const cap = (t, sub, c, color) => h('div', {}, h('div', { style: `color:${color}; font-weight:600` }, t), h('div', { class: 'small', style: 'min-height:1.3em' }, sub), c);
  el.append(
    h('div', { style: 'display:flex; gap:2em' },
      cap('input', '', cIn, C.fg),
      cap('convolution output', 'horizontal-edge kernel 3×3', cConv, C.blue),
      cap('dense layer output', `random ${n * n}×${n * n} matrix`, cDense, C.purple)),
    readout,
    h('div', { class: 'wctl interactive-only' }, h('label', {}, 'shift the input:'),
      h('button', { onclick: () => move(-1, 0) }, '←'), h('button', { onclick: () => move(1, 0) }, '→'),
      h('button', { onclick: () => move(0, -1) }, '↑'), h('button', { onclick: () => move(0, 1) }, '↓'), anim,
      h('button', { style: 'margin-left:1em', onclick: e => { st.bg = st.bg ? 0 : 0.5; e.target.textContent = st.bg ? 'black background' : 'grey background'; draw(); } },
        st.bg ? 'black background' : 'grey background')),
  );
  draw();
}
