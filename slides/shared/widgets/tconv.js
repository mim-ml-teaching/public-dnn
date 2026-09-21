// Transposed convolution ("learnable upsampling").
// mode "numbers": every input pixel paints a copy of the kernel, scaled by its value, into the output; the copies
//   are placed `stride` apart and summed where they overlap. Step through the input pixels one at a time.
// mode "overlap": an input of ones; the number in each output pixel = how many kernel copies cover it (with an
//   all-ones kernel this is exactly the output). Uneven counts = the checkerboard pattern. Buttons change k and s,
//   and "resize + conv" shows the alternative (nearest-neighbour upsampling, then a k×k convolution with stride 1).
// mode "geometry": no numbers; one input pixel at a time is highlighted together with the k×k block of the output
//   its kernel copy lands on (only the current copy is shown, so the overlap counts stay for the viewer to work out).
//   ◀ / ▶ / play, and buttons for k and s (as the geometry mode of conv2d in lecture 6).
// mode "dilated": the same transposed convolution as an ordinary one: spread the input pixels s apart (s − 1 zeros
//   between them), pad with k − 1 zeros, slide the kernel rotated by 180° with stride 1. Step through the output pixels;
//   the readout lists the non-zero products (1, 2 or 4 real pixels under the window for k = 3, s = 2).
// mini (numbers mode): no readout line.
// config: { mode: "numbers" | "overlap" | "geometry" | "dilated", input: [[..]], kernel: [[..]], stride: 2, step: 4 (numbers: 0..n², n² = all),
//           n: 5, k: 3 (overlap, geometry), cell: 46, size: 300 (geometry: side of the output grid in px) }
import { C, h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const sgn = v => (v < 0 ? `−${-v}` : `${v}`);
const PAL = [C.red, C.green, C.blue, C.purple, C.accent, '#7d8594', '#ff6b6b', '#5fd38d', '#5ab0ff'];
const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };

export function mount(el, cfg) {
  el.classList.add('widget');
  const svg = S('svg', { style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const info = h('div', { class: 'readout', style: `font-size:1.1em; min-height:1.5em; margin-top:0.2em; color:${C.fg}` });
  const ctl = h('div', { class: 'wctl interactive-only' });
  el.append(svg, info, ctl);
  if (cfg.mini) info.style.display = 'none';
  ({ overlap, geometry, dilated }[cfg.mode] || numbers)(cfg, svg, info, ctl, el);
}

function grid(g, x0, y0, rows, cols, cs, fn) {
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const o = fn(r, c);
    g.append(S('rect', { x: x0 + c * cs, y: y0 + r * cs, width: cs, height: cs, fill: o.fill || C.bg2, stroke: o.stroke || C.line, 'stroke-width': o.sw || 1 }));
    if (o.text !== undefined && o.text !== null)
      g.append(S('text', { x: x0 + c * cs + cs / 2, y: y0 + r * cs + cs / 2 + cs * 0.15, fill: o.color || C.fg, 'font-size': Math.round(cs * (o.fs || 0.4)),
        'text-anchor': 'middle', 'font-weight': o.bold ? 700 : 400 }, o.text));
  }
}

function numbers(cfg, svg, info, ctl) {
  const X = cfg.input, K = cfg.kernel, s = cfg.stride || 2, n = X.length, k = K.length, N = (n - 1) * s + k;
  const cs = cfg.cell || 46, kc = Math.round(cs * 0.8);
  let t = cfg.step ?? n * n;                      // number of input pixels already painted
  function draw() {
    svg.replaceChildren();
    const g = S('g'); svg.append(g);
    const cur = t < n * n ? t : -1;
    const top = 34, x0 = 20, xv = x0 + n * cs + 70, xk = xv + 76, xc = xk + k * kc + 56, xo = xc + k * kc + 80;
    const W = xo + N * cs + 20, H = top + Math.max(N * cs, n * cs + 40, k * kc + 50) + 10;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const fl = Math.min(17, Math.round(cs * 0.5));
    const label = (x, y, text, color = C.dim) => g.append(S('text', { x, y, fill: color, 'font-size': fl }, text));
    const sym = (x, text, color = C.fg) => g.append(S('text', { x, y: top + k * kc / 2 + 10, fill: color, 'font-size': Math.min(30, Math.round(cs * 0.75)), 'text-anchor': 'middle', 'font-weight': 600 }, text));
    label(x0, 22, `input ${n}×${n}`);
    grid(g, x0, top, n, n, cs, (r, c) => {
      const i = r * n + c, col = PAL[i % PAL.length];
      return { fill: i < t || i === cur ? rgba(col, i === cur ? 0.45 : 0.25) : C.bg2, stroke: i === cur ? col : C.line, sw: i === cur ? 3 : 1, text: sgn(X[r][c]), bold: i === cur };
    });
    // the current pixel × kernel = its scaled copy
    const cr = Math.floor(cur / n), cc = cur % n, ccol = cur >= 0 ? PAL[cur % PAL.length] : C.dim, v = cur >= 0 ? X[cr][cc] : null;
    label(xk, 22, `kernel ${k}×${k}`, C.accent);
    grid(g, xk, top, k, k, kc, (r, c) => ({ fill: 'rgba(242,177,52,0.22)', stroke: C.accent, text: sgn(K[r][c]), bold: true }));
    sym(xc - 28, '=');
    if (cur >= 0) {
      sym(xv + 14, sgn(v), ccol); sym(xk - 26, '×');
      label(xc, 22, `copy = ${sgn(v)} × kernel`, ccol);
      grid(g, xc, top, k, k, kc, (r, c) => ({ fill: rgba(ccol, 0.3), stroke: ccol, text: sgn(v * K[r][c]), bold: true }));
      g.append(S('line', { x1: x0 + (cc + 1) * cs + 4, y1: top + (cr + 0.5) * cs, x2: xv - 4, y2: top + k * kc / 2, stroke: ccol, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }));
    } else {
      sym(xv + 14, 'x', C.dim); sym(xk - 26, '×');
      label(xc, 22, 'copy = x × kernel');
      grid(g, xc, top, k, k, kc, () => ({ fill: C.bg2 }));
    }
    label(xk, top + k * kc + 32, cfg.mini ? `stride ${s}` : `stride ${s}: copies ${s} apart in the output`);
    // output: sum of the painted copies; in the current block, overlapping cells show previous + new
    const Y = Array.from({ length: N }, () => new Array(N).fill(0)), cnt = Array.from({ length: N }, () => new Array(N).fill(0));
    const owner = Array.from({ length: N }, () => new Array(N).fill(-1));
    const upto = cur >= 0 ? cur : n * n;
    for (let i = 0; i < upto; i++) {
      const r = Math.floor(i / n), c = i % n;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
        Y[r * s + a][c * s + b] += X[r][c] * K[a][b]; cnt[r * s + a][c * s + b]++; owner[r * s + a][c * s + b] = i;
      }
    }
    const inCur = (r, c) => cur >= 0 && r >= cr * s && r < cr * s + k && c >= cc * s && c < cc * s + k;
    label(xo, 22, cfg.mini ? `output ${N}×${N}` : `output ${N}×${N} = (${n}−1)·${s} + ${k}`, C.blue);
    grid(g, xo, top, N, N, cs, (r, c) => {
      if (inCur(r, c)) {
        const add = v * K[r - cr * s][c - cc * s];
        if (!cnt[r][c]) return { fill: rgba(ccol, 0.3), text: sgn(add), bold: true };
        return { fill: 'rgba(255,255,255,0.16)' };                 // text drawn below: previous + new
      }
      if (!cnt[r][c]) return { fill: C.bg2 };
      const col = cnt[r][c] > 1 ? C.fg : PAL[owner[r][c] % PAL.length];
      return { fill: cnt[r][c] > 1 ? 'rgba(255,255,255,0.16)' : rgba(col, 0.25), text: sgn(Y[r][c]), bold: cnt[r][c] > 1 };
    });
    let example = null;
    if (cur >= 0) {
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
        const r = cr * s + a, c = cc * s + b;
        if (!cnt[r][c]) continue;
        const add = v * K[a][b], tx = S('text', { x: xo + c * cs + cs / 2, y: top + r * cs + cs / 2 + cs * 0.12, 'font-size': Math.round(cs * 0.32), 'text-anchor': 'middle', 'font-weight': 700, fill: C.fg });
        tx.append(S('tspan', {}, `${sgn(Y[r][c])}+`), S('tspan', { fill: ccol }, sgn(add)));
        g.append(tx);
        example ??= `${sgn(Y[r][c])} + ${sgn(add)} = ${sgn(Y[r][c] + add)}`;
      }
      g.append(S('rect', { x: xo + cc * s * cs, y: top + cr * s * cs, width: k * cs, height: k * cs, fill: 'none', stroke: ccol, 'stroke-width': 3.5, rx: 3 }));
      info.textContent = example ? `overlapping cells are summed: ${example}` : `the copy goes to its ${k}×${k} block of the output`;
    } else info.textContent = 'all copies added; bright cells: two or more copies overlapped and were summed';
  }
  const prev = h('button', { onclick: () => { t = Math.max(0, t - 1); draw(); } }, '◀');
  const next = h('button', { onclick: () => { t = Math.min(n * n, t + 1); draw(); } }, '▶');
  const reset = h('button', { onclick: () => { t = 0; draw(); } }, 'start');
  const all = h('button', { onclick: () => { t = n * n; draw(); } }, 'all');
  ctl.append(reset, prev, next, all);
  draw();
}

function overlap(cfg, svg, info, ctl) {
  const st = { n: cfg.n || 5, k: cfg.k || 3, s: cfg.stride || 2, resize: !!cfg.resize };
  const cs = cfg.cell || 30;
  function draw() {
    const { n, k, s } = st;
    let N, cnt;
    if (!st.resize) {
      N = (n - 1) * s + k;
      cnt = Array.from({ length: N }, () => new Array(N).fill(0));
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) cnt[r * s + a][c * s + b]++;
    } else {
      // nearest-neighbour ×s, then k×k convolution, stride 1, "same" padding: count of input-1 pixels under each window
      const M = n * s; N = M; const p = (k - 1) >> 1;
      cnt = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => {
        let v = 0;
        for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const y = i + a - p, x = j + b - p; if (y >= 0 && x >= 0 && y < M && x < M) v++; }
        return v;
      }));
    }
    const mx = Math.max(...cnt.flat());
    svg.replaceChildren();
    const g = S('g'); svg.append(g);
    const W = N * cs + 40, H = N * cs + 44;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    g.append(S('text', { x: 20, y: 22, fill: C.dim, 'font-size': 16 },
      st.resize ? `input of ones ${n}×${n} → resize ×${s} → conv ${k}×${k}` : `input of ones ${n}×${n}, kernel of ones ${k}×${k}, stride ${s}`));
    // interior: the part not affected by the border
    grid(g, 20, 34, N, N, cs, (r, c) => {
      const v = cnt[r][c], a = 0.08 + 0.62 * v / mx;
      return { fill: `rgba(90,176,255,${a.toFixed(2)})`, text: String(v), fs: 0.42, color: C.fg };
    });
    const inner = st.resize ? [] : (() => { const lo = k - 1, hi = N - k; const vals = new Set(); for (let r = lo; r <= hi; r++) for (let c = lo; c <= hi; c++) vals.add(cnt[r][c]); return [...vals].sort((a, b) => a - b); })();
    const innerR = st.resize ? (() => { const lo = k, hi = N - k - 1; const vals = new Set(); for (let r = lo; r <= hi; r++) for (let c = lo; c <= hi; c++) vals.add(cnt[r][c]); return [...vals]; })() : inner;
    info.textContent = innerR.length > 1 ? `inside: ${innerR.join(' / ')} copies, a checkerboard` : `inside: always ${innerR[0]} copies, even`;
  }
  const group = (name, key, vals, fmt = String) => {
    const bs = vals.map(v => h('button', { onclick: () => { st[key] = v; bs.forEach((b, i) => b.classList.toggle('active', vals[i] === v)); draw(); } }, fmt(v)));
    bs.forEach((b, i) => b.classList.toggle('active', vals[i] === st[key]));
    ctl.append(h('label', { style: 'margin-left:0.6em' }, name), ...bs);
  };
  group('kernel k', 'k', [2, 3, 4]);
  group('stride s', 's', [1, 2]);
  group('', 'resize', [false, true], v => (v ? 'resize + conv' : 'transposed conv'));
  draw();
}

function geometry(cfg, svg, info, ctl, el) {
  const st = { n: cfg.n || 4, k: cfg.k || 3, s: cfg.stride || 2, t: cfg.step || 0, timer: null };
  const size = cfg.size || 300;
  function draw() {
    const { n, k, s } = st, N = (n - 1) * s + k, cs = Math.floor(size / N);
    st.t = ((st.t % (n * n)) + n * n) % (n * n);
    const r = Math.floor(st.t / n), c = st.t % n;
    svg.replaceChildren();
    const g = S('g'); svg.append(g);
    const top = 34, x0 = 10, xo = x0 + n * cs + 90;
    const W = xo + N * cs + 10, H = top + N * cs + 6;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const label = (x, text, color = C.dim) => g.append(S('text', { x, y: 22, fill: color, 'font-size': 17 }, text));
    const amber = 'rgba(242,177,52,0.3)';
    label(x0, `input ${n}×${n}`);
    grid(g, x0, top, n, n, cs, (i, j) => ({ fill: i === r && j === c ? amber : C.bg2 }));
    g.append(S('rect', { x: x0 + c * cs, y: top + r * cs, width: cs, height: cs, fill: 'none', stroke: C.accent, 'stroke-width': 3, rx: 2 }));
    label(xo, `output ${N}×${N}`, C.blue);
    grid(g, xo, top, N, N, cs, (i, j) => ({ fill: i >= r * s && i < r * s + k && j >= c * s && j < c * s + k ? amber : C.bg2 }));
    g.append(S('rect', { x: xo + c * s * cs, y: top + r * s * cs, width: k * cs, height: k * cs, fill: 'none', stroke: C.accent, 'stroke-width': 3, rx: 3 }));
    for (const [a, b] of [[0, 0], [1, k]])
      g.append(S('line', { x1: x0 + (c + 1) * cs, y1: top + (r + a) * cs, x2: xo + c * s * cs, y2: top + (r * s + b) * cs,
        stroke: C.accent, 'stroke-width': 1.5, 'stroke-dasharray': '5 4', opacity: 0.8 }));
    info.textContent = `k = ${k}, s = ${s}: copies ${s} apart`;
  }
  const stop = () => { clearInterval(st.timer); st.timer = null; play.textContent = '▶ play'; };
  const prev = h('button', { onclick: () => { stop(); st.t--; draw(); } }, '◀');
  const next = h('button', { onclick: () => { stop(); st.t++; draw(); } }, '▶');
  const play = h('button', {
    onclick: () => {
      if (st.timer) return stop();
      play.textContent = '❚❚ pause';
      st.timer = setInterval(() => { if (!el.isConnected) return stop(); st.t++; draw(); }, 700);
    },
  }, '▶ play');
  ctl.append(prev, next, play);
  const group = (name, key, vals) => {
    const bs = vals.map(v => h('button', { onclick: () => { st[key] = v; st.t = 0; bs.forEach((b, i) => b.classList.toggle('active', vals[i] === v)); draw(); } }, String(v)));
    bs.forEach((b, i) => b.classList.toggle('active', vals[i] === st[key]));
    ctl.append(h('label', { style: 'margin-left:0.8em' }, name), ...bs);
  };
  group('k', 'k', [2, 3, 4]);
  group('s', 's', [1, 2]);
  draw();
}

function dilated(cfg, svg, info, ctl, el) {
  const X = cfg.input, K = cfg.kernel, s = cfg.stride || 2, n = X.length, k = K.length;
  const D = (n - 1) * s + 1, P = D + 2 * (k - 1), N = P - k + 1;
  const cs = cfg.cell || 34, kc = Math.round(cs * 0.8);
  const st = { t: cfg.step || 0, timer: null };
  const Kf = K.map((row, a) => row.map((_, b) => K[k - 1 - a][k - 1 - b]));      // rotated by 180°
  // padded, spread input: value, and which input pixel it is (-1: inserted zero, -2: padding)
  const cell = (r, c) => {
    const i = r - (k - 1), j = c - (k - 1);
    if (i < 0 || j < 0 || i >= D || j >= D) return { v: 0, src: -2 };
    if (i % s || j % s) return { v: 0, src: -1 };
    return { v: X[i / s][j / s], src: (i / s) * n + j / s };
  };
  const yval = (i, j) => { let y = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) y += cell(i + a, j + b).v * Kf[a][b]; return y; };
  function draw() {
    st.t = ((st.t % (N * N)) + N * N) % (N * N);
    const ci = Math.floor(st.t / N), cj = st.t % N;
    svg.replaceChildren();
    const g = S('g'); svg.append(g);
    const fl = Math.min(17, Math.round(cs * 0.5)), top = 34, x0 = 10, xk = x0 + P * cs + 40, xo = xk + Math.max(k * kc, fl * 8) + 40;
    const W = xo + N * cs + 10, H = top + Math.max(P * cs, N * cs) + 6;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const label = (x, y, text, color = C.dim) => g.append(S('text', { x, y, fill: color, 'font-size': fl }, text));
    label(x0, 22, `input spread out + padded: ${P}×${P}`);
    for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) {
      const { v, src } = cell(r, c), inWin = r >= ci && r < ci + k && c >= cj && c < cj + k;
      const fill = src >= 0 ? rgba(PAL[src % PAL.length], inWin ? 0.5 : 0.3) : inWin ? 'rgba(242,177,52,0.18)' : src === -1 ? C.bg2 : C.bg;
      g.append(S('rect', { x: x0 + c * cs, y: top + r * cs, width: cs, height: cs, fill, stroke: src === -2 ? '#4a505c' : C.line, 'stroke-dasharray': src === -2 ? '4 3' : 'none' }));
      g.append(S('text', { x: x0 + c * cs + cs / 2, y: top + r * cs + cs / 2 + cs * 0.15, fill: src >= 0 ? C.fg : C.dim, 'font-size': Math.round(cs * 0.4),
        'text-anchor': 'middle', 'font-weight': src >= 0 ? 700 : 400, opacity: src >= 0 ? 1 : 0.6 }, sgn(v)));
    }
    g.append(S('rect', { x: x0 + cj * cs, y: top + ci * cs, width: k * cs, height: k * cs, fill: 'none', stroke: C.accent, 'stroke-width': 3, rx: 3 }));
    label(xk, 22, 'kernel rotated 180°', C.accent);
    grid(g, xk, top, k, k, kc, (r, c) => ({ fill: 'rgba(242,177,52,0.22)', stroke: C.accent, text: sgn(Kf[r][c]), bold: true }));
    // the kernel weights that meet a real pixel: outlined in that pixel's colour (window cell and kernel cell)
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
      const { src } = cell(ci + a, cj + b);
      if (src < 0) continue;
      const col = PAL[src % PAL.length];
      g.append(S('rect', { x: xk + b * kc + 2, y: top + a * kc + 2, width: kc - 4, height: kc - 4, fill: 'none', stroke: col, 'stroke-width': 3.5, rx: 2 }));
      g.append(S('rect', { x: x0 + (cj + b) * cs + 2, y: top + (ci + a) * cs + 2, width: cs - 4, height: cs - 4, fill: 'none', stroke: col, 'stroke-width': 3, rx: 2 }));
    }
    label(xk, top + k * kc + 28, 'stride 1');
    label(xo, 22, `output ${N}×${N}`, C.blue);
    grid(g, xo, top, N, N, cs, (i, j) => {
      const idx = i * N + j, cur = idx === st.t;
      return { fill: cur ? 'rgba(90,176,255,0.25)' : C.bg2, text: idx <= st.t ? sgn(yval(i, j)) : null, bold: cur };
    });
    g.append(S('rect', { x: xo + cj * cs, y: top + ci * cs, width: cs, height: cs, fill: 'none', stroke: C.blue, 'stroke-width': 3, rx: 3 }));
    const terms = [];
    for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
      const { v, src } = cell(ci + a, cj + b);
      if (src >= 0 && v) terms.push(`<span style="color:${PAL[src % PAL.length]}; font-weight:700">${sgn(v)}·${sgn(Kf[a][b])}</span>`);
    }
    info.innerHTML = `${terms.length} real pixel${terms.length === 1 ? '' : 's'} under the window: ${terms.join(' + ')} = <b>${sgn(yval(ci, cj))}</b> <span style="color:${C.dim}">(the zeros contribute nothing)</span>`;
  }
  const stop = () => { clearInterval(st.timer); st.timer = null; play.textContent = '▶ play'; };
  const prev = h('button', { onclick: () => { stop(); st.t--; draw(); } }, '◀');
  const next = h('button', { onclick: () => { stop(); st.t++; draw(); } }, '▶');
  const play = h('button', {
    onclick: () => {
      if (st.timer) return stop();
      play.textContent = '❚❚ pause';
      st.timer = setInterval(() => { if (!el.isConnected) return stop(); st.t++; draw(); }, 900);
    },
  }, '▶ play');
  ctl.append(prev, next, play);
  draw();
}
