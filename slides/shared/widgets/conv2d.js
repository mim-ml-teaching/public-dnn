// A 2D convolution step by step: input grid, kernel, output grid; the current window is highlighted and the
// dot product written out. In "geometry" mode the numbers are hidden and padding / stride / kernel size /
// input size can be changed; the output-size formula is shown with the current numbers.
// config: {
//   mode: "numbers" | "geometry",
//   input: [[...], ...] (numbers mode), kernel: [[...], ...],
//   n: 5, k: 3, pad: 0, stride: 1 (geometry mode defaults), step: 0 (initial window position), cell: 44
// }
import { C, h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const sgn = v => (v < 0 ? `−${-v}` : `${v}`);

export function mount(el, cfg) {
  const numbers = (cfg.mode || 'numbers') === 'numbers';
  const st = {
    n: numbers ? cfg.input.length : cfg.n ?? 5,
    k: numbers ? cfg.kernel.length : cfg.k ?? 3,
    pad: cfg.pad ?? 0, stride: cfg.stride ?? 1, t: cfg.step ?? 0, timer: null,
  };
  const cs = cfg.cell || (numbers ? 44 : 34);
  el.classList.add('widget');
  const svg = S('svg', { style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const formula = h('div', { class: 'readout', style: 'font-size:1.15em; min-height:1.6em; margin-top:0.2em; color:' + C.fg });
  const ctl = h('div', { class: 'wctl interactive-only' });
  el.append(svg, formula, ctl);

  const geom = () => {
    const N = st.n + 2 * st.pad, out = Math.floor((N - st.k) / st.stride) + 1;
    return { N, out };
  };
  const xval = (r, c) => {               // value in padded coordinates
    const i = r - st.pad, j = c - st.pad;
    if (i < 0 || j < 0 || i >= st.n || j >= st.n) return 0;
    return numbers ? cfg.input[i][j] : null;
  };
  const yval = (i, j) => {
    let s = 0;
    for (let a = 0; a < st.k; a++) for (let b = 0; b < st.k; b++) s += cfg.kernel[a][b] * xval(i * st.stride + a, j * st.stride + b);
    return s;
  };

  function grid(g, x0, y0, rows, cols, cellFn, cs_ = cs) {
    const cs = cs_;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const o = cellFn(r, c);
      g.append(S('rect', { x: x0 + c * cs, y: y0 + r * cs, width: cs, height: cs, fill: o.fill || C.bg2, stroke: o.stroke || C.line,
        'stroke-width': 1, 'stroke-dasharray': o.dash ? '4 3' : 'none' }));
      if (o.text !== undefined && o.text !== null)
        g.append(S('text', { x: x0 + c * cs + cs / 2, y: y0 + r * cs + cs / 2 + 6, fill: o.color || C.fg, 'font-size': Math.round(cs * 0.42),
          'text-anchor': 'middle', 'font-weight': o.bold ? 700 : 400 }, o.text));
    }
  }

  function draw() {
    const { N, out } = geom();
    const T = out * out;
    st.t = ((st.t % T) + T) % T;
    const ci = Math.floor(st.t / out), cj = st.t % out;
    const wr = ci * st.stride, wc = cj * st.stride;
    svg.replaceChildren();
    const g = S('g');
    svg.append(g);
    if (numbers) return drawNumbers(g, out, ci, cj, wr, wc);
    const pad = 30, top = 34;
    const inW = N * cs, kW = st.k * cs, outW = out * cs;
    const kx = pad + inW + 70, ox = kx + (numbers ? kW : 0) + (numbers ? 120 : 30);
    const W = ox + outW + pad, H = top + Math.max(inW, numbers ? kW + 150 : 0, outW) + 16;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const label = (x, text, color = C.dim) => g.append(S('text', { x, y: 22, fill: color, 'font-size': 17 }, text));

    // input (with padding ring)
    label(pad, numbers ? `input ${st.n}×${st.n}` : `input ${st.n}×${st.n}${st.pad ? `, padding ${st.pad}` : ''}`);
    grid(g, pad, top, N, N, (r, c) => {
      const inPad = r < st.pad || c < st.pad || r >= N - st.pad || c >= N - st.pad;
      const inWin = r >= wr && r < wr + st.k && c >= wc && c < wc + st.k;
      return {
        fill: inWin ? 'rgba(242,177,52,0.22)' : inPad ? C.bg : C.bg2, stroke: inPad ? '#4a505c' : C.line, dash: inPad,
        text: numbers ? xval(r, c) : null, color: inPad ? C.dim : C.fg,
      };
    });
    g.append(S('rect', { x: pad + wc * cs, y: top + wr * cs, width: kW, height: kW, fill: 'none', stroke: C.accent, 'stroke-width': 3, rx: 3 }));

    // kernel
    if (numbers) {
      label(kx, `kernel ${st.k}×${st.k}`, C.accent);
      grid(g, kx, top, st.k, st.k, (r, c) => ({ fill: 'rgba(242,177,52,0.22)', stroke: C.accent, text: sgn(cfg.kernel[r][c]), bold: true }));
    }

    // output
    label(ox, `output ${out}×${out}`, C.blue);
    grid(g, ox, top, out, out, (i, j) => {
      const idx = i * out + j, cur = idx === st.t;
      return { fill: cur ? 'rgba(90,176,255,0.25)' : C.bg2, stroke: cur ? C.blue : C.line, text: numbers && idx <= st.t ? sgn(yval(i, j)) : null, bold: cur };
    });
    g.append(S('rect', { x: ox + cj * cs, y: top + ci * cs, width: cs, height: cs, fill: 'none', stroke: C.blue, 'stroke-width': 3, rx: 3 }));

    // connector lines window → output cell (geometry mode; in numbers mode they would cross the kernel)
    if (!numbers) {
    const ax = pad + (wc + st.k) * cs, ay = top + wr * cs;
    g.append(S('line', { x1: ax, y1: ay, x2: ox + cj * cs, y2: top + ci * cs, stroke: C.accent, 'stroke-width': 1.2, opacity: 0.6 }));
    g.append(S('line', { x1: ax, y1: ay + kW, x2: ox + cj * cs, y2: top + (ci + 1) * cs, stroke: C.accent, 'stroke-width': 1.2, opacity: 0.6 }));
    }

    if (numbers) {
      // the dot product, one kernel row per line
      const ty = top + kW + 36;
      for (let a = 0; a < st.k; a++) {
        const terms = [];
        for (let b = 0; b < st.k; b++) terms.push(`${sgn(xval(wr + a, wc + b))}·${cfg.kernel[a][b] < 0 ? `(${sgn(cfg.kernel[a][b])})` : cfg.kernel[a][b]}`);
        g.append(S('text', { x: kx - 30, y: ty + a * 30, fill: C.fg, 'font-size': 20, 'font-family': 'JetBrains Mono, monospace' }, (a ? '+ ' : '  ') + terms.join(' + ')));
      }
      g.append(S('text', { x: kx - 30, y: ty + st.k * 30 + 8, fill: C.blue, 'font-size': 24, 'font-weight': 700, 'font-family': 'JetBrains Mono, monospace' }, `= ${sgn(yval(ci, cj))}`));
      formula.innerHTML = '';
    } else {
      formula.textContent = `output: ⌊(${st.n} + 2·${st.pad} − ${st.k}) / ${st.stride}⌋ + 1 = ${out}`;
    }
  }


  // numbers mode: input → the window lifted out ⊙ kernel = nine products, their sum → one output cell
  function drawNumbers(g, out, ci, cj, wr, wc) {
    const k = st.k, sc = Math.round(cs * 0.76), pad = 30, top = 34, gap = 44;
    const inW = st.n * cs, x1 = pad + inW + 50, x2 = x1 + k * sc + gap, x3 = x2 + k * sc + gap, ox = x3 + k * sc + 64;
    const W = ox + out * sc + pad, H = top + Math.max(inW, k * sc + 110, out * sc) + 16;
    svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const label = (x, text, color = C.dim) => g.append(S('text', { x, y: 22, fill: color, 'font-size': 17 }, text));
    const sym = (x, text) => g.append(S('text', { x, y: top + k * sc / 2 + 10, fill: C.fg, 'font-size': 30, 'text-anchor': 'middle' }, text));
    const amber = 'rgba(242,177,52,0.22)';

    label(pad, `input ${st.n}×${st.n}`);
    grid(g, pad, top, st.n, st.n, (r, c) => {
      const inWin = r >= wr && r < wr + k && c >= wc && c < wc + k;
      return { fill: inWin ? amber : C.bg2, text: xval(r, c) };
    });
    g.append(S('rect', { x: pad + wc * cs, y: top + wr * cs, width: k * cs, height: k * cs, fill: 'none', stroke: C.accent, 'stroke-width': 3, rx: 3 }));
    // zoom lines from the window to its copy
    [[0, 0], [k, k]].forEach(([a, b]) => g.append(S('line', { x1: pad + (wc + k) * cs, y1: top + (wr + a) * cs, x2: x1, y2: top + b * sc,
      stroke: C.accent, 'stroke-width': 1.5, 'stroke-dasharray': '5 4', opacity: 0.8 })));

    label(x1, 'window');
    grid(g, x1, top, k, k, (r, c) => ({ fill: amber, stroke: C.accent, text: xval(wr + r, wc + c) }), sc);
    sym(x2 - gap / 2, '⊙');
    label(x2, 'kernel', C.accent);
    grid(g, x2, top, k, k, (r, c) => ({ fill: amber, stroke: C.accent, text: sgn(cfg.kernel[r][c]), bold: true }), sc);
    sym(x3 - gap / 2, '=');
    label(x3, 'products');
    grid(g, x3, top, k, k, (r, c) => ({ fill: C.bg2, text: sgn(xval(wr + r, wc + c) * cfg.kernel[r][c]) }), sc);

    // sum → output cell
    const y = yval(ci, cj), sy = top + k * sc + 44;
    g.append(S('text', { x: x1, y: sy, fill: C.fg, 'font-size': 20 }, 'multiply cell by cell, then add all nine:'));
    g.append(S('text', { x: x3 + 10, y: sy + 44, fill: C.blue, 'font-size': 26, 'font-weight': 700, 'font-family': 'JetBrains Mono, monospace' },
      `Σ = ${sgn(y)}`));
    label(ox, `output ${out}×${out}`, C.blue);
    grid(g, ox, top, out, out, (i, j) => {
      const idx = i * out + j, cur = idx === st.t;
      return { fill: cur ? 'rgba(90,176,255,0.25)' : C.bg2, text: idx <= st.t ? sgn(yval(i, j)) : null, bold: cur };
    }, sc);
    g.append(S('rect', { x: ox + cj * sc, y: top + ci * sc, width: sc, height: sc, fill: 'none', stroke: C.blue, 'stroke-width': 3, rx: 3 }));
    g.append(S('line', { x1: x3 + k * sc + 6, y1: sy + 34, x2: ox + cj * sc + sc / 2, y2: top + (ci + 1) * sc + 3,
      stroke: C.blue, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }));
    formula.innerHTML = '';
  }

  const stop = () => { clearInterval(st.timer); st.timer = null; play.textContent = '▶ play'; };
  const prev = h('button', { onclick: () => { stop(); st.t--; draw(); } }, '◀');
  const next = h('button', { onclick: () => { stop(); st.t++; draw(); } }, '▶');
  const play = h('button', {
    onclick: () => {
      if (st.timer) return stop();
      play.textContent = '❚❚ pause';
      st.timer = setInterval(() => { if (!el.isConnected) return stop(); st.t++; draw(); }, numbers ? 900 : 600);
    },
  }, '▶ play');
  ctl.append(prev, next, play);
  if (!numbers) {
    const group = (name, key, vals) => {
      const bs = vals.map(v => h('button', { onclick: () => { st[key] = v; st.t = 0; bs.forEach((b, i) => b.classList.toggle('active', vals[i] === v)); draw(); } }, String(v)));
      bs.forEach((b, i) => b.classList.toggle('active', vals[i] === st[key]));
      ctl.append(h('label', { style: 'margin-left:0.8em' }, name), ...bs);
    };
    group('n', 'n', [5, 6, 7]);
    group('kernel k', 'k', [3, 5]);
    group('padding p', 'pad', [0, 1, 2]);
    group('stride s', 'stride', [1, 2]);
  }
  draw();
}
