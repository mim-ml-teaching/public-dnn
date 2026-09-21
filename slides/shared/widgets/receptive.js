// Receptive field: which input pixels can influence one output pixel after L stacked convolutions?
// Grids for the input and every layer, side by side; the chosen output pixel and its region of influence in
// every earlier layer are highlighted (amber). Coloured outlines and lines: one pixel and the window it is
// computed from in the previous layer, chained back to the input (blue, green, purple). Optional 2×2 max pooling (stride 2) after every convolution.
// config: { layers: 3, k: 3, pool: false, size: 210, mini: false }
// mini: a small reminder copy (no controls, no readout, no size line under the layer names).
import { C, h } from './util.js';

export function mount(el, cfg) {
  el.classList.add('widget');
  const st = { L: cfg.layers ?? 3, k: cfg.k ?? 3, pool: !!cfg.pool };
  const size = cfg.size || 210, mini = !!cfg.mini;
  const svgWrap = h('div', { style: `position:relative; display:flex; gap:${mini ? 10 : 22}px; align-items:flex-end; min-height:${size + (mini ? 30 : 60)}px` });
  let svgs = [], pairs = [];                               // grid svgs; pixel → window pairs drawn in one colour each
  const readout = h('div', { class: 'readout', style: 'font-size:1.15em; color:' + C.fg + '; margin-top:0.3em' });
  const NS = 'http://www.w3.org/2000/svg';
  const S = (tag, a = {}, text) => { const e = document.createElementNS(NS, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };

  function draw() {
    const { L, k, pool } = st, r = (k - 1) / 2;
    // grid sizes: input N; conv keeps the size ("same" padding), pooling halves it
    const sizes = [pool ? 2 ** L * 4 : 15];
    for (let l = 1; l <= L; l++) sizes.push(pool ? sizes[l - 1] / 2 : sizes[l - 1]);
    // region of influence, backwards from one output pixel (near the centre)
    const regions = new Array(L + 1);
    const c = Math.floor((sizes[L] - 1) / 2);
    regions[L] = [c, c];
    for (let l = L; l >= 1; l--) {
      let [lo, hi] = regions[l];
      if (pool) { lo = 2 * lo; hi = 2 * hi + 1; }          // the pooling window of the layer-l pixel
      regions[l - 1] = [lo - r, hi + r];                     // the convolution window
    }
    // The chain of the old slide: the output pixel and the window it is computed from in the previous layer, then
    // one pixel of that window (its top-left) and its own window one layer further back, and so on.
    const PAIR = [C.blue, C.green, C.purple];
    const pix = new Array(L + 1), win = new Array(L);
    pix[L] = c;
    for (let l = L - 1; l >= 0; l--) {
      const p = pix[l + 1];
      win[l] = pool ? [2 * p - r, 2 * p + 1 + r] : [p - r, p + r];
      pix[l] = Math.max(0, win[l][0]);
    }
    svgWrap.replaceChildren();
    svgs = []; pairs = [];
    sizes.forEach((n, l) => {
      const cs = size / n, svg = S('svg', { width: size + 2, height: size + 2 });
      const [lo, hi] = regions[l], cl = Math.max(0, lo), ch = Math.min(n - 1, hi);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const inR = i >= cl && i <= ch && j >= cl && j <= ch, out = l === L && i === c && j === c;
        svg.append(S('rect', { x: 1 + j * cs, y: 1 + i * cs, width: cs, height: cs, fill: out ? C.blue : inR ? 'rgba(242,177,52,0.55)' : C.bg2, stroke: C.line, 'stroke-width': 0.8 }));
      }
      if (l < L || L === 0) svg.append(S('rect', { x: 1 + cl * cs, y: 1 + cl * cs, width: (ch - cl + 1) * cs, height: (ch - cl + 1) * cs, fill: 'none', stroke: C.accent, 'stroke-width': 2.5 }));
      const w = ch - cl + 1;
      svgs.push(svg);
      if (l < L) {                                           // window of the chosen pixel of layer l + 1
        const color = PAIR[(L - 1 - l) % PAIR.length], a = Math.max(0, win[l][0]), b = Math.min(n - 1, win[l][1]);
        const box = { x: 1 + a * cs, y: 1 + a * cs, s: (b - a + 1) * cs };
        svg.append(S('rect', { x: box.x, y: box.y, width: box.s, height: box.s, fill: 'none', stroke: color, 'stroke-width': 3 }));
        pairs.push({ l, color, win: box });
      }
      if (l > 0) {                                           // the chosen pixel of this layer
        const color = PAIR[(L - l) % PAIR.length], q = pix[l];
        svg.append(S('rect', { x: 1 + q * cs, y: 1 + q * cs, width: cs, height: cs, fill: 'none', stroke: color, 'stroke-width': 3 }));
        pairs.find(pp => pp.l === l - 1).pix = { x: 1 + q * cs, y: 1 + q * cs, s: cs };
      }
      svgWrap.append(h('div', { style: 'text-align:center' },
        h('div', { style: `color:${l === 0 ? C.fg : l === L ? C.blue : C.dim}; font-weight:600` }, l === 0 ? 'input' : `layer ${l}`),
        mini ? '' : h('div', { class: 'small', style: 'min-height:1.3em' }, `${n}×${n}${l < L ? ` · region ${w}×${w}` : ''}`), svg));
    });
    drawLines();
    const [lo, hi] = regions[0], rf = hi - lo + 1;
    readout.textContent = pool
      ? `receptive field in the input: ${rf}×${rf} pixels (pooling doubles the growth each layer)`
      : `receptive field: ${rf}×${rf} = 1 + L·(k − 1) = 1 + ${L}·${k - 1}`;
  }
  // Lines from the region in one layer to the region (or pixel) it feeds in the next layer: an overlay over the
  // whole row, positioned from the rendered grids (redrawn when the slide becomes visible or is resized).
  function drawLines() {
    svgWrap.querySelector('svg.rf-lines')?.remove();
    const wr = svgWrap.getBoundingClientRect(), k = wr.width / (svgWrap.offsetWidth || 1);
    if (!wr.width || svgs.length < 2) return;
    const ov = S('svg', { class: 'rf-lines', width: svgWrap.offsetWidth, height: svgWrap.offsetHeight,
      style: 'position:absolute; left:0; top:0; pointer-events:none; overflow:visible' });
    const pos = (svg, b) => { const r = svg.getBoundingClientRect(); return [(r.left - wr.left) / k + b.x, (r.top - wr.top) / k + b.y]; };
    for (const p of pairs) {                                 // window (layer l) → pixel (layer l + 1)
      const [x0, y0] = pos(svgs[p.l], p.win), [x1, y1] = pos(svgs[p.l + 1], p.pix), a = p.win.s, b = p.pix.s;
      for (const [ya, yb] of [[y0, y1], [y0 + a, y1 + b]])
        ov.append(S('line', { x1: x0 + a, y1: ya, x2: x1, y2: yb, stroke: p.color, 'stroke-width': 2 }));
    }
    svgWrap.append(ov);
  }
  new ResizeObserver(drawLines).observe(svgWrap);
  const group = (label, vals, key, fmt = String) => {
    const bs = vals.map(v => h('button', { onclick: () => { st[key] = v; bs.forEach((b, i) => b.classList.toggle('active', vals[i] === v)); draw(); } }, fmt(v)));
    bs.forEach((b, i) => b.classList.toggle('active', vals[i] === st[key]));
    return [h('label', { style: 'margin-left:0.8em' }, label), ...bs];
  };
  if (mini) { el.append(svgWrap); draw(); return; }
  el.append(svgWrap, readout, h('div', { class: 'wctl interactive-only' },
    ...group('layers L', [1, 2, 3], 'L'), ...group('kernel k', [3, 5], 'k'),
    ...group('2×2 pooling after each layer', [false, true], 'pool', v => (v ? 'yes' : 'no'))));
  draw();
}
