// "If I change one weight by ε, what happens to the loss?" — on the tiny network of the hand-gradient slide.
// Pick a parameter; two log-scale sliders set a positive and a negative nudge. The network shows the forward
// pass for w + ε₊; the plot shows L as a function of the chosen weight with the two secant lines to w + ε₊ and
// w − ε₋, whose slopes are the difference quotients. As ε → 0 both secants turn into the tangent.
// config: { param: "w11" }
import { C, h, fmt } from './util.js';
import { X, Y, W1, B1, W2, B2, compute } from './handgrad.js';

const PARAMS = {
  w11: { label: 'w₁₁', get: p => p.W1[0][0], set: (p, v) => { p.W1[0][0] = v; } },
  w12: { label: 'w₁₂', get: p => p.W1[0][1], set: (p, v) => { p.W1[0][1] = v; } },
  w21: { label: 'w₂₁', get: p => p.W1[1][0], set: (p, v) => { p.W1[1][0] = v; } },
  w22: { label: 'w₂₂', get: p => p.W1[1][1], set: (p, v) => { p.W1[1][1] = v; } },
  v1: { label: 'v₁', get: p => p.W2[0], set: (p, v) => { p.W2[0] = v; } },
  v2: { label: 'v₂', get: p => p.W2[1], set: (p, v) => { p.W2[1] = v; } },
  b2: { label: 'b₂', get: p => p.B2, set: (p, v) => { p.B2 = v; } },
};
const clone = () => ({ W1: W1.map(r => r.slice()), B1: B1.slice(), W2: W2.slice(), B2 });
const evalAt = (key, eps) => { const p = clone(); PARAMS[key].set(p, PARAMS[key].get(p) + eps); return compute(p.W1, p.B1, p.W2, p.B2); };
const LOGMIN = -4, LOGMAX = Math.log10(0.5);
const epsOf = s => 10 ** (LOGMIN + (LOGMAX - LOGMIN) * s);   // slider position in [0, 1] → ε magnitude
const POS = '#3987e5', NEG = '#c98500';

export function mount(el, cfg) {
  const st = { key: cfg.param || 'w11', sp: cfg.sp ?? 1, sn: cfg.sn ?? 1 };        // slider positions (1 = largest ε)
  const L0 = compute(W1, B1, W2, B2).L;
  const ns = 'http://www.w3.org/2000/svg';
  const S = (tag, a = {}, text) => { const e = document.createElementNS(ns, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };
  const NW = 820, NH = 290;
  const net = S('svg', { viewBox: `0 0 ${NW} ${NH}`, width: 560, height: Math.round(560 * NH / NW), style: 'font-family: Inter, Arial, sans-serif' });
  const PW = 560, PH = 230;
  const plot = S('svg', { viewBox: `0 0 ${PW} ${PH}`, width: PW, height: PH, style: 'font-family: Inter, Arial, sans-serif; background:#1e2128; border:1px solid #3a3f4b; border-radius:8px' });
  const readout = h('div', { style: 'line-height:1.55; font-size:1.0em' });
  const P = { x: [[50, 65], [50, 225]], z: [[380, 65], [380, 225]], y: [[600, 145]], L: [[750, 145]] };
  const HW = { x: 30, z: 59, y: 30, L: 46 };
  const amber = '#f2b134';

  function drawNet(v, epsP) {
    net.replaceChildren();
    const edge = (a, aw, b, bw, key, label, t) => {
      const ax = a[0] + aw, ay = a[1], bx = b[0] - bw, by = b[1], sel = key === st.key;
      net.append(S('line', { x1: ax, y1: ay, x2: bx, y2: by, stroke: sel ? amber : '#5a606c', 'stroke-width': sel ? 4 : 2 }));
      const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy), side = key === 'w12' ? -1 : 1, nx = dy / len * side, ny = -dx / len * side;
      const txt = S('text', { x: ax + dx * t + nx * 14, y: ay + dy * t + ny * 14 + (ny > 0 ? 10 : 0), fill: sel ? amber : C.fg, 'font-size': 18, 'font-weight': sel ? 700 : 400, 'text-anchor': 'middle', style: 'cursor:pointer' }, label);
      txt.addEventListener('click', () => { st.key = key; sync(); draw(); });
      net.append(txt);
    };
    const val = (key, base) => (key === st.key ? `${base}+ε` : `${base}`);
    [['w11', 0, 0], ['w12', 0, 1], ['w21', 1, 0], ['w22', 1, 1]].forEach(([key, i, j]) =>
      edge(P.x[j], HW.x, P.z[i], HW.z, key, `${PARAMS[key].label}=${val(key, W1[i][j])}`, i === j ? 0.3 : 0.8));
    edge(P.z[0], HW.z, P.y[0], HW.y, 'v1', `v₁=${val('v1', W2[0])}`, 0.45);
    edge(P.z[1], HW.z, P.y[0], HW.y, 'v2', `v₂=${val('v2', W2[1])}`, 0.45);
    net.append(S('line', { x1: P.y[0][0] + HW.y, y1: 145, x2: P.L[0][0] - HW.L, y2: 145, stroke: '#5a606c', 'stroke-width': 2 }));
    const node = ([x, y], w, name, value, color) => {
      net.append(S('rect', { x: x - w, y: y - 27, width: 2 * w, height: 54, rx: 10, fill: '#272b34', stroke: color, 'stroke-width': 2 }));
      net.append(S('text', { x, y: y - 5, fill: C.dim, 'font-size': 16, 'text-anchor': 'middle' }, name));
      net.append(S('text', { x, y: y + 18, fill: C.fg, 'font-size': 19, 'font-weight': 600, 'text-anchor': 'middle' }, value));
    };
    node(P.x[0], HW.x, 'x₁', String(X[0]), C.dim); node(P.x[1], HW.x, 'x₂', String(X[1]), C.dim);
    node(P.z[0], HW.z, 'h₁ = ReLU(z₁)', fmt(v.hh[0], 4), C.purple);
    node(P.z[1], HW.z, 'h₂ = ReLU(z₂)', fmt(v.hh[1], 4), v.z[1] > 0 ? C.purple : '#5a606c');
    node(P.y[0], HW.y, 'ŷ', fmt(v.yh, 4), C.blue);
    node(P.L[0], HW.L, `L (y=${Y})`, fmt(v.L, 4), C.accent);
    const b2t = S('text', { x: P.y[0][0], y: 145 - 36, fill: st.key === 'b2' ? amber : C.dim, 'font-size': 16, 'text-anchor': 'middle', style: 'cursor:pointer' }, `b₂=${val('b2', B2)}`);
    b2t.addEventListener('click', () => { st.key = 'b2'; sync(); draw(); });
    net.append(b2t);
    net.append(S('text', { x: 10, y: NH - 6, fill: C.dim, 'font-size': 15 }, `forward pass shown for ε = +${fmt(epsP, 5)}`));
  }

  function drawPlot(eP, eN) {
    plot.replaceChildren();
    const w0 = PARAMS[st.key].get(clone()), span = 0.6;
    const Lw = d => evalAt(st.key, d).L;
    const M = { l: 44, r: 12, t: 12, b: 28 };
    const ds = Array.from({ length: 241 }, (_, i) => -span + 2 * span * i / 240), ls = ds.map(Lw);
    const lmin = Math.min(...ls), lmax = Math.max(...ls), pad = Math.max(0.05, (lmax - lmin) * 0.12);
    const sx = d => M.l + (d + span) / (2 * span) * (PW - M.l - M.r), sy = l => PH - M.b - (l - (lmin - pad)) / (lmax - lmin + 2 * pad) * (PH - M.t - M.b);
    plot.append(S('line', { x1: sx(0), x2: sx(0), y1: M.t, y2: PH - M.b, stroke: '#3a3f4b', 'stroke-dasharray': '3 4' }));
    plot.append(S('path', { d: ds.map((d, i) => `${i ? 'L' : 'M'}${sx(d).toFixed(1)},${sy(ls[i]).toFixed(1)}`).join(''), fill: 'none', stroke: C.fg, 'stroke-width': 2 }));
    // secants: slope = difference quotient, extended across the plot
    const secant = (d1, color, dash) => {
      const l1 = Lw(d1), m = (l1 - L0) / d1;
      const xa = -span, xb = span;
      plot.append(S('line', { x1: sx(xa), y1: sy(L0 + m * xa), x2: sx(xb), y2: sy(L0 + m * xb), stroke: color, 'stroke-width': 2, 'stroke-dasharray': dash ? '7 6' : 'none' }));
      plot.append(S('circle', { cx: sx(d1), cy: sy(l1), r: 5, fill: color }));
    };
    plot.setAttribute('overflow', 'hidden');
    secant(eP, POS, false); secant(-eN, NEG, true);
    plot.append(S('circle', { cx: sx(0), cy: sy(L0), r: 6, fill: C.fg, stroke: '#1e2128', 'stroke-width': 2 }));
    plot.append(S('text', { x: PW - M.r, y: PH - 8, fill: C.dim, 'font-size': 14, 'text-anchor': 'end' }, `${PARAMS[st.key].label} → (others fixed)`));
    plot.append(S('text', { x: 8, y: 18, fill: C.dim, 'font-size': 14 }, 'L'));
    plot.append(S('text', { x: sx(0) + 6, y: PH - 8, fill: C.dim, 'font-size': 13 }, `${fmt(w0, 2)}`));
  }

  function draw() {
    const eP = epsOf(st.sp), eN = epsOf(st.sn);
    const vP = evalAt(st.key, eP), vN = evalAt(st.key, -eN);
    drawNet(vP, eP); drawPlot(eP, eN);
    const qP = (vP.L - L0) / eP, qN = (L0 - vN.L) / eN;
    const hl = st.animating ? 'background:#3a3f4b; border-radius:4px; padding:0 0.25em; font-size:1.15em; transition:font-size .2s' : '';
    readout.innerHTML =
      `<div>L(θ) = <b>${fmt(L0, 4)}</b></div>` +
      `<div style="margin-top:0.3em"><span style="color:${POS}">■</span> ε = +${fmt(eP, 5)}</div>` +
      `<div>&nbsp;&nbsp;(L(w+ε) − L)/ε = <b style="color:${POS}; ${hl}">${qP.toFixed(5)}</b></div>` +
      `<div style="margin-top:0.3em"><span style="color:${NEG}">■</span> ε = −${fmt(eN, 5)}</div>` +
      `<div>&nbsp;&nbsp;(L − L(w−ε))/ε = <b style="color:${NEG}; ${hl}">${qN.toFixed(5)}</b></div>`;
    lblP.textContent = `+${fmt(eP, 5)}`; lblN.textContent = `−${fmt(eN, 5)}`;
  }
  const btns = Object.entries(PARAMS).map(([k, d]) => h('button', { onclick: () => { st.key = k; sync(); draw(); } }, d.label));
  function sync() { btns.forEach((b, i) => b.classList.toggle('active', Object.keys(PARAMS)[i] === st.key)); }
  const sliders = {};
  const mkSlider = key => { const s = sliders[key] = h('input', { type: 'range', min: 0, max: 1, step: 0.002, value: st[key], style: `width:9em; accent-color:${key === 'sp' ? POS : NEG}` }); s.addEventListener('input', () => { st[key] = +s.value; draw(); }); return s; };
  const lblP = h('span', { class: 'readout', style: 'display:inline-block; width:5.2em' }), lblN = h('span', { class: 'readout', style: 'display:inline-block; width:5.2em' });

  // ▶ shrink ε: glide both sliders from the largest to the smallest ε (log scale), about 6 s
  const play = h('button', {
    onclick: () => {
      if (st.animating) return;
      st.animating = true; play.disabled = true;
      const t0 = performance.now(), dur = 6000;
      const step = now => {
        const u = Math.min(1, (now - t0) / dur), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        st.sp = st.sn = 1 - e; sliders.sp.value = st.sp; sliders.sn.value = st.sn; draw();
        if (u < 1 && el.isConnected) requestAnimationFrame(step);
        else setTimeout(() => { st.animating = false; play.disabled = false; draw(); }, 1500);
      };
      requestAnimationFrame(step);
    },
  }, '▶ shrink ε → 0');
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' },
    h('div', { style: 'display:flex; flex-direction:column; gap:0.3em' }, net, plot),
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:17em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'flex-direction:column; align-items:flex-start; gap:0.35em' },
        h('div', { style: 'display:flex; flex-wrap:wrap; gap:0.25em' }, btns),
        h('div', {}, h('label', { style: `color:${POS}` }, 'ε > 0 '), mkSlider('sp'), ' ', lblP),
        h('div', {}, h('label', { style: `color:${NEG}` }, 'ε < 0 '), mkSlider('sn'), ' ', lblN),
        play,
        h('div', { class: 'dim' }, 'log scale: slide left to shrink ε towards 0')))));
  sync(); draw();
}
