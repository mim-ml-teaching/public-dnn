// Gradients by hand on a tiny network (2 inputs → 2 ReLU units → 1 output), one data point, squared loss.
//   z = W1 x + b1,  h = relu(z),  ŷ = w2·h + b2,  L = ½ (ŷ − y)²
// Step through: forward values, loss, backward (chain rule), one gradient step. Forward values are white,
// gradients amber. One hidden unit is inactive (z < 0), so its gradient is blocked by the ReLU.
// config: { step: 0 }
import { C, h, fmt } from './util.js';

export const X = [1, 2], Y = 1, LR = 0.1;
export const W1 = [[0.5, 0.25], [-1, 0.25]], B1 = [0, 0.2], W2 = [1.5, 2.0], B2 = 0.5;

export function compute(W1, B1, W2, B2) {
  const z = W1.map((r, i) => r[0] * X[0] + r[1] * X[1] + B1[i]);
  const hh = z.map(v => Math.max(0, v));
  const yh = W2[0] * hh[0] + W2[1] * hh[1] + B2;
  const L = 0.5 * (yh - Y) ** 2;
  const dyh = yh - Y;
  const dW2 = hh.map(v => dyh * v), dB2 = dyh;
  const dh = W2.map(w => w * dyh);
  const dz = dh.map((v, i) => (z[i] > 0 ? v : 0));
  const dW1 = dz.map(d => X.map(x => d * x)), dB1 = dz.slice();
  return { z, hh, yh, L, dyh, dW2, dB2, dh, dz, dW1, dB1 };
}

const STEPS = [
  { t: 'The network and one training example', eq: String.raw`\mathbf{x} = (1, 2),\; y = 1 \qquad \mathbf{z} = W_1 \mathbf{x} + \mathbf{b}_1,\; \mathbf{h} = \mathrm{ReLU}(\mathbf{z}),\; \hat y = \mathbf{v}^\top \mathbf{h} + b_2,\; L = \tfrac12(\hat y - y)^2` },
  { t: 'Forward: pre-activations', eq: String.raw`z_1 = 0.5\cdot1 + 0.25\cdot2 + 0 = 1,\qquad z_2 = -1\cdot1 + 0.25\cdot2 + 0.2 = -0.3` },
  { t: 'Forward: ReLU', eq: String.raw`h_1 = \mathrm{ReLU}(1) = 1,\qquad h_2 = \mathrm{ReLU}(-0.3) = 0` },
  { t: 'Forward: output and loss', eq: String.raw`\hat y = 1.5\cdot1 + 2\cdot0 + 0.5 = 2,\qquad L = \tfrac12(2-1)^2 = 0.5` },
  { t: 'Backward: start at the loss', eq: String.raw`\frac{\partial L}{\partial \hat y} = \hat y - y = 1` },
  { t: 'Backward: output layer', eq: String.raw`\frac{\partial L}{\partial v_i} = \frac{\partial L}{\partial \hat y}\, h_i = (1,\ 0),\qquad \frac{\partial L}{\partial b_2} = 1` },
  { t: 'Backward: into the hidden layer', eq: String.raw`\frac{\partial L}{\partial h_i} = \frac{\partial L}{\partial \hat y}\, v_i = (1.5,\ 2)` },
  { t: 'Backward: through the ReLU', eq: String.raw`\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial h_i}\,\mathrm{ReLU}'(z_i) = (1.5\cdot1,\ 2\cdot0) = (1.5,\ 0)\quad\text{— unit 2 is off}` },
  { t: 'Backward: first layer', eq: String.raw`\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_i}\, x_j = \begin{bmatrix}1.5 & 3\\ 0 & 0\end{bmatrix},\qquad \frac{\partial L}{\partial \mathbf{b}_1} = (1.5,\ 0)` },
  { t: 'One gradient step (η = 0.1)', eq: String.raw`\theta \leftarrow \theta - \eta\,\nabla_\theta L \quad\Rightarrow\quad L:\ 0.5 \;\to\; LNEW` },
];

export function mount(el, cfg) {
  const st = { k: cfg.step ?? 0 };
  const base = compute(W1, B1, W2, B2);
  // after one step
  const nW1 = W1.map((r, i) => r.map((w, j) => w - LR * base.dW1[i][j])), nB1 = B1.map((b, i) => b - LR * base.dB1[i]);
  const nW2 = W2.map((w, i) => w - LR * base.dW2[i]), nB2 = B2 - LR * base.dB2;
  const after = compute(nW1, nB1, nW2, nB2);
  STEPS.at(-1).eq = STEPS.at(-1).eq.replace('LNEW', fmt(after.L, 3));

  const ns = 'http://www.w3.org/2000/svg';
  const S = (tag, a = {}, text) => { const e = document.createElementNS(ns, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };
  const W = 860, H = 350;
  const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, width: Math.round(W * 1.3), height: Math.round(H * 1.3), style: 'font-family: Inter, Arial, sans-serif' });
  const eqEl = h('div', { style: 'min-height:3.2em; font-size:1.05em' });
  const title = h('div', { style: 'font-weight:600; color:' + C.blue });

  const P = { x: [[60, 80], [60, 250]], z: [[420, 80], [420, 250]], y: [[640, 165]], L: [[775, 165]] };
  const HW = { x: 30, z: 59, y: 30 };   // half widths of the node boxes
  function draw() {
    svg.replaceChildren();
    const k = st.k, fwd = v => fmt(v, 3), amber = '#f2b134';
    const edge = (a, aw, b, bw, label, grad, gradOn, off, t = 0.5, side = 1) => {
      const ax = a[0] + aw, ay = a[1], bx = b[0] - bw, by = b[1];
      svg.append(S('line', { x1: ax, y1: ay, x2: bx, y2: by, stroke: off ? '#3a3f4b' : '#5a606c', 'stroke-width': 2 }));
      const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy), nx = dy / len * side, ny = -dx / len * side;
      const px = ax + dx * t, py = ay + dy * t;
      svg.append(S('text', { x: px + nx * 13, y: py + ny * 13 + (ny > 0 ? 10 : 0), fill: C.fg, 'font-size': 15, 'text-anchor': 'middle' }, label));
      if (gradOn) svg.append(S('text', { x: px - nx * 14, y: py - ny * 14 + (ny < 0 ? 12 : 0), fill: amber, 'font-size': 14, 'text-anchor': 'middle' }, `∂=${fmt(grad, 3)}`));
    };
    // edges with weights; gradients appear at steps 5 and 8
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) edge(P.x[j], HW.x, P.z[i], HW.z, `w${'₁₂'[i]}${'₁₂'[j]}=${W1[i][j]}`, base.dW1[i][j], k >= 8, k >= 7 && i === 1, i === j ? 0.3 : 0.8, i === j ? 1 : (i === 0 ? -1 : 1));
    for (let i = 0; i < 2; i++) edge(P.z[i], HW.z, P.y[0], HW.y, `v${'₁₂'[i]}=${W2[i]}`, base.dW2[i], k >= 5, k >= 7 && i === 1, 0.45, i === 0 ? 1 : -1);
    svg.append(S('line', { x1: P.y[0][0] + HW.y, y1: 165, x2: P.L[0][0] - (k >= 9 ? 65 : 43), y2: 165, stroke: '#5a606c', 'stroke-width': 2 }));
    // nodes
    const node = ([x, y], name, val, show, grad, gshow, color = C.purple, w = 60) => {
      svg.append(S('rect', { x: x - w / 2, y: y - 26, width: w, height: 52, rx: 10, fill: '#272b34', stroke: color, 'stroke-width': 2 }));
      svg.append(S('text', { x, y: y - 3, fill: C.dim, 'font-size': 14, 'text-anchor': 'middle' }, name));
      if (show) svg.append(S('text', { x, y: y + 17, fill: C.fg, 'font-size': 15, 'font-weight': 600, 'text-anchor': 'middle' }, val));
      if (gshow) svg.append(S('text', { x, y: y + 46, fill: amber, 'font-size': 14, 'text-anchor': 'middle' }, grad));
    };
    node(P.x[0], 'x₁', '1', true, '', false, C.dim);
    node(P.x[1], 'x₂', '2', true, '', false, C.dim);
    for (let i = 0; i < 2; i++) {
      const off = k >= 2 && base.z[i] <= 0;
      node(P.z[i], `z${'₁₂'[i]} → h${'₁₂'[i]}`, k >= 2 ? `${fwd(base.z[i])} → ${fwd(base.hh[i])}` : fwd(base.z[i]), k >= 1,
        k >= 7 ? `∂z=${fmt(base.dz[i], 3)}` : `∂h=${fmt(base.dh[i], 3)}`, k >= 6, off ? '#5a606c' : C.purple, 118);
      svg.append(S('text', { x: P.z[i][0], y: P.z[i][1] - 34, fill: C.dim, 'font-size': 12, 'text-anchor': 'middle' }, `b₁=${B1[i]}${k >= 8 ? ` (∂=${fmt(base.dB1[i], 3)})` : ''}`));
    }
    node(P.y[0], 'ŷ', fwd(base.yh), k >= 3, `∂=${fmt(base.dyh, 3)}`, k >= 4, C.blue);
    svg.append(S('text', { x: P.y[0][0], y: 165 - 34, fill: C.dim, 'font-size': 12, 'text-anchor': 'middle' }, `b₂=${B2}${k >= 5 ? ` (∂=${fmt(base.dB2, 3)})` : ''}`));
    node(P.L[0], `L (y=${Y})`, k >= 9 ? `${fwd(base.L)} → ${fwd(after.L)}` : fwd(base.L), k >= 3, '', false, C.accent, k >= 9 ? 130 : 86);
    if (k >= 4 && k <= 8) svg.append(S('text', { x: 430, y: 338, fill: amber, 'font-size': 14, 'text-anchor': 'middle' }, '← gradients flow backwards (chain rule)'));
    if (k >= 1 && k <= 3) svg.append(S('text', { x: 430, y: 338, fill: C.dim, 'font-size': 14, 'text-anchor': 'middle' }, 'forward pass →'));

    title.textContent = `${k}/${STEPS.length - 1} · ${STEPS[k].t}`;
    eqEl.innerHTML = `$${STEPS[k].eq}$`;
    window.renderMathInElement?.(eqEl, { delimiters: [{ left: '$', right: '$', display: false }], throwOnError: false });
    prev.disabled = k === 0; next.disabled = k === STEPS.length - 1;
  }
  const prev = h('button', { onclick: () => { st.k = Math.max(0, st.k - 1); draw(); } }, '◀ back');
  const next = h('button', { onclick: () => { st.k = Math.min(STEPS.length - 1, st.k + 1); draw(); } }, 'next ▶');
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.6em; align-items:center' }, h('div', { class: 'wctl interactive-only', style: 'margin:0; flex:none' }, prev, next), title), svg, eqEl);
  draw();
}
