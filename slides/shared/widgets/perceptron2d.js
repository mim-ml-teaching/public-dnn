// The perceptron on binary inputs: y = [w1 x1 + w2 x2 + b ≥ 0].
//   mode "single": sliders for w1, w2, b; targets AND / OR / XOR; the line w·x + b = 0 and per-point check.
//   mode "two-layer": h1 = [x1 − x2 ≥ 1], h2 = [x2 − x1 ≥ 1], y = [h1 + h2 ≥ 1] solves XOR.
// config: { mode: "single" | "two-layer", target: "AND", width, height }
import { LIGHT } from '../theme.js';
import { C, h, fmt } from './util.js';

const PTS = [[0, 0], [1, 0], [0, 1], [1, 1]];
const TARGETS = { AND: (a, b) => a & b, OR: (a, b) => a | b, XOR: (a, b) => a ^ b };
const step = z => (z >= 0 ? 1 : 0);

export function mount(el, cfg) {
  const W = cfg.width || 440, H = cfg.height || 400;
  const st = { mode: cfg.mode || 'single', target: cfg.target || 'AND', w1: 1, w2: 1, b: -0.5 };
  const canvas = h('canvas', { width: W, height: H });
  const g = canvas.getContext('2d');
  const readout = h('div', { style: 'line-height:1.55; min-height:6.5em' });
  const X0 = -0.35, X1 = 1.75, Y0 = -0.35, Y1 = 1.4;      // visible region of the input plane
  const sx = x => (x - X0) / (X1 - X0) * W, sy = y => H - (y - Y0) / (Y1 - Y0) * H;
  const inv = (px, py) => [X0 + px / W * (X1 - X0), Y0 + (H - py) / H * (Y1 - Y0)];

  const unitsFor = () => (st.mode === 'single'
    ? [{ w: [st.w1, st.w2], b: st.b, color: C.purple }]
    : [{ w: [1, -1], b: -0.5, color: C.purple, name: 'h₁' }, { w: [-1, 1], b: -0.5, color: '#5fd38d', name: 'h₂' }]);
  const out = (x1, x2) => {
    if (st.mode === 'single') return step(st.w1 * x1 + st.w2 * x2 + st.b);
    const h1 = step(x1 - x2 - 0.5), h2 = step(x2 - x1 - 0.5);
    return step(h1 + h2 - 0.5);
  };

  function draw() {
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    // decision regions (class 1 tinted amber)
    const img = g.createImageData(W, H);
    for (let py = 0; py < H; py += 2) for (let px = 0; px < W; px += 2) {
      const [x1, x2] = inv(px + 1, py + 1), o = out(x1, x2);
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const k = 4 * ((py + dy) * W + px + dx);
        const c = LIGHT ? (o ? [252, 232, 186] : [232, 238, 248]) : (o ? [70, 58, 30] : [30, 36, 48]);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    // axes
    g.strokeStyle = '#5a606c'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(sx(-0.25), sy(0)); g.lineTo(sx(1.25), sy(0)); g.moveTo(sx(0), sy(-0.25)); g.lineTo(sx(0), sy(1.25)); g.stroke();
    g.fillStyle = C.dim; g.font = '16px Inter, Arial';
    g.fillText('x₁', sx(1.25) - 6, sy(0) + 22); g.fillText('x₂', sx(0) + 8, sy(1.3));
    // lines w·x + b = 0
    unitsFor().forEach(u => {
      const [a, b] = u.w, c = u.b, ends = [];
      if (Math.abs(b) > 1e-9) { ends.push([-1, -(a * -1 + c) / b], [2, -(a * 2 + c) / b]); }
      else if (Math.abs(a) > 1e-9) { ends.push([-c / a, -1], [-c / a, 2]); }
      if (ends.length) {
        g.strokeStyle = u.color; g.lineWidth = 2.5; g.beginPath(); g.moveTo(sx(ends[0][0]), sy(ends[0][1])); g.lineTo(sx(ends[1][0]), sy(ends[1][1])); g.stroke();
      }
      if (u.name) { g.fillStyle = u.color; g.font = '17px Inter, Arial'; g.fillText(u.name, sx(u.w[0] > 0 ? 1.0 : -0.25), sy(u.w[0] > 0 ? 0.62 : 0.4)); g.font = '16px Inter, Arial'; }
    });
    // the four points: colour = target, ring = correct / wrong
    const tgt = st.mode === 'single' ? TARGETS[st.target] : TARGETS.XOR;
    let ok = 0;
    PTS.forEach(([a, b]) => {
      const t = tgt(a, b), o = out(a, b), good = t === o; ok += good;
      g.beginPath(); g.arc(sx(a), sy(b), 13, 0, 7); g.fillStyle = t ? '#f2b134' : '#5ab0ff'; g.fill();
      g.lineWidth = 3; g.strokeStyle = good ? '#5fd38d' : '#ff6b6b'; g.stroke();
      g.fillStyle = C.fg; g.font = '15px Inter, Arial'; g.fillText(`(${a},${b}) → ${t}`, sx(a) + 18, sy(b) + 5);
    });
    readout.innerHTML = st.mode === 'single'
      ? `<div>$y = [\\,${fmt(st.w1, 2)}\\,x_1 + ${fmt(st.w2, 2)}\\,x_2 ${st.b < 0 ? '-' : '+'} ${fmt(Math.abs(st.b), 2)} \\ge 0\\,]$</div>` +
        `<div>target: <b>${st.target}</b> · correct: <b style="color:${ok === 4 ? C.green : C.red}">${ok}/4</b></div>` +
        `<div class="dim">${st.target === 'XOR' ? 'no line separates the amber from the blue points' : ok === 4 ? 'solved' : 'move the line until all four are correct'}</div>`
      : `<div>$h_1 = [x_1 - x_2 \\ge \\tfrac12]$</div><div>$h_2 = [x_2 - x_1 \\ge \\tfrac12]$</div><div>$y = [h_1 + h_2 \\ge \\tfrac12]$</div>` +
        `<div>XOR · correct: <b style="color:${C.green}">${ok}/4</b></div><div class="dim">two lines, one hidden layer: XOR is solved</div>`;
    window.renderMathInElement?.(readout, { delimiters: [{ left: '$', right: '$', display: false }], throwOnError: false });
  }

  const sl = (key, name) => { const s = h('input', { type: 'range', min: -2, max: 2, step: 0.05, value: st[key] }); s.addEventListener('input', () => { st[key] = +s.value; draw(); }); return h('div', {}, h('label', { style: 'display:inline-block; width:2em' }, name), s); };
  const tBtns = Object.keys(TARGETS).map(t => h('button', { onclick: () => { st.target = t; sync(); draw(); } }, t));
  const mBtns = [['single', 'one perceptron'], ['two-layer', 'two layers']].map(([m, txt]) => h('button', { onclick: () => { st.mode = m; sync(); draw(); } }, txt));
  const singleCtl = h('div', { style: 'display:flex; flex-direction:column; gap:0.3em' }, h('div', { style: 'display:flex; gap:0.3em' }, tBtns), sl('w1', 'w₁'), sl('w2', 'w₂'), sl('b', 'b'));
  function sync() {
    tBtns.forEach(b => b.classList.toggle('active', b.textContent === st.target));
    mBtns.forEach((b, i) => b.classList.toggle('active', ['single', 'two-layer'][i] === st.mode));
    singleCtl.style.display = st.mode === 'single' ? '' : 'none';
  }
  el.classList.add('widget');
  el.append(h('div', { style: 'display:flex; gap:0.8em; align-items:flex-start' }, canvas,
    h('div', { style: 'display:flex; flex-direction:column; gap:0.5em; width:17em; flex:none' }, readout,
      h('div', { class: 'wctl interactive-only', style: 'flex-direction:column; align-items:flex-start; gap:0.4em' },
        h('div', { style: 'display:flex; gap:0.3em' }, mBtns), singleCtl,
        h('div', { class: 'dim' }, 'amber region: output 1 · ring: green = correct, red = wrong')))));
  sync(); draw();
}
