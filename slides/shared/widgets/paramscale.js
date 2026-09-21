// Number of parameters of well-known networks, horizontal bars on a log scale, with hover details.
// config: { width, height, highlight: "ours" }
import { C, h } from './util.js';

const MODELS = [
  { name: 'LeNet-5', year: 1998, n: 6.0e4, note: 'digit recognition (LeCun et al.)' },
  { name: 'ResNet-50', year: 2015, n: 2.56e7, note: 'image classification' },
  { name: 'AlexNet', year: 2012, n: 6.1e7, note: 'the ImageNet breakthrough' },
  { name: 'BERT-base', year: 2018, n: 1.1e8, note: 'language understanding' },
  { name: 'GPT-3', year: 2020, n: 1.75e11, note: 'dense language model' },
  { name: 'Llama 3.1 405B', year: 2024, n: 4.05e11, note: 'dense open-weight language model' },
  { name: 'Kimi K2', year: 2025, n: 1.0e12, active: 3.2e10, note: 'mixture of experts: 384 experts, 8 used per token' },
  { name: 'DeepSeek V4-Pro', year: 2026, n: 1.6e12, active: 4.9e10, note: 'mixture of experts, open weights (April 2026)' },
  { name: 'Kimi K3', year: 2026, n: 2.8e12, active: 1.04e11, note: 'mixture of experts: 896 experts, 16 used per token; largest open-weight model (July 2026)' },
];

const human = n => (n >= 1e12 ? `${+(n / 1e12).toFixed(1)} T` : n >= 1e9 ? `${+(n / 1e9).toFixed(1)} B` : n >= 1e6 ? `${+(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${+(n / 1e3).toFixed(0)} k` : `${n}`);

export function mount(el, cfg) {
  const W = cfg.width || 900, rowH = 38, M = { l: 300, r: 210, t: 8, b: 58 };
  const H = M.t + MODELS.length * rowH + M.b;
  const lo = 3, hi = 13;                                   // 10^0 … 10^13
  const sx = n => M.l + (Math.log10(n) - lo) / (hi - lo) * (W - M.l - M.r);
  const ns = 'http://www.w3.org/2000/svg';
  const S = (tag, a = {}, text) => { const e = document.createElementNS(ns, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text !== undefined) e.textContent = text; return e; };
  const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: 'font-family: Inter, Arial, sans-serif' });
  for (let e = lo; e <= 12; e += 3) {
    svg.append(S('line', { x1: sx(10 ** e), x2: sx(10 ** e), y1: M.t, y2: H - M.b, stroke: '#2d313a' }));
    svg.append(S('text', { x: sx(10 ** e), y: H - 32, fill: C.dim, 'font-size': 16, 'text-anchor': 'middle' }, `10^${e}`));
  }
  const tip = h('div', { style: 'position:absolute; pointer-events:none; display:none; background:#272b34; border:1px solid #3a3f4b; border-radius:6px; padding:0.3em 0.6em; font-size:0.8em' });
  MODELS.forEach((m, i) => {
    const y = M.t + i * rowH;
    const color = '#3987e5';
    svg.append(S('text', { x: M.l - 10, y: y + rowH / 2 + 5, fill: C.fg, 'font-size': 18, 'text-anchor': 'end' }, `${m.name}${m.year ? ` (${m.year})` : ''}`));
    const bar = S('rect', { x: M.l, y: y + 7, width: Math.max(3, sx(m.n) - M.l), height: rowH - 14, rx: 4, fill: color });
    svg.append(bar);
    if (m.active) svg.append(S('rect', { x: M.l, y: y + 7 + (rowH - 14) * 0.3, width: Math.max(3, sx(m.active) - M.l), height: (rowH - 14) * 0.4, rx: 2, fill: '#c98500' }));
    svg.append(S('text', { x: sx(m.n) + 8, y: y + rowH / 2 + 6, fill: C.fg, 'font-size': 17 }, human(m.n) + (m.active ? `  (${human(m.active)} active)` : '')));
    const hit = S('rect', { x: 0, y, width: W, height: rowH, fill: 'transparent' });
    hit.addEventListener('pointermove', ev => {
      const r = el.getBoundingClientRect();
      tip.innerHTML = `<b>${m.name}</b>${m.year ? ` · ${m.year}` : ''}<br>${m.n.toLocaleString('en-US')} parameters${m.active ? `<br>${m.active.toLocaleString('en-US')} active per token` : ''}<br><span style="color:${C.dim}">${m.note}</span>`;
      tip.style.display = 'block';
      tip.style.left = `${(ev.clientX - r.left) / (r.width / el.offsetWidth) + 14}px`;
      tip.style.top = `${(ev.clientY - r.top) / (r.height / el.offsetHeight) - 10}px`;
    });
    hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
    svg.append(hit);
  });
  svg.append(S('text', { x: (M.l + W - M.r) / 2, y: H - 6, fill: C.dim, 'font-size': 16, 'text-anchor': 'middle' }, 'number of parameters (log scale)'));
  const legend = h('div', { style: 'display:flex; gap:1.2em; font-size:0.8em; color:' + C.dim },
    h('span', {}, h('span', { style: 'display:inline-block;width:14px;height:10px;border-radius:2px;background:#3987e5;margin-right:6px' }), 'total parameters'),
    h('span', {}, h('span', { style: 'display:inline-block;width:14px;height:5px;border-radius:2px;background:#c98500;margin-right:6px;vertical-align:middle' }), 'used per token (mixture of experts)'));
  el.classList.add('widget');
  el.style.position = 'relative';
  el.append(legend, svg, tip);
}
