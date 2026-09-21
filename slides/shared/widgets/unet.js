// U-Net as a diagram: the encoder goes down (conv, conv, max pool: half the size, double the channels), the
// decoder goes up (up-conv: double the size; concatenate the encoder map of the same size; conv, conv), and a
// 1×1 convolution gives C class scores for every pixel. Block height ~ spatial size, width ~ channels.
// config: { levels: [[256, 64], [128, 128], [64, 256], [32, 512], [16, 1024]], width: 1100, in: 3,
//           highlight: decoder level to outline (dashed amber, labelled "zoom"), legendSize: 15 }
import { C } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};

export function mount(el, cfg) {
  el.classList.add('widget');
  const L = cfg.levels || [[256, 64], [128, 128], [64, 256], [32, 512], [16, 1024]];
  const n = L.length, VW = 1180, VH = 545, gap = 10, step = 108;
  const svg = S('svg', { viewBox: `0 0 ${VW} ${VH}`, width: cfg.width || VW, height: (cfg.width || VW) * VH / VW, style: 'font-family: Inter, Arial, sans-serif; display:block' });
  const hgt = s => 18 + 96 * Math.sqrt(s / L[0][0]);
  const wid = c => 5 + 1.5 * Math.sqrt(c);
  const yc = i => 80 + 95 * i;
  const text = (x, y, t, color = C.fg, size = 13, anchor = 'middle') => svg.append(S('text', { x, y, fill: color, 'font-size': size, 'text-anchor': anchor }, t));
  const rect = (x, i, w, color) => {
    const hh = hgt(L[i][0]);
    svg.append(S('rect', { x, y: yc(i) - hh / 2, width: w, height: hh, fill: color, 'fill-opacity': 0.3, stroke: color, 'stroke-width': 1.5 }));
  };
  const arrow = (x1, y1, x2, y2, color, dash) => {
    const a = Math.atan2(y2 - y1, x2 - x1), hx = x2 - 9 * Math.cos(a), hy = y2 - 9 * Math.sin(a);
    svg.append(S('line', { x1, y1, x2: hx, y2: hy, stroke: color, 'stroke-width': 2, 'stroke-dasharray': dash ? '6 4' : 'none' }));
    svg.append(S('polygon', { points: `${x2},${y2} ${hx - 5 * Math.sin(a)},${hy + 5 * Math.cos(a)} ${hx + 5 * Math.sin(a)},${hy - 5 * Math.cos(a)}`, fill: color }));
  };
  const top = i => yc(i) - hgt(L[i][0]) / 2, bot = i => yc(i) + hgt(L[i][0]) / 2;

  // encoder: two conv blocks per level (the bottom level is the bottleneck, centred)
  const enc = [];
  for (let i = 0; i < n; i++) {
    const w = wid(L[i][1]), pw = 2 * w + gap;
    const x = i < n - 1 ? 110 + i * step : VW / 2 - pw / 2 - 30;
    enc.push({ x, w, pw });
  }
  // decoder: [copied encoder map | upsampled map] (concatenated), conv, conv
  const dec = [];
  for (let i = 0; i < n - 1; i++) {
    const w = wid(L[i][1]), gw = 4 * w + 2 * gap, right = VW - 150 - i * (step - 12);
    dec.push({ x: right - gw, w, gw });
  }
  // input
  const e0 = enc[0], win = wid(cfg.in || 3);
  rect(e0.x - 50, 0, win, '#9aa1ae'); text(e0.x - 50 + win / 2, top(0) - 8, String(cfg.in || 3), C.dim);
  arrow(e0.x - 50 + win + 2, yc(0), e0.x - 2, yc(0), C.dim);
  text(e0.x - 50 + win / 2, bot(0) + 18, 'image', C.dim, 13);
  for (let i = 0; i < n; i++) {
    const { x, w } = enc[i], [s, c] = L[i];
    rect(x, i, w, C.blue); rect(x + w + gap, i, w, C.blue);
    arrow(x + w + 1, yc(i), x + w + gap - 1, yc(i), C.dim);
    text(x + w + gap / 2, top(i) - 7, String(c));
    text(x + w + gap / 2, bot(i) + 16, `${s}×${s}`, C.dim, 12);
    if (i < n - 1) arrow(x + 1.5 * w + gap, bot(i) + 2, enc[i + 1].x + enc[i + 1].w / 2, top(i + 1) - 3, C.red);
  }
  for (let i = n - 2; i >= 0; i--) {
    const { x, w } = dec[i], c = L[i][1];
    rect(x, i, w, C.blue); rect(x + w, i, w, C.green);
    rect(x + 2 * w + gap, i, w, C.green); rect(x + 3 * w + 2 * gap, i, w, C.green);
    arrow(x + 2 * w + 1, yc(i), x + 2 * w + gap - 1, yc(i), C.dim);
    arrow(x + 3 * w + gap + 1, yc(i), x + 3 * w + 2 * gap - 1, yc(i), C.dim);
    text(x + w, top(i) - 7, `${c}+${c}`);
    text(x + 3 * w + 1.5 * gap, top(i) - 7, String(c));
    // skip connection: encoder level i → decoder level i
    const e = enc[i];
    arrow(e.x + e.pw + 3, yc(i) - hgt(L[i][0]) / 2 + 12, x - 3, yc(i) - hgt(L[i][0]) / 2 + 12, C.accent, true);
    // up-conv from the level below
    const fx = i === n - 2 ? enc[n - 1].x + enc[n - 1].pw - enc[n - 1].w / 2 : dec[i + 1].x + dec[i + 1].gw - dec[i + 1].w / 2;
    arrow(fx, top(i + 1) - 3, x + 1.5 * w, bot(i) + 3, C.green);
  }
  if (cfg.highlight !== undefined) {
    const i = cfg.highlight, { x, gw } = dec[i], hh = hgt(L[i][0]);
    svg.append(S('rect', { x: x - 8, y: yc(i) - hh / 2 - 26, width: gw + 16, height: hh + 34, fill: 'none', stroke: C.accent, 'stroke-width': 2.5, 'stroke-dasharray': '7 5', rx: 6 }));
    text(x + gw + 14, yc(i) + 5, 'zoom →', C.accent, 18, 'start');
  }
  // 1×1 conv → class scores
  const d0 = dec[0], ox = d0.x + d0.gw + 40;
  arrow(d0.x + d0.gw + 2, yc(0), ox - 2, yc(0), C.purple);
  rect(ox, 0, wid(cfg.classes || 2) + 2, C.purple);
  text(ox + 6, top(0) - 7, 'C'); text(ox + 6, bot(0) + 18, 'scores per pixel', C.dim, 13, 'start');
  // legend
  const ly = VH - 16, items = [[C.dim, 'conv 3×3 + ReLU', false], [C.red, 'max pool 2×2', false], [C.green, 'up-conv 2×2', false], [C.accent, 'copy + concatenate', true], [C.purple, 'conv 1×1 → C scores', false]];
  let lx = 40;
  items.forEach(([c, t, dsh]) => { arrow(lx, ly, lx + 34, ly, c, dsh); text(lx + 42, ly + 5, t, C.fg, cfg.legendSize || 15, 'start'); lx += 60 + t.length * 7.8 * (cfg.legendSize || 15) / 15; });
  el.append(svg);
}
