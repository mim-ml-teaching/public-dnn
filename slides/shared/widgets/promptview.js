// A photo with what a promptable model was asked and what it returned (experiments/sam_modes.py, open_vocab.py):
// text prompts as coloured chips above the photo, boxes (dashed: a box given as the prompt; solid: a detected box
// with its label and score), clicks (green +, red −), and a mask overlay (RGBA PNG of the photo's size).
// config: { data: ".../sam_modes.json", panel: "box", photos: "../../assets/lecture07/", width: 400, title, scores: true,
//           crop: [x, y, w, h] (zoom, photo pixels) }
import { C, h } from './util.js';
import { mapHex, LIGHT } from '../theme.js';

const NS = 'http://www.w3.org/2000/svg';
const S = (tag, a = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const col = c => (LIGHT ? mapHex(c) : c);

export async function mount(el, cfg) {
  el.classList.add('widget');
  const d = await (await fetch(cfg.data)).json();
  const p = d.panels[cfg.panel], [W, H] = p.size, dir = cfg.data.slice(0, cfg.data.lastIndexOf('/') + 1);
  const [cx, cy, cw, ch] = cfg.crop || [0, 0, W, H];       // optional zoom: [x, y, w, h] in photo pixels
  const width = cfg.width || 400, k = cw / width;         // photo pixels per screen pixel: strokes and text scale with it
  if (cfg.title) el.append(h('div', { style: 'font-weight:600; margin-bottom:0.15em' }, cfg.title));
  if (p.prompts) el.append(h('div', { style: 'display:flex; flex-wrap:wrap; gap:0.3em; margin-bottom:0.25em; font-size:0.8em' },
    ...p.prompts.map(q => h('span', { style: `border:1.5px solid ${col(q.color)}; color:${col(q.color)}; border-radius:5px; padding:0 0.35em; font-family:var(--mono)` }, `“${q.text}”`))));
  const svg = S('svg', { viewBox: `${cx} ${cy} ${cw} ${ch}`, width, height: width * ch / cw, style: 'display:block; border-radius:6px; font-family: Inter, Arial, sans-serif' });
  svg.append(S('image', { href: (cfg.photos || '../../assets/lecture07/') + p.src, x: 0, y: 0, width: W, height: H }));
  if (p.overlay) svg.append(S('image', { href: dir + p.overlay, x: 0, y: 0, width: W, height: H }));
  for (const b of p.boxes || []) {
    const [x0, y0, x1, y1] = b.b, c = col(b.color || C.accent);
    svg.append(S('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: 'none', stroke: c, 'stroke-width': 2.5 * k, 'stroke-dasharray': b.dash ? `${8 * k} ${5 * k}` : 'none' }));
    if (b.label) {
      const t = cfg.scores === false ? b.label : `${b.score.toFixed(2)}`, fs = 11 * k, tw = t.length * fs * 0.6 + 6 * k;
      const ty = y0 - fs - 4 * k < 0 ? y1 : y0 - fs - 4 * k;
      svg.append(S('rect', { x: x0, y: ty, width: tw, height: fs + 4 * k, fill: c }));
      svg.append(S('text', { x: x0 + 3 * k, y: ty + fs, fill: '#111', 'font-size': fs, 'font-weight': 700 }, t));
    }
  }
  for (const q of p.points || []) {
    const r = 9 * k, c = q.pos ? C.green : C.red;
    svg.append(S('circle', { cx: q.x, cy: q.y, r, fill: c, stroke: '#fff', 'stroke-width': 2 * k }));
    svg.append(S('text', { x: q.x, y: q.y + 5 * k, fill: '#111', 'font-size': 15 * k, 'font-weight': 700, 'text-anchor': 'middle' }, q.pos ? '+' : '−'));
  }
  el.append(svg);
}
