// Conditional generation as a family of distributions. Four panels: the data distribution (a cloud with a cluster of
// dogs, cats, landscapes and other images), then the conditional distribution for each of three prompts: the same
// cloud, re-weighted so that only the images matching the prompt remain. Captions (formulas) are in the slide HTML,
// in columns of the same width. config: { size: 240, gap: 24, n: 110, groups: ['dog', 'cat', 'landscape'] (which prompts to show) }
import { C, h } from './util.js';
import { rng, gauss } from './flowlib.js';

const GROUPS = [
  { name: 'dog', color: '#f2b134', mean: [-1.5, 0.9], sd: [0.55, 0.45] },
  { name: 'cat', color: '#5ab0ff', mean: [1.4, 1.1], sd: [0.45, 0.5] },
  { name: 'landscape', color: '#5fd38d', mean: [0.1, -1.4], sd: [0.9, 0.4] },
];
const R = 3.2;

export function mount(el, cfg) {
  const S = cfg.size || 240, GAP = cfg.gap ?? 24, N = cfg.n || 110, groups = cfg.groups ? GROUPS.filter(gr => cfg.groups.includes(gr.name)) : GROUPS;
  const r = rng(21);
  const pts = [];
  groups.forEach((gr, k) => { for (let i = 0; i < N; i++) pts.push([gr.mean[0] + gr.sd[0] * gauss(r), gr.mean[1] + gr.sd[1] * gauss(r), k]); });
  for (let i = 0; i < 90; i++) pts.push([2.2 * (r() - 0.5) * 2.4, 2.2 * (r() - 0.5) * 2.4, -1]);   // other images
  const sx = x => (x + R) / (2 * R) * S, sy = y => S - (y + R) / (2 * R) * S;
  const cvs = [-1, ...groups.keys()].map(() => h('canvas', { width: S, height: S, style: `width:${S}px; height:${S}px; flex:none` }));
  cvs.forEach((cv, p) => {
    const g = cv.getContext('2d'), sel = p - 1;    // sel = −1: all data, otherwise the index of the prompt
    g.fillStyle = C.bg2; g.fillRect(0, 0, S, S);
    for (const [x, y, k] of pts) {
      const on = sel < 0 || k === sel;
      g.fillStyle = on ? (k < 0 ? '#7d8594' : groups[k].color) : 'rgba(125,133,148,0.18)';
      g.beginPath(); g.arc(sx(x), sy(y), on ? 3 : 2.2, 0, 7); g.fill();
    }
    const small = S < 180;   // small panels: one caption each, the group names would overlap
    g.font = `${small ? 10 : 13}px Inter, Arial`; g.textAlign = 'center';
    if (sel < 0) {
      if (small) { g.fillStyle = '#9aa1ae'; g.fillText('all images', S / 2, S - 8); }
      else { groups.forEach(gr => { g.fillStyle = gr.color; g.fillText(gr.name + 's', sx(gr.mean[0]), sy(gr.mean[1] + 1.1 * gr.sd[1] + 0.35)); }); g.fillStyle = '#7d8594'; g.fillText('other images', sx(-1.4), sy(-2.6)); }
    } else { const gr = groups[sel]; g.fillStyle = gr.color; g.font = small ? '10px Inter, Arial' : 'bold 16px Inter, Arial'; g.fillText(`y = "${gr.name}"`, S / 2, S - (small ? 8 : 12)); }
  });
  el.classList.add('widget');
  el.append(h('div', { style: `display:flex; gap:${GAP}px; justify-content:center` }, ...cvs));
}
