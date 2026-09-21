// A photo and a semantic segmentation of it (one class per pixel), from experiments/segmentation.py.
// Buttons switch between the photo, the prediction at full resolution, and a coarse one (logits at 1/32,
// upsampled), so the loss of detail is visible.
// config: { src: "...street.jpg", data: "...street_seg.json", fine: "...fine.png", coarse: "...coarse.png",
//           mid: "...mid.png" (optional, 1/16), width: 640, show: "fine" | "mid" | "coarse" | "photo", buttons: true, legend: true,
//           views: ["photo", "coarse"] (which buttons) }
import { C, h } from './util.js';
import { loadImage, loadJSON, photoCanvas, hex2rgb, freezeForPrint } from './imgutil.js';
import { mapHex, LIGHT } from '../theme.js';

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [img, d, fine, coarse, mid] = await Promise.all([loadImage(cfg.src), loadJSON(cfg.data), loadImage(cfg.fine), cfg.coarse ? loadImage(cfg.coarse) : null, cfg.mid ? loadImage(cfg.mid) : null]);
  const P = photoCanvas(img, cfg.width || 640, d.size);
  const labels = {};
  for (const [key, im] of Object.entries({ fine, coarse, mid })) {
    if (!im) continue;
    const c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
    const g = c.getContext('2d'); g.drawImage(im, 0, 0);
    labels[key] = g.getImageData(0, 0, im.width, im.height).data;
  }
  const color = {};
  d.classes.forEach(c => { if (c.id !== 0) color[c.id] = hex2rgb(LIGHT ? mapHex(c.color) : c.color); });
  let mode = cfg.show || 'fine';
  const layer = key => {
    const L = labels[key], W = d.size[0], H = d.size[1];
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d'), im = g.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const col = color[L[4 * i]];
      if (!col) continue;
      im.data[4 * i] = col[0]; im.data[4 * i + 1] = col[1]; im.data[4 * i + 2] = col[2]; im.data[4 * i + 3] = 165;
    }
    g.putImageData(im, 0, 0);
    return c;
  };
  const cache = {};
  function draw() {
    P.drawPhoto(mode === 'photo' ? 1 : 0.55);
    if (mode !== 'photo') P.g.drawImage(cache[mode] ||= layer(mode), 0, 0);
    bs.forEach(b => b.classList.toggle('active', b.dataset.m === mode));
  }
  const names = { photo: 'photo', fine: `prediction (logits at 1/8)`, mid: 'logits at 1/16, upsampled', coarse: 'logits at 1/32, upsampled' };
  const bs = (cfg.views || ['photo', 'fine', ...(coarse ? ['coarse'] : [])]).map(m => { const b = h('button', { onclick: () => { mode = m; draw(); } }, names[m]); b.dataset.m = m; return b; });
  const legend = h('div', { style: 'display:flex; flex-wrap:wrap; gap:0.3em 1em; margin-top:0.35em' },
    d.classes.filter(c => c.id !== 0).map(c => h('span', {}, h('span', { style: `display:inline-block; width:0.9em; height:0.9em; border-radius:2px; vertical-align:-0.1em; margin-right:0.3em; background:${c.color}` }), c.name)),
    h('span', { class: 'dim' }, 'uncoloured: background'));
  el.append(P.cv);
  if (cfg.legend !== false) el.append(legend);
  if (cfg.buttons !== false) el.append(h('div', { class: 'wctl interactive-only' }, ...bs));
  draw();
  freezeForPrint(P.cv);
}
