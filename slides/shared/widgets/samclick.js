// Segment Anything, precomputed (experiments/sam_points.py): click anywhere on the photo; the widget shows the
// three masks SAM returned for the nearest click of a 24-px grid, with SAM's own quality estimate (predicted IoU).
// All three are shown as thumbnails under the photo (click one to show it large); the most confident is preselected.
// config: { src, data, width: 640, x: 396, y: 324 (initial click, photo pixels), pick: 0..2 }
import { C, h } from './util.js';
import { loadImage, loadJSON, photoCanvas, decodeMask, maskLayer, hex2rgb, freezeForPrint } from './imgutil.js';
import { mapHex, LIGHT } from '../theme.js';

const COLS = [C.blue, C.green, C.accent];

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [img, d] = await Promise.all([loadImage(cfg.src), loadJSON(cfg.data)]);
  const P = photoCanvas(img, cfg.width || 640, d.size);
  P.cv.style.cursor = 'crosshair';
  const st = { x: cfg.x ?? 396, y: cfg.y ?? 324, pick: cfg.pick ?? null };
  const cache = new Map();
  const layer = (mi, col) => {
    const key = mi + col;
    if (!cache.has(key)) { const m = d.masks[mi]; cache.set(key, maskLayer(decodeMask(m), m.w, m.h, hex2rgb(LIGHT ? mapHex(col) : col), 0.5)); }
    return cache.get(key);
  };
  const tw = Math.floor(((cfg.width || 640) - 2 * 10) / 3);
  const thumbs = [0, 1, 2].map(() => {
    const T = photoCanvas(img, tw, d.size), cap = h('button', { style: 'width:100%; margin:0.2em 0 0; font-size:0.8em; padding:0.15em 0.2em' });
    const box = h('div', { style: `width:${tw}px; cursor:pointer` }, T.cv, cap);
    return { T, cap, box };
  });
  const tabs = h('div', { style: 'display:flex; gap:10px; margin-top:0.35em' }, ...thumbs.map(t => t.box));
  function nearest() {
    let best = null, bd = Infinity;
    for (const p of d.points) { const dd = (p.x - st.x) ** 2 + (p.y - st.y) ** 2; if (dd < bd) { bd = dd; best = p; } }
    return best;
  }
  function draw() {
    const p = nearest();
    // order the three masks by area (sub-part, part, whole) for a stable reading
    const area = mi => d.masks[mi].rle.reduce((a, n, i) => a + (i % 2 ? n : 0), 0);
    const idx = [0, 1, 2].sort((a, b) => area(p.m[a]) - area(p.m[b]));
    if (st.pick === null) st.pick = idx.reduce((a, b) => (p.s[b] > p.s[a] ? b : a));
    P.drawPhoto(0.7);
    P.g.drawImage(layer(p.m[st.pick], COLS[idx.indexOf(st.pick)]), 0, 0, d.size[0], d.size[1]);
    const g = P.g, r = 9;
    g.fillStyle = C.green; g.strokeStyle = '#ffffff'; g.lineWidth = 2.5;
    g.beginPath(); g.arc(p.x, p.y, r, 0, 7); g.fill(); g.stroke();
    idx.forEach((i, k) => {
      const { T, cap, box } = thumbs[k];
      T.drawPhoto(0.6);
      T.g.drawImage(layer(p.m[i], COLS[k]), 0, 0, d.size[0], d.size[1]);
      T.g.fillStyle = C.green; T.g.strokeStyle = '#ffffff'; T.g.lineWidth = 6;
      T.g.beginPath(); T.g.arc(p.x, p.y, 18, 0, 7); T.g.fill(); T.g.stroke();
      T.cv.style.outline = i === st.pick ? `3px solid ${COLS[k]}` : 'none';
      cap.replaceChildren(h('span', { style: `display:inline-block; width:0.8em; height:0.8em; border-radius:2px; margin-right:0.35em; background:${COLS[k]}` }),
        `mask ${k + 1}: IoU ${p.s[i].toFixed(2)}`);
      cap.classList.toggle('active', i === st.pick);
      box.onclick = () => { st.pick = i; draw(); };
    });
  }
  P.cv.addEventListener('click', e => {
    const r = P.cv.getBoundingClientRect();
    st.x = (e.clientX - r.left) / r.width * d.size[0]; st.y = (e.clientY - r.top) / r.height * d.size[1]; st.pick = null; draw();
  });
  el.append(P.cv, tabs);
  draw();
  freezeForPrint(P.cv);
  thumbs.forEach(t => freezeForPrint(t.T.cv));
}
