// A photo with detections and/or masks from our experiments (Mask R-CNN, Faster R-CNN, SAM).
// config: {
//   src, data,                 photo and JSON with "objects": [{ b: [x1,y1,x2,y2], c or name, s, mask? }]
//                              or "boxes" (raw Faster R-CNN output; then nms: 0.5 is applied here)
//   task: "boxes" | "instance" | "semantic" | "classify" | "masks"  (masks = instance masks without boxes)
//   minScore: 0, nms: null, width: 560, classOf: true (strip a trailing number: "pear 2" → "pear"),
//   label: true (class tag on boxes), scores: true, caption: "pear" (classify: the label to show),
//   colors: { "pear": "#5fd38d", ... }, noMask: ["dining table"] (classes drawn as a box only)
// }
import { C, h } from './util.js';
import { loadImage, loadJSON, photoCanvas, decodeMask, maskLayer, hex2rgb, tag, nms, freezeForPrint } from './imgutil.js';
import { mapHex, LIGHT } from '../theme.js';

const INST = [C.accent, C.blue, C.green, C.red, C.purple, '#ff9f5a', '#5ad1c8', '#e48bd6'];

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [img, d] = await Promise.all([loadImage(cfg.src), loadJSON(cfg.data)]);
  const P = photoCanvas(img, cfg.width || 560, d.size);
  const classOf = o => {
    const n = o.c || o.name;
    return cfg.classOf === false ? n : n.replace(/\s+\d+$/, '');
  };
  let objs = (d.objects || d.boxes).filter(o => o.s === undefined || o.s >= (cfg.minScore || 0));
  if (cfg.nms) { const r = nms(objs, cfg.nms); objs = objs.filter((_, i) => r[i].keep); }
  if (cfg.only) objs = objs.filter(o => cfg.only.includes(classOf(o)));
  const task = cfg.task || 'boxes';
  const ccol = name => (cfg.colors && cfg.colors[name]) || C.accent;
  P.drawPhoto(task === 'semantic' || task === 'masks' || task === 'instance' ? 0.6 : 1);
  const g = P.g;
  objs.forEach((o, i) => {
    const col = task === 'semantic' || task === 'boxes' ? ccol(classOf(o)) : INST[i % INST.length];
    if (o.mask && !(cfg.noMask || []).includes(classOf(o)) && (task === 'semantic' || task === 'instance' || task === 'masks')) {
      const m = decodeMask(o.mask);
      g.drawImage(maskLayer(m, o.mask.w, o.mask.h, hex2rgb(LIGHT ? mapHex(col) : col), 0.55, task !== 'semantic'), 0, 0, d.size[0], d.size[1]);
    }
    if (task === 'boxes' || task === 'instance') {
      const [x1, y1, x2, y2] = o.b;
      g.strokeStyle = col; g.lineWidth = 3 / P.k * P.dpr; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if (cfg.label !== false) tag(g, x1, y1, classOf(o) + (cfg.scores !== false && o.s !== undefined && o.s < 1.5 ? ` ${o.s.toFixed(2)}` : ''), col, P.k, cfg.tagSize || 12, P.dpr);
    }
  });
  if (task === 'classify') {
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0);
    const u = P.dpr / 2, fs = P.dpr * (cfg.tagSize || 16); g.font = `600 ${fs}px Inter, Arial, sans-serif`;
    const t = cfg.caption || ''; const tw = g.measureText(t).width + 20 * u;
    g.fillStyle = C.accent; g.fillRect(12 * u, 12 * u, tw, fs + 14 * u); g.fillStyle = '#15171c'; g.fillText(t, 22 * u, 12 * u + fs + 2 * u);
    g.restore();
  }
  el.append(P.cv);
  freezeForPrint(P.cv);
}
