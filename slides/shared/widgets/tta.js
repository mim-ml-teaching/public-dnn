// Test-time augmentation as a diagram, from our experiment (experiments/tta.py):
// one test image → 10 copies (4 corner crops + centre, and their mirror images) → the same network on every
// copy (prediction under each copy) → the average of the 10 predictions.
// config: { src: "../../assets/lecture06/bison.jpg", data: "../../assets/lecture06/data/tta_bison.json",
//           tile: 92, orig: 190 }
import { C, h } from './util.js';

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [d, img] = await Promise.all([fetch(cfg.data).then(r => r.json()), (async () => { const i = new Image(); i.src = cfg.src; await i.decode(); return i; })()]);
  const T = cfg.tile || 92, sc = img.width / d.size[0];
  const crops = [...d.crops.filter(c => !c.flip), ...d.crops.filter(c => c.flip)];

  // the test image with the 5 crop windows
  const OW = cfg.orig || 190, OH = Math.round(OW * d.size[1] / d.size[0]), os = OW / d.size[0];
  const oc = h('canvas'); oc.width = OW; oc.height = OH; oc.style.width = OW + 'px'; oc.style.height = OH + 'px'; oc.style.borderRadius = '4px';
  const og = oc.getContext('2d');
  og.drawImage(img, 0, 0, OW, OH);
  og.strokeStyle = C.accent; og.lineWidth = 1.5; og.setLineDash([5, 3]);
  crops.filter(c => !c.flip).forEach(c => og.strokeRect(c.box[0] * os + 1, c.box[1] * os + 1, c.box[2] * os - 2, c.box[3] * os - 2));

  const tiles = crops.map(c => {
    const cv = h('canvas'); cv.width = T; cv.height = T; cv.style.width = T + 'px'; cv.style.height = T + 'px'; cv.style.borderRadius = '3px';
    const g = cv.getContext('2d');
    g.save(); if (c.flip) { g.translate(T, 0); g.scale(-1, 1); }
    g.drawImage(img, c.box[0] * sc, c.box[1] * sc, c.box[2] * sc, c.box[3] * sc, 0, 0, T, T);
    g.restore();
    const ok = c.top1 === d.true, color = ok ? C.green : C.red;
    return h('div', { style: `width:${T}px` }, cv,
      h('div', { style: `height:6px; background:${C.bg3}; border-radius:3px; margin-top:2px` }, h('div', { style: `height:6px; width:${Math.round(100 * c.p_true)}%; background:${C.green}; border-radius:3px` })),
      h('div', { class: 'small', style: `line-height:1.15; font-size:0.5em; color:${color}; white-space:nowrap` }, ok ? d.true : c.top1, h('br'), `p(${d.true}) = ${c.p_true.toFixed(2)}`));
  });
  const arrow = (top, bottom = '') => h('div', { style: 'display:flex; flex-direction:column; align-items:center; text-align:center; min-width:4.6em' },
    h('div', { class: 'small', style: `color:${C.accent}; line-height:1.15` }, top),
    h('div', { style: `font-size:1.6em; line-height:1; color:${C.dim}` }, '⟶'),
    h('div', { class: 'small', style: 'line-height:1.15' }, bottom));
  const avg = d.avg_top.map(([name, p]) => h('div', { style: 'display:flex; align-items:center; gap:0.4em; margin:0.2em 0' },
    h('div', { style: 'width:5.6em; text-align:right; white-space:nowrap' }, name),
    h('div', { style: `width:${Math.round(90 * p)}px; height:16px; background:${name === d.true ? C.green : C.dim}; border-radius:3px` }),
    h('span', { class: 'readout' }, p.toFixed(2))));

  el.append(h('div', { style: 'display:flex; gap:0.4em; align-items:center' },
    h('div', { style: 'display:flex; flex-direction:column; align-items:center; gap:0.2em' }, oc,
      h('div', { class: 'small', style: `color:${C.accent}; text-align:center; line-height:1.2` }, 'one test image', h('br'), h('span', { style: `color:${C.dim}` }, '5 crop windows'))),
    arrow('10 copies', '5 crops × mirror'),
    h('div', { style: 'display:flex; flex-direction:column; gap:0.3em' },
      h('div', { class: 'small', style: 'text-align:center' }, 'the same network ', h('i', {}, 'f'), h('sub', {}, 'θ'), ' on every copy'),
      h('div', { style: `display:grid; grid-template-columns:repeat(5, ${T}px); gap:6px 8px` }, ...tiles)),
    arrow('average', 'of the 10'),
    h('div', { style: 'font-size:0.8em; width:11em; flex:none' }, h('div', { style: `color:${C.accent}; font-weight:600; margin-bottom:0.2em` }, 'averaged prediction'), ...avg)));
}
