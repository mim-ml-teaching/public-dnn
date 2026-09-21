// Which anchor learns which object? Anchors (sizes × aspect ratios) sit at every position of a grid (one position =
// one pixel of the feature map, `stride` image pixels apart). Click a position: its anchors are coloured by their
// best IoU with a true box: ≥ 0.5 object (green; trained to predict that box's class and the offsets to it),
// < 0.4 background (red), in between ignored (grey, no loss). Every true box also gets its single best anchor,
// even below 0.5. "all positive anchors" shows every positive anchor in the image and the counts.
// True boxes: a detector's confident boxes (score ≥ minScore) standing in for annotations.
// config: { src, data, width: 720, stride: 32, sizes: [64,128,256], ratios: [0.5,1,2] (height / width),
//           classes: ["person","car"], minScore: 0.9, at: [x, y] (image pixels, initial click), all: false,
//           zoom: false (side panel: zoom on the clicked position + what each anchor is trained to predict), zoomWidth: 300,
//           pairs: false (third panel: each matched anchor with its object and offsets), pairWidth: 220 }
import { C, h } from './util.js';
import { loadImage, loadJSON, photoCanvas, iou, nms, freezeForPrint } from './imgutil.js';

const CLS = { person: C.red, car: C.blue, bus: C.accent, truck: C.purple };

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [img, d] = await Promise.all([loadImage(cfg.src), loadJSON(cfg.data)]);
  const P = photoCanvas(img, cfg.width || 720, d.size);
  P.cv.style.cursor = 'pointer';
  const [W, H] = d.size, s = cfg.stride || 32;
  const sizes = cfg.sizes || [64, 128, 256], ratios = cfg.ratios || [0.5, 1, 2];
  const classes = cfg.classes || ['person'];
  let gts = d.boxes.filter(b => classes.includes(b.c) && b.s >= (cfg.minScore ?? 0.9));
  const kept = nms(gts, 0.5); gts = gts.filter((_, i) => kept[i].keep);
  // names for the objects, numbered per class from left to right: "person 3"
  const perClass = {};
  [...gts].sort((a, b) => a.b[0] - b.b[0]).forEach(t => { perClass[t.c] = (perClass[t.c] || 0) + 1; t.name = `${t.c} ${perClass[t.c]}`; });
  const cols = Math.floor(W / s), rows = Math.floor(H / s), K = sizes.length * ratios.length;
  const anchorsAt = (c, r) => {
    const cx = (c + 0.5) * s, cy = (r + 0.5) * s, out = [];
    for (const z of sizes) for (const q of ratios) { const w = z / Math.sqrt(q), hh = z * Math.sqrt(q); out.push([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2]); }
    return out;
  };
  // labels for all anchors (index = (r * cols + c) * K + k), with the best-anchor rule
  const all = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) anchorsAt(c, r).forEach(a => {
    let v = 0, gi = -1; gts.forEach((g, i) => { const u = iou(a, g.b); if (u > v) { v = u; gi = i; } });
    all.push({ a, v, gi, kind: v >= 0.5 ? 'pos' : v < 0.4 ? 'neg' : 'ign', forced: false });
  });
  gts.forEach((g, i) => {
    let bi = 0, bv = -1; all.forEach((x, j) => { const u = iou(x.a, g.b); if (u > bv) { bv = u; bi = j; } });
    if (all[bi].kind !== 'pos') { all[bi].kind = 'pos'; all[bi].gi = i; all[bi].v = bv; all[bi].forced = true; }
  });
  const npos = all.filter(x => x.kind === 'pos').length;
  const at = cfg.at || [592, 528];
  const st = { c: Math.floor(at[0] / s), r: Math.floor(at[1] / s), all: !!cfg.all };
  const readout = h('div', { class: 'readout', style: `min-height:2.7em; color:${C.fg}; line-height:1.35; white-space:pre-line` });
  const COL = { pos: C.green, neg: C.red, ign: '#9aa1ae' };
  const off = (a, t) => {
    const [aw, ah, ax, ay] = [a[2] - a[0], a[3] - a[1], (a[0] + a[2]) / 2, (a[1] + a[3]) / 2];
    const [tw, th, tx, ty] = [t[2] - t[0], t[3] - t[1], (t[0] + t[2]) / 2, (t[1] + t[3]) / 2];
    return [(tx - ax) / aw, (ty - ay) / ah, Math.log(tw / aw), Math.log(th / ah)].map(v => v.toFixed(2)).join(', ');
  };
  // side panel: a zoom on the clicked position with only its anchors and the true boxes they overlap, and a table
  // of what each anchor is trained to predict
  const zw = cfg.zoomWidth || 300, dpr = new URLSearchParams(location.search).has('print-pdf') ? 1 : 2;
  const zc = h('canvas', { style: `width:${zw}px; height:${zw}px; display:block; border-radius:6px` });
  zc.width = zc.height = zw * dpr;
  const table = h('div', { class: 'readout', style: `font-size:0.72em; line-height:1.3; margin-top:0.3em; color:${C.fg}` });
  const side = h('div', { style: `width:${zw + 40}px; flex:none` }, zc, table);
  function drawZoom(mine) {
    const zg = zc.getContext('2d');
    zg.setTransform(1, 0, 0, 1, 0, 0); zg.clearRect(0, 0, zc.width, zc.height);
    if (!mine) { table.textContent = 'click a position on the photo'; return; }
    const rel = [...new Set(mine.filter(x => x.v >= 0.1 && x.gi >= 0).map(x => x.gi))].map(i => gts[i]);
    let focus = mine.filter(x => x.kind !== 'neg');
    if (!focus.length) focus = [mine.reduce((p, x) => (x.v > p.v ? x : p))];
    const fgt = [...new Set(focus.filter(x => x.gi >= 0).map(x => x.gi))].map(i => gts[i]);
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const b of [...focus.map(x => x.a), ...fgt.map(t => t.b)]) { x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]); }
    const side_ = Math.max(x1 - x0, y1 - y0) * 1.5, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const sx = cx - side_ / 2, sy = cy - side_ / 2, k = zc.width / side_;
    zg.fillStyle = '#111'; zg.fillRect(0, 0, zc.width, zc.height);
    zg.globalAlpha = 0.55; zg.drawImage(img, sx * img.width / W, sy * img.height / H, side_ * img.width / W, side_ * img.height / H, 0, 0, zc.width, zc.height); zg.globalAlpha = 1;
    zg.setTransform(k, 0, 0, k, -sx * k, -sy * k);
    const lw = 2.2 / k * dpr;
    zg.setLineDash([7 / k * dpr, 5 / k * dpr]); zg.lineWidth = lw * 1.2;
    rel.forEach(t => { zg.strokeStyle = '#ffffff'; zg.strokeRect(t.b[0], t.b[1], t.b[2] - t.b[0], t.b[3] - t.b[1]); });
    zg.setLineDash([]);
    zg.font = `600 ${13 / k * dpr}px Inter, Arial, sans-serif`; zg.fillStyle = '#ffffff';
    rel.forEach(t => zg.fillText(t.name, t.b[0] + 3 / k * dpr, t.b[3] - 5 / k * dpr));
    const order = mine.map((x, i) => i).sort((i, j) => (mine[i].kind === 'pos') - (mine[j].kind === 'pos') || mine[i].v - mine[j].v);
    order.forEach(i => { const x = mine[i]; zg.globalAlpha = x.kind === 'neg' ? 0.45 : 1; zg.strokeStyle = COL[x.kind]; zg.lineWidth = (x.kind === 'pos' ? 2.2 : x.kind === 'ign' ? 1.6 : 0.8) * lw; zg.strokeRect(x.a[0], x.a[1], x.a[2] - x.a[0], x.a[3] - x.a[1]); });
    zg.globalAlpha = 1;
    zg.fillStyle = C.accent; zg.beginPath(); zg.arc((mine[0].a[0] + mine[0].a[2]) / 2, (mine[0].a[1] + mine[0].a[3]) / 2, 5 / k * dpr, 0, 7); zg.fill();
    const rows = [...mine].sort((p, q) => q.v - p.v).map(x => {
      const w = Math.round(x.a[2] - x.a[0]), hh = Math.round(x.a[3] - x.a[1]);
      const what = x.kind === 'pos' ? `→ ${gts[x.gi].name}${x.forced ? ' (best)' : ''}` : x.kind === 'ign' ? 'nothing (ignored)' : '“background”';
      return h('div', { style: `display:grid; grid-template-columns: 5.2em 3.4em 1fr; color:${COL[x.kind]}` },
        h('span', {}, `${w}×${hh}`), h('span', {}, x.v.toFixed(2)), h('span', {}, what));
    });
    table.replaceChildren(h('div', { style: `display:grid; grid-template-columns: 5.2em 3.4em 1fr; color:${C.dim}` }, h('span', {}, 'anchor'), h('span', {}, 'IoU'), h('span', {}, 'learns')), ...rows);
    if (cfg.pairs) drawPairs(mine.filter(x => x.kind === 'pos').sort((p, q) => q.v - p.v));
  }
  // third panel: every matched anchor with the true box it learns, and the offsets
  const pw = cfg.pairWidth || 220;
  const pairsEl = h('div', { style: `width:${pw}px; flex:none; display:flex; flex-direction:column; gap:0.5em` });
  function drawPairs(pos) {
    pairsEl.replaceChildren(h('div', { class: 'small', style: `color:${C.dim}` }, 'each matched anchor and its object'));
    if (!pos.length) { pairsEl.append(h('div', { class: 'readout', style: `font-size:0.72em; color:${C.red}` }, 'none here: every anchor learns “background”')); return; }
    for (const x of pos.slice(0, 3)) {
      const t = gts[x.gi], cv = h('canvas', { style: `width:${pw}px; height:${Math.round(pw * 0.7)}px; display:block; border-radius:6px` });
      cv.width = pw * dpr; cv.height = Math.round(pw * 0.7) * dpr;
      const pg = cv.getContext('2d');
      const bx0 = Math.min(x.a[0], t.b[0]), by0 = Math.min(x.a[1], t.b[1]), bx1 = Math.max(x.a[2], t.b[2]), by1 = Math.max(x.a[3], t.b[3]);
      const sc = Math.min(cv.width / ((bx1 - bx0) * 1.3), cv.height / ((by1 - by0) * 1.3)), cx = (bx0 + bx1) / 2, cy = (by0 + by1) / 2;
      const sx = cx - cv.width / sc / 2, sy = cy - cv.height / sc / 2;
      pg.fillStyle = '#111'; pg.fillRect(0, 0, cv.width, cv.height);
      pg.globalAlpha = 0.55; pg.drawImage(img, sx * img.width / W, sy * img.height / H, cv.width / sc * img.width / W, cv.height / sc * img.height / H, 0, 0, cv.width, cv.height); pg.globalAlpha = 1;
      pg.setTransform(sc, 0, 0, sc, -sx * sc, -sy * sc);
      const lw = 2.2 / sc * dpr;
      pg.setLineDash([7 / sc * dpr, 5 / sc * dpr]); pg.strokeStyle = '#ffffff'; pg.lineWidth = lw * 1.1; pg.strokeRect(t.b[0], t.b[1], t.b[2] - t.b[0], t.b[3] - t.b[1]); pg.setLineDash([]);
      pg.strokeStyle = C.green; pg.lineWidth = lw * 1.3; pg.strokeRect(x.a[0], x.a[1], x.a[2] - x.a[0], x.a[3] - x.a[1]);
      const ax = (x.a[0] + x.a[2]) / 2, ay = (x.a[1] + x.a[3]) / 2, tx = (t.b[0] + t.b[2]) / 2, ty = (t.b[1] + t.b[3]) / 2;
      pg.strokeStyle = C.accent; pg.fillStyle = C.accent; pg.lineWidth = lw;
      pg.beginPath(); pg.moveTo(ax, ay); pg.lineTo(tx, ty); pg.stroke();
      pg.beginPath(); pg.arc(ax, ay, 4 / sc * dpr, 0, 7); pg.fill();
      pg.fillStyle = '#ffffff'; pg.beginPath(); pg.arc(tx, ty, 4 / sc * dpr, 0, 7); pg.fill();
      pg.font = `600 ${13 / sc * dpr}px Inter, Arial, sans-serif`; pg.fillText(t.name, t.b[0] + 3 / sc * dpr, t.b[3] - 5 / sc * dpr);
      const w = Math.round(x.a[2] - x.a[0]), hh = Math.round(x.a[3] - x.a[1]);
      pairsEl.append(h('div', {}, cv, h('div', { class: 'readout', style: `font-size:0.6em; line-height:1.3; color:${C.fg}` },
        h('span', { style: `color:${C.green}` }, `${w}×${hh}`), ` → ${t.name}`, h('br'), `IoU ${x.v.toFixed(2)}`, h('br'), `t = (${off(x.a, t.b)})`)));
      freezeForPrint(cv);
    }
  }
  function draw() {
    P.drawPhoto(0.5);
    const g = P.g, lw = 2 / P.k * P.dpr;
    g.strokeStyle = 'rgba(255,255,255,0.18)'; g.lineWidth = lw * 0.4;
    for (let c = 0; c <= cols; c++) { g.beginPath(); g.moveTo(c * s, 0); g.lineTo(c * s, H); g.stroke(); }
    for (let r = 0; r <= rows; r++) { g.beginPath(); g.moveTo(0, r * s); g.lineTo(W, r * s); g.stroke(); }
    g.setLineDash([6 / P.k * P.dpr, 4 / P.k * P.dpr]); g.lineWidth = lw * 1.1;
    gts.forEach(t => { g.strokeStyle = '#ffffff'; g.strokeRect(t.b[0], t.b[1], t.b[2] - t.b[0], t.b[3] - t.b[1]); });
    g.setLineDash([]);
    if (st.all) {
      all.forEach(x => { if (x.kind !== 'pos') return; g.strokeStyle = C.green; g.lineWidth = lw * 0.8; g.strokeRect(x.a[0], x.a[1], x.a[2] - x.a[0], x.a[3] - x.a[1]); });
      if (cfg.zoom) { drawZoom(null); readout.style.minHeight = '2.7em'; }
      readout.textContent = `${all.length.toLocaleString('en')} anchors (${cols}×${rows} positions × ${K}), ${gts.length} objects\n` +
        `${npos} positive (${(100 * npos / all.length).toFixed(1)} %), the rest background`;
      return;
    }
    const base = (st.r * cols + st.c) * K, mine = all.slice(base, base + K);
    mine.forEach(x => { g.strokeStyle = COL[x.kind]; g.lineWidth = (x.kind === 'pos' ? 1.8 : 1) * lw; g.strokeRect(x.a[0], x.a[1], x.a[2] - x.a[0], x.a[3] - x.a[1]); });
    g.fillStyle = C.accent; g.beginPath(); g.arc((st.c + 0.5) * s, (st.r + 0.5) * s, 5 / P.k * P.dpr, 0, 7); g.fill();
    if (cfg.zoom) { drawZoom(mine); readout.textContent = ''; readout.style.minHeight = '0'; return; }
    const pos = mine.filter(x => x.kind === 'pos').length, ign = mine.filter(x => x.kind === 'ign').length;
    const b = mine.reduce((p, x) => (x.v > p.v ? x : p));
    let txt = `${K} anchors here: ${pos} object, ${ign} ignored, ${K - pos - ign} background`;
    txt += b.kind === 'pos' && b.gi >= 0
      ? `\nbest: ${gts[b.gi].c}, IoU ${b.v.toFixed(2)}${b.forced ? ' (its best)' : ''}, t = (${off(b.a, gts[b.gi].b)})`
      : `\nbest IoU ${b.v.toFixed(2)}: “background”, no box`;
    readout.textContent = txt;
  }
  P.cv.addEventListener('click', e => {
    const r = P.cv.getBoundingClientRect();
    st.c = Math.min(cols - 1, Math.floor((e.clientX - r.left) / r.width * W / s)); st.r = Math.min(rows - 1, Math.floor((e.clientY - r.top) / r.height * H / s));
    st.all = false; bAll.classList.remove('active'); draw();
  });
  const bAll = h('button', { onclick: () => { st.all = !st.all; bAll.classList.toggle('active', st.all); draw(); } }, 'all positive anchors');
  bAll.classList.toggle('active', st.all);
  const legend = h('div', { class: 'small', style: 'display:flex; flex-wrap:wrap; gap:0.3em 1.1em; margin-top:0.25em' },
    ...[['object (IoU ≥ 0.5)', C.green, 'solid'], ['ignored', '#9aa1ae', 'solid'], ['background (IoU < 0.4)', C.red, 'solid'], ['true boxes', '#ffffff', 'dashed']].map(([t, c, ls]) =>
      h('span', {}, h('span', { style: `display:inline-block; width:1.4em; height:0; border-top:3px ${ls} ${c}; vertical-align:middle; margin-right:0.3em` }), t)));
  const main = h('div', {}, P.cv, legend, readout, h('div', { class: 'wctl interactive-only' }, h('span', { class: 'dim' }, 'click a position'), bAll));
  if (cfg.zoom) el.append(h('div', { style: 'display:flex; gap:1.0em; align-items:flex-start' }, main, side, ...(cfg.pairs ? [pairsEl] : [])));
  else el.append(main);
  draw();
  freezeForPrint(P.cv);
  if (cfg.zoom) freezeForPrint(zc);
}
