// Non-maximum suppression on real detector output: Faster R-CNN on the crowd photo with NMS switched off
// (experiments/detection.py). Left: all boxes above the score threshold; the IoU threshold slider and the
// "step" button run greedy NMS here in the browser (per class): the kept box is drawn solid, the boxes it
// removed dashed, until they disappear.
// config: { src, data, width: 760, minScore: 0.5, thr: 0.5, show: "raw" | "nms", only: ["person"] }
import { C, h } from './util.js';
import { loadImage, loadJSON, photoCanvas, iou, freezeForPrint } from './imgutil.js';

export async function mount(el, cfg) {
  el.classList.add('widget');
  const [img, d] = await Promise.all([loadImage(cfg.src), loadJSON(cfg.data)]);
  const P = photoCanvas(img, cfg.width || 760, d.size);
  let boxes = d.boxes.filter(b => b.s >= (cfg.minScore ?? 0.5));
  if (cfg.only) boxes = boxes.filter(b => cfg.only.includes(b.c));
  const order = boxes.map((_, i) => i).sort((i, j) => boxes[j].s - boxes[i].s);
  const st = { thr: cfg.thr ?? 0.5, step: cfg.show === 'raw' ? 0 : Infinity };
  const readout = h('div', { class: 'readout', style: `min-height:1.4em; color:${C.fg}` });

  function run() {
    // greedy NMS, recording the order of decisions so that it can be shown step by step
    const res = boxes.map(() => null), steps = [];
    for (const i of order) {
      if (res[i]) continue;
      res[i] = { keep: true, at: steps.length };
      const removed = [];
      for (const j of order) if (!res[j] && boxes[j].c === boxes[i].c && iou(boxes[i].b, boxes[j].b) > st.thr) { res[j] = { keep: false, at: steps.length, by: i }; removed.push(j); }
      steps.push({ i, removed });
    }
    return { res, steps };
  }
  function draw() {
    const { res, steps } = run();
    const n = Math.min(st.step, steps.length);
    P.drawPhoto(0.8);
    const g = P.g, lw = 2 / P.k * P.dpr;
    // undecided boxes thin, kept solid, removed in the latest step dashed red
    boxes.forEach((b, i) => {
      const r = res[i], [x1, y1, x2, y2] = b.b;
      if (r.at < n - 1 || (r.at === n - 1 && r.keep)) {
        if (!r.keep) return;
        g.setLineDash([]); g.strokeStyle = C.accent; g.lineWidth = 1.6 * lw;
      } else if (r.at === n - 1) {
        g.setLineDash([6 / P.k * P.dpr, 4 / P.k * P.dpr]); g.strokeStyle = C.red; g.lineWidth = lw;
      } else { g.setLineDash([]); g.strokeStyle = 'rgba(90,176,255,0.85)'; g.lineWidth = 0.7 * lw; }
      g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    g.setLineDash([]);
    if (n > 0 && n <= steps.length && st.step !== Infinity) {
      const s = steps[n - 1], b = boxes[s.i];
      readout.textContent = `step ${n}: keep ${b.c} ${b.s.toFixed(2)} (highest left), remove ${s.removed.length} box${s.removed.length === 1 ? '' : 'es'} with IoU > ${st.thr.toFixed(2)}`;
    } else {
      const kept = res.filter(r => r.keep).length;
      readout.textContent = n === 0 ? `${boxes.length} boxes with score ≥ ${(cfg.minScore ?? 0.5).toFixed(1)}, before NMS`
        : `${boxes.length} boxes → ${kept} after NMS (IoU threshold ${st.thr.toFixed(2)})`;
    }
    bRaw.classList.toggle('active', st.step === 0); bAll.classList.toggle('active', st.step === Infinity);
  }
  const slider = h('input', { type: 'range', min: 0.1, max: 0.95, step: 0.05, value: st.thr, oninput: () => { st.thr = +slider.value; thrv.textContent = st.thr.toFixed(2); draw(); } });
  const thrv = h('span', { class: 'readout', style: 'display:inline-block; width:2.6em' }, st.thr.toFixed(2));
  const bRaw = h('button', { onclick: () => { st.step = 0; draw(); } }, 'before NMS');
  const bStep = h('button', { onclick: () => { st.step = st.step === Infinity ? 1 : st.step + 1; draw(); } }, 'step ▶');
  const bAll = h('button', { onclick: () => { st.step = Infinity; draw(); } }, 'after NMS');
  el.append(P.cv, readout, h('div', { class: 'wctl interactive-only' }, bRaw, bStep, bAll, h('label', { style: 'margin-left:1em' }, 'IoU threshold'), slider, thrv));
  draw();
  freezeForPrint(P.cv);
}
