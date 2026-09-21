// Line chart(s) from a JSON file: one panel or small multiples sharing axes, hover crosshair + tooltip,
// direct end labels + legend, optional vertical annotation, table view.
// config: {
//   src: "../../assets/lecture01/data/x.json",
//   x: "epoch" | path, xlog: false, xlabel, ylabel, ymin, ymax, yfmt: "pct" | "num",
//   panels: [{ title, base: "true_labels" (object path inside the JSON, optional) }],   // default: one panel
//   series: [{ key: "train_acc", label: "train", color: "#3987e5", dash: false }],
//   vline: { x: 1000, label: "…" }, width, height, reveal: n   // reveal: show only the first n panels
// }
import { C, h, fmt } from './util.js';

const get = (o, path) => (path ? path.split('.').reduce((a, k) => a?.[k], o) : o);

export async function mount(el, cfg) {
  const data = await (await fetch(cfg.src)).json();
  const panels = cfg.panels || [{ title: null, base: null }];
  const W = cfg.width || 900, H = cfg.height || 330;
  const gap = 28, PW = (W - gap * (panels.length - 1)) / panels.length;
  const M = { l: 64, r: 104, t: (panels[0].title ? 36 : 14) + (cfg.brackets ? 62 : 0), b: 54 };
  const ns = 'http://www.w3.org/2000/svg';
  const S = (tag, attrs = {}, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const yf = v => (cfg.yfmt === 'pct' ? `${Math.round(v * 100)}%` : cfg.yfmt === 'sci' ? (v >= 1000 || v < 0.01 ? v.toExponential(0).replace('e+', 'e') : fmt(v, 2)) : fmt(v, 2));

  // shared scales
  const xs = get(data, (panels[0].base ? panels[0].base + '.' : '') + cfg.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ymin = cfg.ymin ?? 0, ymax = cfg.ymax ?? 1;
  const sx = v => (cfg.xlog ? (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) : (v - x0) / (x1 - x0)) * (PW - M.l - M.r) + M.l;
  const sy = cfg.ylog
    ? v => M.t + (1 - (Math.log10(Math.min(ymax, Math.max(ymin, v))) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin))) * (H - M.t - M.b)
    : v => M.t + (1 - (Math.min(ymax, Math.max(ymin, v)) - ymin) / (ymax - ymin)) * (H - M.t - M.b);
  const xticks = cfg.xlog
    ? Array.from({ length: 10 }, (_, k) => 10 ** k).filter(v => v >= x0 && v <= x1)
    : niceTicks(x0, x1, 6);
  const yticks = cfg.ylog
    ? Array.from({ length: 20 }, (_, k) => 10 ** (Math.ceil(Math.log10(ymin)) + k)).filter(v => v <= ymax)
    : niceTicks(ymin, ymax, 5);

  const svg = S('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: 'font-family: Inter, Arial, sans-serif; overflow: visible' });
  const tip = h('div', { class: 'viz-tip', style: 'position:absolute; pointer-events:none; display:none; background:#272b34; border:1px solid #3a3f4b; border-radius:6px; padding:0.3em 0.6em; font-size:0.85em; line-height:1.4; white-space:nowrap' });

  panels.forEach((pn, pi) => {
    const g = S('g', { transform: `translate(${pi * (PW + gap)},0)` });
    if (cfg.reveal !== undefined && pi >= cfg.reveal) g.setAttribute('class', 'fragment');
    svg.append(g);
    const base = pn.base ? pn.base + '.' : '';
    const X = get(data, base + cfg.x);
    if (pn.title) g.append(S('text', { x: M.l, y: 16, fill: C.fg, 'font-size': 20, 'font-weight': 600 }, pn.title));
    // grid + axes (recessive)
    yticks.forEach(t => {
      g.append(S('line', { x1: M.l, x2: PW - M.r, y1: sy(t), y2: sy(t), stroke: '#2d313a', 'stroke-width': 1 }));
      g.append(S('text', { x: M.l - 8, y: sy(t) + 4, fill: C.dim, 'font-size': 16, 'text-anchor': 'end' }, yf(t)));
    });
    xticks.forEach(t => g.append(S('text', { x: sx(t), y: H - M.b + 22, fill: C.dim, 'font-size': 16, 'text-anchor': 'middle' },
      cfg.xlog ? t.toLocaleString('en-US') : String(t))));
    g.append(S('line', { x1: M.l, x2: PW - M.r, y1: H - M.b, y2: H - M.b, stroke: '#5a606c' }));
    if (cfg.xlabel) g.append(S('text', { x: (M.l + PW - M.r) / 2, y: H - 6, fill: C.dim, 'font-size': 17, 'text-anchor': 'middle' }, cfg.xlabel));
    if (cfg.ylabel && pi === 0) g.append(S('text', { x: 16, y: (M.t + H - M.b) / 2, fill: C.dim, 'font-size': 17, 'text-anchor': 'middle', transform: `rotate(-90 16 ${(M.t + H - M.b) / 2})` }, cfg.ylabel));
    // brackets above the plot: { x0, x1, label: ["line 1", "line 2"] }
    (cfg.brackets || []).forEach(br => {
      const a = sx(br.x0) + 3, b = sx(br.x1) - 3, y = M.t - 10, m = (a + b) / 2;
      g.append(S('path', { d: `M${a},${y} q0,-8 8,-8 L${m - 8},${y - 8} q8,0 8,-8 q0,8 8,8 L${b - 8},${y - 8} q8,0 8,8`, fill: 'none', stroke: C.dim, 'stroke-width': 1.5 }));
      br.label.forEach((t, i) => g.append(S('text', { x: m, y: y - 46 + i * 17, fill: C.fg, 'font-size': 15, 'text-anchor': 'middle' }, t)));
    });
    if (cfg.band) {
      const bx0 = sx(cfg.band.x0), bx1 = sx(cfg.band.x1);
      g.append(S('rect', { x: bx0, y: M.t, width: bx1 - bx0, height: H - M.t - M.b, fill: '#f2b134', opacity: 0.12 }));
      if (cfg.band.label) g.append(S('text', { x: (bx0 + bx1) / 2, y: H - M.b - 10, fill: C.accent, 'font-size': 15, 'text-anchor': 'middle' }, cfg.band.label));
    }
    if (cfg.vline) {
      const vx = sx(cfg.vline.x);
      g.append(S('line', { x1: vx, x2: vx, y1: M.t, y2: H - M.b, stroke: C.dim, 'stroke-dasharray': '4 4' }));
      g.append(S('text', { x: vx + 6, y: M.t + 14, fill: C.dim, 'font-size': 16 }, cfg.vline.label));
    }
    // series
    const ends = [];
    cfg.series.forEach(se => {
      const Y = get(data, base + se.key);
      const d = X.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(Y[i]).toFixed(1)}`).join('');
      g.append(S('path', { d, fill: 'none', stroke: se.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-dasharray': se.dash ? '6 4' : 'none' }));
      ends.push({ y: sy(Y[Y.length - 1]), se, v: Y[Y.length - 1] });
    });
    // direct end labels, nudged apart
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 20) ends[i].y = ends[i - 1].y + 20;
    ends.forEach(e => g.append(S('text', { x: PW - M.r + 8, y: e.y + 4, fill: C.fg, 'font-size': 17 }, `${e.se.label} ${yf(e.v)}`)));
    // hover crosshair
    const cross = S('line', { y1: M.t, y2: H - M.b, stroke: C.dim, 'stroke-width': 1, visibility: 'hidden' });
    const dots = cfg.series.map(se => S('circle', { r: 4.5, fill: se.color, stroke: '#1e2128', 'stroke-width': 2, visibility: 'hidden' }));
    g.append(cross, ...dots);
    const hit = S('rect', { x: M.l, y: M.t, width: PW - M.l - M.r, height: H - M.t - M.b, fill: 'transparent', style: 'cursor: crosshair' });
    g.append(hit);
    hit.addEventListener('pointermove', ev => {
      const r = hit.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * (PW - M.l - M.r) + M.l;
      let best = 0;
      X.forEach((x, i) => { if (Math.abs(sx(x) - px) < Math.abs(sx(X[best]) - px)) best = i; });
      const cx = sx(X[best]);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      const rows = cfg.series.map((se, k) => {
        const v = get(data, base + se.key)[best];
        dots[k].setAttribute('cx', cx); dots[k].setAttribute('cy', sy(v)); dots[k].setAttribute('visibility', 'visible');
        return `<div><span style="display:inline-block;width:10px;height:3px;background:${se.color};vertical-align:middle;margin-right:6px"></span>${se.label}: <b>${yf(v)}</b></div>`;
      });
      tip.innerHTML = `<div style="color:${C.dim}">${cfg.xlabel || cfg.x} ${X[best].toLocaleString('en-US')}</div>${rows.join('')}`;
      const er = el.getBoundingClientRect();
      tip.style.display = 'block';
      tip.style.left = `${(ev.clientX - er.left) / (er.width / el.offsetWidth) + 14}px`;
      tip.style.top = `${(ev.clientY - er.top) / (er.height / el.offsetHeight) - 10}px`;
    });
    hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
  });

  // legend + table view
  const legend = h('div', { style: 'display:flex; gap:1.2em; align-items:center; font-size:0.8em; color:' + C.dim },
    cfg.series.map(se => h('span', {}, h('span', { style: `display:inline-block;width:18px;height:0;border-top:2px ${se.dash ? 'dashed' : 'solid'} ${se.color};vertical-align:middle;margin-right:6px` }), se.label)));
  const table = h('div', { style: 'display:none; max-height:300px; overflow:auto; font-size:0.7em' });
  const tbtn = h('button', { onclick: () => { const on = table.style.display === 'none'; table.style.display = on ? 'block' : 'none'; svg.style.display = on ? 'none' : 'block'; tbtn.textContent = on ? 'chart' : 'table'; } }, 'table');
  table.append(h('table', {}, h('tr', {}, h('th', {}, cfg.xlabel || cfg.x), panels.flatMap(pn => cfg.series.map(se => h('th', {}, `${pn.title ? pn.title + ': ' : ''}${se.label}`)))),
    xs.map((x, i) => h('tr', {}, h('td', {}, x.toLocaleString('en-US')), panels.flatMap(pn => cfg.series.map(se => h('td', {}, yf(get(data, (pn.base ? pn.base + '.' : '') + se.key)[i]))))))));

  el.classList.add('widget');
  el.style.position = 'relative';
  el.append(h('div', { style: 'display:flex; justify-content:space-between; align-items:center; margin-bottom:0.2em' }, legend, h('div', { class: 'wctl interactive-only' }, tbtn)), svg, table, tip);
}

function niceTicks(a, b, n) {
  const step0 = (b - a) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
