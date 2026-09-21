// The parameter cost of a mixture of Gaussians as one big formula: (parameter budget) / (numbers per Gaussian) =
// (number of Gaussians). A Gaussian in d dimensions needs d numbers for its mean alone (another d for a diagonal
// covariance, d(d+1)/2 for a full one). The widget loops through data sets of growing dimension and the numbers roll:
// the denominator grows, the result collapses. The slider and the covariance buttons take over, "loop" hands back.
// config: { budget: 7e9, images: 1e9, preset: 0 (the data set shown first), loop: true }
import { C, h, isPrint, loopPlayer } from './util.js';

const PRESETS = [
  { d: 2, label: '2-D toy data', ch: 1 },
  { d: 28 * 28, label: 'MNIST digit, 28 × 28', ch: 1 },
  { d: 32 * 32 * 3, label: 'CIFAR image, 32 × 32 × 3', ch: 3 },
  { d: 256 * 256 * 3, label: 'photo, 256 × 256 × 3', ch: 3 },
  { d: 1024 * 1024 * 3, label: 'photo, 1024 × 1024 × 3', ch: 3 },
];
const COV = [{ name: 'mean only', cost: d => d, text: 'd' }, { name: '+ diagonal Σ', cost: d => 2 * d, text: '2d' }, { name: '+ full Σ', cost: d => d + d * (d + 1) / 2, text: 'd + d(d+1)/2' }];
const grp = n => Math.round(n).toLocaleString('en-US').replace(/,/g, ' ');   // 3 145 728
const big = n => n >= 1e9 ? `${+(n / 1e9).toPrecision(2)} billion` : n >= 1e6 ? `${+(n / 1e6).toPrecision(2)} million` : grp(n);

export function mount(el, cfg) {
  const BUDGET = cfg.budget || 7e9, IMAGES = cfg.images || 1e9;
  const st = { t: 0, preset: cfg.preset || 0, cov: 0, d: PRESETS[cfg.preset || 0].d };
  let from = st.d, to = st.d, t0 = 0;   // the displayed d glides from one data set to the next
  const span = (txt, style = '') => h('span', { style }, txt);
  const numEl = span(grp(BUDGET), 'font-size:1.7em; font-weight:700'), denEl = span('', 'font-size:1.7em; font-weight:700; color:var(--blue)'), resEl = span('', 'font-size:2em; font-weight:700; color:var(--accent)');
  const costNote = span('', 'color:var(--blue); opacity:0.85'), dataEl = span('', 'font-weight:600; color:var(--blue)'), noteEl = span('', 'color:var(--fg-dim)');
  // the data: a picture of the resolution (blue, like the denominator), its grid refining as d grows
  const IS = 112, img = h('canvas', { width: IS, height: IS, style: `width:${IS}px; height:${IS}px; background:transparent; border:none` }), gi = img.getContext('2d');
  const imgBox = h('div', { style: 'text-align:center; flex:none' }, img, h('div', { style: 'margin-top:0.2em' }, dataEl));
  function drawImg() {
    const p = PRESETS[st.preset], ch = p.ch, blue = C.blue;
    gi.clearRect(0, 0, IS, IS);
    if (st.preset === 0) {   // a point in the plane: two coordinates
      gi.strokeStyle = blue; gi.lineWidth = 2; gi.fillStyle = 'rgba(90,176,255,0.2)';
      const cell = Math.round(IS * 0.34), gap = Math.round(IS * 0.08), x0 = Math.round((IS - 2 * cell - gap) / 2), y = Math.round((IS - cell) / 2);   // two cells, centred, sized by the canvas
      [x0, x0 + cell + gap].forEach((x, i) => { gi.fillRect(x, y, cell, cell); gi.strokeRect(x, y, cell, cell); gi.fillStyle = blue; gi.font = `bold ${Math.round(IS * 0.14)}px Inter, Arial`; gi.textAlign = 'center'; gi.fillText(i ? 'x₂' : 'x₁', x + cell / 2, y + cell * 0.66); gi.fillStyle = 'rgba(90,176,255,0.2)'; });
      return;
    }
    const n = Math.max(1, Math.round(Math.sqrt(st.d / ch))), off = ch === 3 ? 9 : 0, side = IS - 6 - 2 * off, x0 = 3, y0 = 3 + 2 * off;
    for (let c = ch - 1; c >= 0; c--) {   // colour images: three stacked channels
      const x = x0 + c * off, y = y0 - c * off, sp = side / n;
      gi.fillStyle = c === 0 ? 'rgba(90,176,255,0.28)' : 'rgba(90,176,255,0.14)'; gi.fillRect(x, y, side, side);
      gi.strokeStyle = c === 0 ? blue : 'rgba(90,176,255,0.6)'; gi.lineWidth = 1.5; gi.strokeRect(x + 0.5, y + 0.5, side - 1, side - 1);
      if (c === 0) {
        if (sp >= 2.5) { gi.strokeStyle = 'rgba(90,176,255,0.8)'; gi.lineWidth = 1; gi.beginPath(); for (let i = 1; i < n; i++) { gi.moveTo(x + i * sp, y); gi.lineTo(x + i * sp, y + side); gi.moveTo(x, y + i * sp); gi.lineTo(x + side, y + i * sp); } gi.stroke(); }
        else { gi.fillStyle = 'rgba(90,176,255,0.55)'; gi.fillRect(x, y, side, side); }
        // the resolution, on one or two lines, at a size that fits the square
        const l1 = `${n} × ${n}`, l2 = ch === 3 ? '× 3' : ''; let fs = 15; gi.textAlign = 'center';
        for (; fs > 9; fs--) { gi.font = `bold ${fs}px Inter, Arial`; if (gi.measureText(l1).width <= side - 10) break; }
        gi.fillStyle = C.fg; gi.fillText(l1, x + side / 2, y + side / 2 + (l2 ? -2 : 5)); if (l2) gi.fillText(l2, x + side / 2, y + side / 2 + fs + 2);
      }
    }
  }
  const formula = h('div', { style: 'display:flex; align-items:center; justify-content:center; gap:0.9em; font-variant-numeric:tabular-nums; line-height:1.15' },
    h('div', { style: 'text-align:center' },
      h('div', { style: 'padding:0 0.5em 0.15em' }, numEl, span(' parameters', 'color:var(--fg-dim)')),
      h('div', { style: 'border-top:3px solid var(--fg); padding:0.2em 0.5em 0' }, denEl, span(' numbers per Gaussian', 'color:var(--fg-dim)'), h('br'), costNote)),
    span('=', 'font-size:2.2em; color:var(--fg-dim)'),
    h('div', {}, resEl, span(' Gaussians', 'color:var(--fg-dim)')));
  const row = h('div', { style: 'display:flex; align-items:center; justify-content:center; gap:1.6em' }, imgBox, formula);
  const sub = h('div', { style: 'text-align:center; margin-top:0.3em' }, noteEl);
  function draw() {
    const cost = COV[st.cov].cost(st.d), n = BUDGET / cost, p = PRESETS[st.preset];
    denEl.textContent = grp(cost); costNote.textContent = `(${COV[st.cov].text} with d = ${grp(st.d)})`;
    resEl.textContent = n >= 1 ? grp(n) : '0';
    dataEl.textContent = p.label; drawImg();
    noteEl.textContent = n >= 1 ? `training set of ${big(IMAGES)} images → ${IMAGES / n >= 2 ? `one Gaussian per ${big(IMAGES / n)} images` : `${big(n / IMAGES)} Gaussians per image`}` : `not even one: a single full covariance needs ${big(cost)} numbers`;
  }
  // glide to a data set: d moves on a log scale over 700 ms
  const glide = now => { const u = Math.min(1, (now - t0) / 700), e = u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u); st.d = Math.round(Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * e)); return u >= 1; };
  const goTo = (p, now = performance.now()) => { st.preset = p; slider.value = p; from = st.d; to = PRESETS[p].d; t0 = now; };
  let raf = 0;
  const animate = () => { cancelAnimationFrame(raf); const tick = now => { const done = glide(now); draw(); if (!done && el.isConnected) raf = requestAnimationFrame(tick); }; raf = requestAnimationFrame(tick); };
  const loop = loopPlayer({ dur: 13000, hold: 1500, hold0: 0, alive: () => el.isConnected, step: t => { st.t = t; const p = Math.min(PRESETS.length - 1, Math.floor(t * PRESETS.length)); if (p !== st.preset) goTo(p); glide(performance.now()); draw(); } });
  const setBtn = () => { loopBtn.classList.toggle('active', loop.playing); };
  const manual = () => { loop.stop(); setBtn(); };
  const slider = h('input', { type: 'range', min: 0, max: PRESETS.length - 1, step: 1, value: st.preset, style: 'width:9em', oninput: e => { manual(); goTo(+e.target.value); animate(); } });
  const covBtns = COV.map((c, i) => h('button', { onclick: () => { manual(); st.cov = i; covBtns.forEach((b, j) => b.classList.toggle('active', j === i)); draw(); } }, c.name));
  covBtns[0].classList.add('active');
  const loopBtn = h('button', { onclick: () => { if (!loop.playing) { loop.start(st.t >= 1 ? 0 : st.t); setBtn(); } } }, '⟳ loop');
  el.classList.add('widget');
  el.append(row, sub, h('div', { class: 'wctl interactive-only', style: 'gap:0.5em; margin-top:0.5em; justify-content:center' }, h('label', {}, 'data'), slider, h('span', { class: 'dim', style: 'margin-left:0.6em' }, 'covariance:'), ...covBtns, loopBtn));
  const reset = () => { loop.stop(); st.t = 0; st.preset = cfg.preset || 0; st.d = from = to = PRESETS[st.preset].d; slider.value = st.preset; if (cfg.loop !== false && !isPrint()) loop.start(0); setBtn(); draw(); };
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!el.isConnected) return; if (ev.currentSlide?.contains(el)) reset(); else manual(); });
  if (isPrint()) { st.preset = PRESETS.length - 1; st.d = PRESETS[st.preset].d; draw(); return; }
  draw();
  if (el.closest('section')?.classList.contains('present')) reset();
}
