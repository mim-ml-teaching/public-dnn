// Data augmentation gallery: the original photo and random label-preserving transformations of it.
// mode "geometric": flip, random crop (rescaled), rotation; "photometric": brightness, contrast, hue, grayscale,
// blur; "both"; "split": first row geometric, second row photometric; "row": n tiles without the original, the
// first half geometric, the rest photometric. "new samples" draws new random transformations.
// pipeline: true adds a row above the gallery: original + label → random transform (preprocessing) → a new
// version every epoch → network → loss with the same label.
// config: { src: "../../assets/lecture06/bison.jpg", mode: "both", n: 7, tile: 210, seed: 1, label: "bison",
//           pipeline: false, pipeTile: 170 }
import { C, h } from './util.js';

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

export async function mount(el, cfg) {
  el.classList.add('widget');
  const img = new Image();
  img.src = cfg.src;
  await img.decode();
  const n = cfg.n || 7, T0 = cfg.tile || 210, TH0 = Math.round(T0 * img.height / img.width), mode = cfg.mode || 'both';
  let seed = cfg.seed || 1;
  const grid = h('div', { style: `display:grid; grid-template-columns:repeat(${cfg.cols || 4}, ${T0}px); gap:10px 14px` });

  // size given: returns { c, ops } (canvas of that width, no caption); otherwise a captioned gallery tile
  function tile(rand, original, tmode = mode, size = 0) {
    const [T, TH] = size ? [size, Math.round(size * img.height / img.width)] : [T0, TH0];
    const c = h('canvas'); c.width = T; c.height = TH; c.style.width = T + 'px'; c.style.height = TH + 'px';
    const g = c.getContext('2d'), ops = [];
    g.fillStyle = '#000'; g.fillRect(0, 0, T, TH);
    let sx = 0, sy = 0, sw = img.width, sh = img.height, flip = false, rot = 0;
    const filters = [];
    if (!original) {
      const geo = tmode !== 'photometric', pho = tmode !== 'geometric';
      if (geo) {
        if (rand() < 0.5) { flip = true; ops.push('flip'); }
        if (rand() < 0.7) {
          const s = 0.45 + 0.5 * rand(), ar = Math.exp((rand() - 0.5) * 0.5);
          sw = img.width * Math.sqrt(s * ar); sh = img.height * Math.sqrt(s / ar);
          sw = Math.min(sw, img.width); sh = Math.min(sh, img.height);
          sx = rand() * (img.width - sw); sy = rand() * (img.height - sh);
          ops.push(`crop ${Math.round(100 * sw * sh / img.width / img.height)}%`);
        }
        if (rand() < 0.5) { rot = Math.round((rand() - 0.5) * 50); ops.push(`rotate ${rot}°`); }
      }
      if (pho) {
        const which = Math.floor(rand() * 5);            // one photometric change per sample, sometimes two
        const pick = [which, rand() < 0.3 ? (which + 1 + Math.floor(rand() * 4)) % 5 : -1];
        for (const q of pick) {
          if (q === 0) { const b = rand() < 0.5 ? 0.5 + 0.3 * rand() : 1.25 + 0.3 * rand(); filters.push(`brightness(${b.toFixed(2)})`); ops.push(`brightness ×${b.toFixed(1)}`); }
          if (q === 1) { const v = rand() < 0.5 ? 0.5 : 1.6; filters.push(`contrast(${v})`); ops.push(`contrast ×${v}`); }
          if (q === 2) { filters.push('grayscale(1)'); ops.push('grayscale'); }
          if (q === 3) { const v = Math.round((rand() < 0.5 ? -1 : 1) * (30 + 60 * rand())); filters.push(`hue-rotate(${v}deg)`); ops.push(`hue ${v > 0 ? '+' : ''}${v}°`); }
          if (q === 4) { const v = (1.5 + 1.5 * rand()).toFixed(1); filters.push(`blur(${v}px)`); ops.push('blur'); }
        }
      }
      if (!ops.length) { flip = true; ops.push('flip'); }
    }
    g.save();
    g.filter = filters.join(' ') || 'none';
    g.translate(T / 2, TH / 2); g.rotate(rot * Math.PI / 180); if (flip) g.scale(-1, 1);
    g.drawImage(img, sx, sy, sw, sh, -T / 2, -TH / 2, T, TH);
    g.restore();
    if (size) return { c, ops };
    return h('div', {}, c, h('div', { class: 'small', style: `color:${original ? C.accent : C.dim}; line-height:1.25; min-height:2.5em` },
      original ? `original · label: ${cfg.label || 'bison'}` : ops.join(' · ') + (cfg.pipeline ? '' : `  →  ${cfg.label || 'bison'}`)));
  }
  const pipe = h('div', { style: 'display:flex; align-items:center; gap:0.5em; margin-bottom:0.5em' });
  const lbl = cfg.label || 'bison';
  const arrow = (top, bottom = '') => h('div', { style: `display:flex; flex-direction:column; align-items:center; min-width:${top ? 6.5 : 2.2}em; text-align:center; white-space:nowrap` },
    h('div', { class: 'small', style: `color:${C.accent}; line-height:1.15` }, top),
    h('div', { style: `font-size:1.6em; line-height:1; color:${C.dim}` }, '⟶'),
    h('div', { class: 'small', style: 'line-height:1.15' }, bottom));
  const box = (txt, color) => h('div', { style: `border:2px solid ${color}; border-radius:10px; padding:0.35em 0.6em; text-align:center; line-height:1.25` }, ...txt);
  function drawPipe() {
    const PT = cfg.pipeTile || 170, PH = Math.round(PT * img.height / img.width), off = 12, rand = rng(seed * 7 + 1);
    const orig = tile(rand, true, mode, PT).c;
    const vers = [0, 1, 2].map(() => tile(rand, false, 'both', PT));
    const stack = h('div', { style: `position:relative; width:${PT + 2 * off}px; height:${PH + 2 * off}px` },
      ...vers.map((v, i) => { v.c.style.cssText += `;position:absolute; left:${(2 - i) * off}px; top:${i * off}px; border-radius:4px; box-shadow:0 0 0 2px ${C.bg}; opacity:${i === 2 ? 1 : 0.75}`; return v.c; }));
    const cap = (el2, t1, t2, color = C.dim) => h('div', { style: 'display:flex; flex-direction:column; align-items:center; gap:0.2em' }, el2,
      h('div', { class: 'small', style: `color:${color}; text-align:center; line-height:1.2` }, t1, h('br'), t2));
    orig.style.borderRadius = '4px';
    pipe.replaceChildren(
      cap(orig, 'training image', `label: ${lbl}`, C.accent),
      arrow('random transform', 'in preprocessing'),
      cap(stack, 'a new version', 'every epoch'),
      arrow(''),
      box([h('span', { style: 'white-space:nowrap' }, 'network ', h('i', {}, 'f'), h('sub', {}, 'θ'))], C.blue),
      arrow(''),
      h('div', { class: 'small', style: 'text-align:center; line-height:1.2' }, 'prediction'),
      arrow(''),
      box([h('span', { style: 'white-space:nowrap' }, 'loss( · , ', h('b', { style: `color:${C.accent}` }, lbl), ')'), h('br'), h('span', { class: 'small' }, 'label unchanged')], C.accent));
  }
  function draw() {
    if (cfg.pipeline) drawPipe();
    const rand = rng(seed);
    if (mode === 'row') {     // one row: geometric, then photometric
      const g = Math.floor(n / 2);
      grid.replaceChildren(...Array.from({ length: g }, () => tile(rand, false, 'geometric')),
        ...Array.from({ length: n - g }, () => tile(rand, false, 'photometric')));
    } else if (mode === 'split') {   // row 1: original + geometric, row 2: photometric
      const per = (cfg.cols || 4);
      grid.replaceChildren(tile(rand, true), ...Array.from({ length: per - 1 }, () => tile(rand, false, 'geometric')),
        ...Array.from({ length: per }, () => tile(rand, false, 'photometric')));
    } else grid.replaceChildren(tile(rand, true), ...Array.from({ length: n }, () => tile(rand, false)));
  }
  if (cfg.pipeline) el.append(pipe, h('div', { class: 'small', style: 'margin-bottom:0.2em' }, 'more random versions of the same image:'));
  el.append(grid, h('div', { class: 'wctl interactive-only' }, h('button', { onclick: () => { seed++; draw(); } }, 'new samples')));
  draw();
}
