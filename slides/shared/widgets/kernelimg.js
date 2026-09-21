// Image ∗ kernel = output, live. The kernel is editable (numbers or fractions such as 1/9) and has presets.
// Kernels whose entries sum to 0 (edge detectors) give values around 0: they are shown as |output|, brighter =
// stronger response, as is usual for edge maps.
// config: { src: "../../assets/lecture06/chelsea.jpg", preset: "blur", width: 340, presets: ["identity", ...] }
import { C, h } from './util.js';

export const PRESETS = {
  identity: { label: 'identity', k: [['0', '0', '0'], ['0', '1', '0'], ['0', '0', '0']] },
  blur: { label: 'blur 3×3', k: [['1/9', '1/9', '1/9'], ['1/9', '1/9', '1/9'], ['1/9', '1/9', '1/9']] },
  blur5: { label: 'blur 5×5', k: Array.from({ length: 5 }, () => Array(5).fill('1/25')) },
  sharpen: { label: 'sharpen', k: [['0', '-1', '0'], ['-1', '5', '-1'], ['0', '-1', '0']] },
  vedge: { label: 'vertical edges', k: [['-1', '0', '1'], ['-2', '0', '2'], ['-1', '0', '1']] },
  hedge: { label: 'horizontal edges', k: [['-1', '-2', '-1'], ['0', '0', '0'], ['1', '2', '1']] },
  outline: { label: 'pixel − mean of neighbours', k: [['0', '-1/4', '0'], ['-1/4', '1', '-1/4'], ['0', '-1/4', '0']] },
};

const parse = s => {
  s = String(s).trim().replace('−', '-');
  if (s.includes('/')) { const [a, b] = s.split('/').map(Number); return b ? a / b : NaN; }
  return Number(s);
};

// Convolution (cross-correlation, as in deep learning) of an RGB float image with replicate padding.
export function convolveRGB(src, W, H, K) {
  const k = K.length, r = (k - 1) / 2, out = new Float32Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s0 = 0, s1 = 0, s2 = 0;
    for (let a = 0; a < k; a++) {
      const yy = Math.min(H - 1, Math.max(0, y + a - r));
      for (let b = 0; b < k; b++) {
        const w = K[a][b];
        if (!w) continue;
        const xx = Math.min(W - 1, Math.max(0, x + b - r)), i = (yy * W + xx) * 3;
        s0 += w * src[i]; s1 += w * src[i + 1]; s2 += w * src[i + 2];
      }
    }
    const o = (y * W + x) * 3;
    out[o] = s0; out[o + 1] = s1; out[o + 2] = s2;
  }
  return out;
}

export async function loadImage(src, width) {
  const img = new Image();
  img.src = src;
  await img.decode();
  const W = width, H = Math.round(img.height * width / img.width);
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.drawImage(img, 0, 0, W, H);
  const d = g.getImageData(0, 0, W, H).data, f = new Float32Array(W * H * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { f[j] = d[i] / 255; f[j + 1] = d[i + 1] / 255; f[j + 2] = d[i + 2] / 255; }
  return { W, H, f };
}

export function show(canvas, f, W, H, abs = false, gain = 1) {
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext('2d'), im = g.createImageData(W, H);
  for (let i = 0, j = 0; j < f.length; i += 4, j += 3) {
    for (let c = 0; c < 3; c++) {
      let v = f[j + c];
      if (abs) v = Math.abs(v) * gain;
      im.data[i + c] = Math.max(0, Math.min(255, v * 255));
    }
    im.data[i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
}

export async function mount(el, cfg) {
  el.classList.add('widget');
  const width = cfg.width || 340;
  const { W, H, f } = await loadImage(cfg.src, width);
  const cin = h('canvas'), cout = h('canvas');
  [cin, cout].forEach(c => { c.style.width = W + 'px'; c.style.height = H + 'px'; c.style.imageRendering = 'auto'; });
  show(cin, f, W, H);
  const mat = h('div', { class: 'matrix' });
  const sumEl = h('div', { class: 'readout', style: 'margin-top:0.3em; text-align:center' });
  const note = h('div', { class: 'small', style: 'text-align:center; min-height:1.3em; width:' + W + 'px' });
  let inputs = [];

  function render() {
    const K = inputs.map(row => row.map(i => parse(i.value)));
    if (K.flat().some(v => !Number.isFinite(v))) { sumEl.textContent = 'invalid entry'; return; }
    const sum = K.flat().reduce((a, b) => a + b, 0);
    const zero = Math.abs(sum) < 1e-9;
    sumEl.textContent = `sum of entries = ${+sum.toFixed(3)}`;
    note.textContent = zero ? 'entries sum to 0: shown as |output|' : '';
    const pos = K.flat().reduce((a, b) => a + Math.max(0, b), 0);
    show(cout, convolveRGB(f, W, H, K), W, H, zero, zero ? 4 / Math.max(pos, 1e-9) : 1);
  }
  function setKernel(rows) {
    mat.replaceChildren();
    mat.style.gridTemplateColumns = `repeat(${rows.length}, auto)`;
    inputs = rows.map(row => row.map(v => {
      const i = h('input', { type: 'text', value: v, style: `font: inherit; font-size:1em; width:${rows.length > 3 ? 2.9 : 3.2}em; font-family:var(--mono); padding:0.1em 0; text-align:center; color:${C.fg}; background:${C.bg}; border:1px solid ${C.line}; border-radius:4px` });
      i.addEventListener('input', render);
      mat.append(i);
      return i;
    }));
    render();
  }
  const names = cfg.presets || Object.keys(PRESETS);
  const btns = names.map(n => h('button', { onclick: () => { btns.forEach(b => b.classList.remove('active')); btns[names.indexOf(n)].classList.add('active'); setKernel(PRESETS[n].k); } }, PRESETS[n].label));
  const start = cfg.preset || names[0];
  btns[names.indexOf(start)]?.classList.add('active');

  const op = t => h('div', { style: 'font-size:2.2em; color:' + C.dim + '; padding:0 0.25em' }, t);
  el.append(
    h('div', { style: 'display:flex; align-items:center; gap:0.3em' },
      h('div', {}, h('div', { class: 'small', style: 'text-align:center' }, 'input image'), cin),
      op('∗'),
      h('div', { style: 'display:flex; flex-direction:column; align-items:center' }, h('div', { class: 'small', style: 'color:' + C.accent }, 'kernel'), mat, sumEl),
      op('='),
      h('div', {}, h('div', { class: 'small', style: 'text-align:center' }, 'output'), cout, note)),
    h('div', { class: 'wctl interactive-only' }, h('label', {}, 'presets:'), ...btns),
  );
  setKernel(PRESETS[start].k);
}
