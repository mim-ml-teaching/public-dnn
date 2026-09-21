// Text-to-image generation as a short animation: a prompt is typed in, then the picture is "generated" from noise:
// pure noise first, then the coarse layout, then the details. The picture is an image file; the animation is a cartoon
// of sampling (blur + noise fading out), not a model. Plays in a loop by itself (cfg.loop: false → on demand).
// config: { image: 'path relative to the page', prompt: '…', width: 480, height: 270, steps: 50, t: 0, print_t: 1 }
import { C, h, isPrint, loopPlayer } from './util.js';

const T_TYPE = 0.3, T_NOISE = 0.38, T_DONE = 0.92;   // phases of t ∈ [0, 1]: typing | pure noise | denoising | final image

export async function mount(el, cfg) {
  const W = cfg.width || 480, H = cfg.height || 270, STEPS = cfg.steps || 50, PROMPT = cfg.prompt || '';
  const st = { t: cfg.t ?? 0 };
  const canvas = h('canvas', { width: W, height: H, style: `width:${W}px; height:${H}px` }), g = canvas.getContext('2d');
  const off = document.createElement('canvas'); off.width = W; off.height = H; const go = off.getContext('2d');
  // three frames of grey noise, cycled while generating so that the noise flickers
  const noise = [0, 1, 2].map(() => {
    const c = document.createElement('canvas'); c.width = W; c.height = H; const gc = c.getContext('2d'), im = gc.createImageData(W, H), d = im.data;
    for (let i = 0; i < d.length; i += 4) { d[i] = d[i + 1] = d[i + 2] = 40 + Math.random() * 175; d[i + 3] = 255; }
    gc.putImageData(im, 0, 0); return c;
  });
  const img = new Image(); img.src = cfg.image || '';
  const ok = await new Promise(res => { img.onload = () => res(true); img.onerror = () => res(false); });
  // the prompt box (fixed height: two lines, so nothing moves while the text is typed), the image, a status line
  const typed = h('span', {}), cursor = h('span', { style: `color:${C.accent}` }, '▌');
  // the box is sized by an invisible copy of the full prompt, the typed text is laid over it: nothing moves while typing
  const PAD = 'padding:0.35em 0.6em';
  const promptBox = h('div', { style: `position:relative; font-family:var(--mono); font-size:0.85em; line-height:1.35; background:var(--bg-3); border:1px solid var(--line); border-radius:8px; box-sizing:border-box; width:${W}px; margin-bottom:0.4em; white-space:pre-wrap` },
    h('div', { style: `${PAD}; visibility:hidden` }, 'prompt: ', PROMPT, '▌'),
    h('div', { style: `position:absolute; inset:0; ${PAD}` }, h('span', { style: 'color:var(--fg-dim)' }, 'prompt: '), typed, cursor));
  const sLeft = h('span', {}, ''), sRight = h('span', {}, '');
  const status = h('div', { class: 'dim', style: `display:flex; justify-content:space-between; width:${W}px; min-height:1.4em; margin-top:0.3em; font-variant-numeric:tabular-nums` }, sLeft, sRight);

  function draw() {
    const t = st.t, n = PROMPT.length, shown = t < T_TYPE ? Math.floor(n * t / T_TYPE) : n;
    typed.textContent = PROMPT.slice(0, shown); cursor.style.visibility = t < T_TYPE ? 'visible' : 'hidden';
    g.fillStyle = C.bg2; g.fillRect(0, 0, W, H);
    if (t < T_TYPE) { sLeft.textContent = ''; sRight.textContent = ''; return; }
    const frame = noise[Math.floor(t * 60) % 3];
    if (t < T_NOISE) { g.drawImage(frame, 0, 0); sLeft.textContent = 'generating…'; sRight.textContent = `step 0 / ${STEPS}`; return; }
    const p = Math.min(1, (t - T_NOISE) / (T_DONE - T_NOISE)), k = Math.round(p * STEPS);
    if (ok) {   // the picture at a resolution growing with p, scaled up smoothly: coarse layout first, details last
      const f = 0.015 + 0.985 * p ** 3, w = Math.max(3, Math.round(W * f)), hh = Math.max(2, Math.round(H * f));
      go.clearRect(0, 0, W, H); go.imageSmoothingEnabled = true; go.drawImage(img, 0, 0, w, hh);
      g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, w, hh, 0, 0, W, H);
    } else { g.fillStyle = C.dim; g.font = '14px Inter, Arial'; g.textAlign = 'center'; g.fillText(`image not found: ${cfg.image}`, W / 2, H / 2); }
    const a = (1 - p) ** 1.5 * 0.92;
    if (a > 0.01) { g.globalAlpha = a; g.drawImage(frame, 0, 0); g.globalAlpha = 1; }
    sLeft.textContent = p < 1 ? 'generating…' : 'generated image'; sRight.textContent = `step ${k} / ${STEPS}`;
  }
  // plays in a loop by itself: 7 s from the empty prompt to the picture, a pause on the picture, again
  const loop = loopPlayer({ dur: 7000, hold: 3500, hold0: 600, alive: () => canvas.isConnected, step: t => { st.t = t; slider.value = t; draw(); } });
  const setBtn = () => { playBtn.textContent = loop.playing ? '⏸' : '▶'; };
  const play = () => { if (loop.playing) loop.stop(); else loop.start(st.t); setBtn(); };
  const reset = (autoplay = cfg.loop !== false) => { loop.stop(); st.t = cfg.t ?? 0; slider.value = st.t; if (autoplay && !isPrint()) loop.start(st.t); setBtn(); };
  const playBtn = h('button', { onclick: play, style: 'min-width:2.4em' }, '▶');
  const slider = h('input', { type: 'range', min: 0, max: 1, step: 0.005, value: st.t, style: 'width:10em', oninput: e => { loop.stop(); setBtn(); st.t = +e.target.value; draw(); } });
  el.classList.add('widget');
  el.append(promptBox, canvas, status, h('div', { class: 'wctl interactive-only', style: 'gap:0.5em; margin-top:0.3em' }, slider, playBtn));
  el.closest('.reveal')?.addEventListener('slidechanged', ev => { if (!canvas.isConnected) return; if (ev.currentSlide?.contains(el)) { reset(); draw(); } else if (loop.playing) { loop.stop(); setBtn(); } });
  if (isPrint()) { st.t = cfg.print_t ?? 1; slider.value = st.t; }
  draw();
  if (!isPrint() && el.closest('section')?.classList.contains('present')) reset();   // mounted on the current slide: start now
}
