// A YouTube video that starts when the slide's fragment containing this widget is shown (arrow key), with sound, at a
// given second. The iframe is created only then (so nothing loads or plays before the click) and removed when the
// fragment is hidden or the slide is left. Before that, and as a fallback, a dark box with a play button starts it on click.
// config: { id: 'iv-5mZ_9CPY', start: 1696 (seconds), width: 880, height: 495, label: 'from 28:16' }
import { C, h, isPrint } from './util.js';

export function mount(el, cfg) {
  const W = cfg.width || 880, H = cfg.height || 495;
  const url = `https://www.youtube.com/watch?v=${cfg.id}&t=${cfg.start || 0}s`;
  const box = h('div', { style: `position:relative; width:${W}px; height:${H}px; background:#0b0c0f; border:1px solid ${C.line}; border-radius:10px; overflow:hidden; display:flex; align-items:center; justify-content:center; text-align:center` });
  const play = h('button', { style: 'font-size:1.1em; padding:0.5em 1.1em', onclick: () => start() }, `▶ play ${cfg.label || ''}`);
  const placeholder = () => { box.replaceChildren(isPrint() ? h('div', { style: `color:${C.dim}; font-size:1.1em` }, `video: ${url}`) : play); };
  const start = () => {
    if (isPrint() || box.querySelector('iframe')) return;
    const src = `https://www.youtube.com/embed/${cfg.id}?start=${cfg.start || 0}&autoplay=1&rel=0&playsinline=1`;
    box.replaceChildren(h('iframe', { src, title: cfg.label || 'video', style: 'width:100%; height:100%; border:0', allow: 'autoplay; encrypted-media; picture-in-picture; fullscreen', allowfullscreen: '', referrerpolicy: 'strict-origin-when-cross-origin' }));
  };
  const stop = () => { if (box.querySelector('iframe')) placeholder(); };
  el.classList.add('widget');
  el.append(box);
  placeholder();
  if (isPrint()) return;
  const deck = el.closest('.reveal'), frag = el.closest('.fragment'), sec = el.closest('section');
  deck?.addEventListener('fragmentshown', ev => { if (box.isConnected && frag && ev.fragment === frag) start(); });
  deck?.addEventListener('fragmenthidden', ev => { if (box.isConnected && frag && ev.fragment === frag) stop(); });
  deck?.addEventListener('slidechanged', ev => { if (box.isConnected && !ev.currentSlide?.contains(el)) stop(); });
}
