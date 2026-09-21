// Helpers for the photo-based widgets of lecture 7 (segmentation, detection, SAM): loading, run-length decoded
// masks, IoU and non-maximum suppression, and a canvas sized for the slide.
import { h, isPrint } from './util.js';

export const loadImage = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
export const loadJSON = src => fetch(src).then(r => r.json());

// {h, w, rle: [zeros, ones, zeros, ...]} → Uint8Array(h*w) of 0/1
export function decodeMask(m) {
  const out = new Uint8Array(m.h * m.w);
  let p = 0, v = 0;
  for (const n of m.rle) { if (v) out.fill(1, p, p + n); p += n; v ^= 1; }
  return out;
}

// boxes as [x1, y1, x2, y2]
export function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih, ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

// Greedy NMS per class: sort by score, keep the best, remove every box of the same class with IoU > thr with it.
// Returns for every box: { keep: true } or { keep: false, by: index of the box that removed it }.
export function nms(boxes, thr) {
  const order = boxes.map((_, i) => i).sort((i, j) => boxes[j].s - boxes[i].s);
  const res = boxes.map(() => null);
  for (const i of order) {
    if (res[i]) continue;
    res[i] = { keep: true };
    for (const j of order) {
      if (res[j] || boxes[j].c !== boxes[i].c) continue;
      if (iou(boxes[i].b, boxes[j].b) > thr) res[j] = { keep: false, by: i };
    }
  }
  return res;
}

// A canvas of display width `width` (CSS px) drawn at 2× for sharpness (1× when printing: large 2× canvases make
// Chrome's printToPDF hang, see lecture 6's tta widget); g is scaled so that 1 unit = 1 photo pixel.
export function photoCanvas(img, width, size) {
  const [W, H] = size || [img.width, img.height];
  const dw = width, dh = Math.round(width * H / W), dpr = isPrint() ? 1 : 2;
  const cv = h('canvas', { style: `width:${dw}px; height:${dh}px; display:block; border-radius:6px` });
  cv.width = dw * dpr; cv.height = dh * dpr;
  const g = cv.getContext('2d');
  const k = dw * dpr / W;
  return { cv, g, k, dpr, dw, dh, W, H, drawPhoto(alpha = 1) { g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.globalAlpha = alpha; g.drawImage(img, 0, 0, cv.width, cv.height); g.globalAlpha = 1; g.setTransform(k, 0, 0, k, 0, 0); } };
}

// Print view: replace a finished canvas by a JPEG image of it. Many large 2D canvases make Chrome's printToPDF
// hang (lecture 7 has a dozen photo canvases); images print fine.
export function freezeForPrint(cv) {
  if (!isPrint() || !cv.isConnected) return;
  const img = new Image();
  img.src = cv.toDataURL('image/jpeg', 0.9);
  img.style.cssText = cv.style.cssText;
  img.style.margin = '0';
  cv.replaceWith(img);
}

// Paint a decoded mask (mask resolution mw×mh, photo W×H) in colour `rgb` = [r, g, b] with opacity a.
export function maskLayer(mask, mw, mh, rgb, a = 0.5, edge = true) {
  const c = document.createElement('canvas'); c.width = mw; c.height = mh;
  const g = c.getContext('2d'), im = g.createImageData(mw, mh);
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    const i = y * mw + x;
    if (!mask[i]) continue;
    const border = edge && (x === 0 || y === 0 || x === mw - 1 || y === mh - 1 || !mask[i - 1] || !mask[i + 1] || !mask[i - mw] || !mask[i + mw]);
    im.data[4 * i] = rgb[0]; im.data[4 * i + 1] = rgb[1]; im.data[4 * i + 2] = rgb[2]; im.data[4 * i + 3] = Math.round(255 * (border ? 1 : a));
  }
  g.putImageData(im, 0, 0);
  return c;
}

export const hex2rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));

// box label: filled tag above the top-left corner
export function tag(g, x, y, text, color, k, size = 13, dpr = 2) {
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0);
  const fs = size * dpr, u = dpr / 2;
  g.font = `600 ${fs}px Inter, Arial, sans-serif`;
  const tw = g.measureText(text).width + 10 * u;
  const X = x * k, Y = Math.max(fs + 6 * u, y * k);
  g.fillStyle = color; g.fillRect(X - u, Y - fs - 6 * u, tw, fs + 6 * u);
  g.fillStyle = '#15171c'; g.fillText(text, X + 4 * u, Y - 6 * u);
  g.restore();
}
