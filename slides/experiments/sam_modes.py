"""What SAM (ViT-B) can be asked, on the lecture photos: segment everything (a grid of clicks, automatic), a box
prompt, and several clicks with positive and negative labels, and a
click on a part (a stalk). For the optional SAM slide.

Checkpoint: experiments/data/sam/sam_vit_b_01ec64.pth (https://dl.fbaipublicfiles.com/segment_anything/), pip install segment-anything
Output: assets/lecture07/data/sam_modes.json and sam_*.png (RGBA mask overlays, same size as the photo)
Run:    .venv/bin/python experiments/sam_modes.py
"""
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
PAL = ['#ff6b6b', '#5fd38d', '#5ab0ff', '#b28dff', '#f2b134', '#4fd1c5', '#f78fb3', '#c3e88d', '#82aaff', '#ffcb6b']
rgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]

def overlay(masks, colors, path, alpha=150):
    h, w = masks[0].shape
    o = np.zeros((h, w, 4), np.uint8)
    for m, c in zip(masks, colors):
        o[m] = rgb(c) + [alpha]
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        o[edge] = [255, 255, 255, 230]
    Image.fromarray(o, 'RGBA').save(OUT / path, optimize=True)
    return path


def load(name):
    return np.array(Image.open(ROOT / f'assets/lecture07/{name}.jpg').convert('RGB'))



def main():
    sam = sam_model_registry['vit_b'](checkpoint=str(ROOT / 'experiments/data/sam/sam_vit_b_01ec64.pth')).eval()
    panels = {}
    # 1. everything: SAM's automatic mode (a 32×32 grid of clicks, keep confident and stable masks, remove duplicates)
    img = load('crowd')
    t = time.perf_counter()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92)
    auto = sorted(gen.generate(img), key=lambda a: -a['area'])
    t_auto = time.perf_counter() - t
    rng = np.random.default_rng(0)
    cols = [PAL[i % len(PAL)] for i in rng.permutation(len(auto))]
    panels['everything'] = {'src': 'crowd.jpg', 'size': [img.shape[1], img.shape[0]], 'n': len(auto), 'seconds': round(t_auto, 1),
                            'overlay': overlay([a['segmentation'] for a in auto], cols, 'sam_everything.png')}

    pred = SamPredictor(sam)
    # 2. a box around the bus
    img = load('street'); pred.set_image(img)
    box = np.array([768, 341, 960, 634])
    m, s, _ = pred.predict(box=box, multimask_output=False)
    panels['box'] = {'src': 'street.jpg', 'size': [img.shape[1], img.shape[0]], 'score': round(float(s[0]), 3),
                     'boxes': [{'b': box.tolist(), 'color': '#f2b134', 'dash': True}],
                     'overlay': overlay([m[0]], ['#5ab0ff'], 'sam_box.png')}
    # 3. clicks with labels: + on the second and third pear, − on the apple: one mask for both pears
    img = load('pears'); pred.set_image(img)
    pts = np.array([[395, 320], [577, 320], [752, 330]]); lab = np.array([1, 1, 0])
    m, s, _ = pred.predict(point_coords=pts, point_labels=lab, multimask_output=False)
    panels['clicks'] = {'src': 'pears.jpg', 'size': [img.shape[1], img.shape[0]], 'score': round(float(s[0]), 3),
                        'points': [{'x': int(x), 'y': int(y), 'pos': bool(l)} for (x, y), l in zip(pts, lab)],
                        'overlay': overlay([m[0]], ['#5fd38d'], 'sam_clicks.png')}
    # 4. a click on a stalk: of the three masks, SAM is most confident in the stalk alone (a part)
    pt = np.array([[425, 140]])
    ms, s, _ = pred.predict(point_coords=pt, point_labels=np.array([1]), multimask_output=True)
    i = int(np.argmax(s))
    panels['part'] = {'src': 'pears.jpg', 'size': [img.shape[1], img.shape[0]], 'score': round(float(s[i]), 3),
                      'points': [{'x': 425, 'y': 140, 'pos': True}], 'overlay': overlay([ms[i]], ['#b28dff'], 'sam_part.png')}
    json.dump({'model': 'SAM ViT-B (sam_vit_b_01ec64)', 'panels': panels}, open(OUT / 'sam_modes.json', 'w'), indent=1)
    print({k: {kk: v for kk, v in p.items() if kk in ('n', 'seconds', 'score')} for k, p in panels.items()})


if __name__ == '__main__':
    main()
