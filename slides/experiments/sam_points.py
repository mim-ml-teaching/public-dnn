"""Segment Anything (SAM, ViT-B) on the pears photo: the image encoder runs once, then the mask decoder for a
grid of single-click prompts. For every click SAM returns 3 masks (whole / part / sub-part) with predicted IoU
scores; the slide widget shows them for the click nearest to where the user clicks. Also one mask per fruit
(a click at its centre) for the "four tasks" slide, and the encoder / decoder times on CPU.

Checkpoint: experiments/data/sam/sam_vit_b_01ec64.pth (https://dl.fbaipublicfiles.com/segment_anything/), pip install segment-anything
Output: assets/lecture07/data/pears_sam.json
Run:    .venv/bin/python experiments/sam_points.py
"""
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from lecture07_common import mask_json

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
STEP = 24          # click grid spacing, pixels of the 960-px photo
DS = 2             # masks stored at half resolution

sam = sam_model_registry['vit_b'](checkpoint=str(ROOT / 'experiments/data/sam/sam_vit_b_01ec64.pth')).eval()
pred = SamPredictor(sam)
img = np.array(Image.open(ROOT / 'assets/lecture07/pears.jpg').convert('RGB'))
H, W = img.shape[:2]
t = time.perf_counter(); pred.set_image(img); t_enc = time.perf_counter() - t
emb = pred.features.shape

uniq = []          # unique masks (half resolution, bool)
areas = []


def index_of(m):
    a = m.sum()
    for i, u in enumerate(uniq):
        if abs(int(areas[i]) - int(a)) > 0.05 * max(a, areas[i]):
            continue
        inter = np.logical_and(u, m).sum()
        if inter / (a + areas[i] - inter + 1e-9) > 0.95:
            return i
    uniq.append(m); areas.append(a)
    return len(uniq) - 1


points, t_dec = [], []
for y in range(STEP // 2, H, STEP):
    for x in range(STEP // 2, W, STEP):
        t = time.perf_counter()
        masks, scores, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
        t_dec.append(time.perf_counter() - t)
        small = masks[:, DS // 2::DS, DS // 2::DS]
        points.append({'x': x, 'y': y, 'm': [index_of(m) for m in small], 's': [round(float(s), 3) for s in scores]})

FRUIT = {'pear 1': (210, 320), 'pear 2': (395, 320), 'pear 3': (577, 320), 'apple': (752, 330)}
objects = []
for name, (x, y) in FRUIT.items():
    masks, scores, _ = pred.predict(point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True)
    # the best-scoring of the three masks that covers one fruit (bounding box narrower than 30% of the photo);
    # the others are the whole row of fruit
    def width(k):
        xs = np.nonzero(masks[k].any(0))[0]
        return xs.max() - xs.min()
    i = max((k for k in range(3) if width(k) < 0.3 * W), key=lambda k: scores[k])
    ys, xs = np.nonzero(masks[i])
    objects.append({'name': name, 'click': [x, y], 'score': round(float(scores[i]), 3),
                    'b': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], 'mask': mask_json(masks[i])})

res = {'size': [W, H], 'step': STEP, 'ds': DS, 'model': 'SAM ViT-B (sam_vit_b_01ec64)',
       'embedding': list(emb), 'encoder_s': round(t_enc, 2), 'decoder_ms': round(1000 * float(np.median(t_dec)), 1),
       'points': points, 'masks': [mask_json(m) for m in uniq], 'objects': objects}
json.dump(res, open(OUT / 'pears_sam.json', 'w'))
print(f'encoder {t_enc:.2f} s, embedding {tuple(emb)}, decoder {res["decoder_ms"]} ms per click, '
      f'{len(points)} clicks, {len(uniq)} unique masks, {(OUT / "pears_sam.json").stat().st_size / 1e6:.1f} MB')
print([(o['name'], o['score'], o['b']) for o in objects])
