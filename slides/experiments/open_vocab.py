"""Open-vocabulary detection with OWLv2 (Minderer et al., 2023; google/owlv2-base-patch16-ensemble, Apache-2.0)
on the lecture photos: the classes are text queries, including ones no COCO detector knows ("a pear", "a crane",
"a man in a floral shirt"). On the pears, the boxes are then given to SAM as box prompts: text → boxes → masks.
For the optional open-vocabulary slide.

Needs: pip install transformers segment-anything; SAM checkpoint as in sam_points.py; OWLv2 weights download on first run.
Output: assets/lecture07/data/open_vocab.json and ov_pears.png (RGBA mask overlay)
Run:    .venv/bin/python experiments/open_vocab.py
"""
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from sam_modes import overlay   # noqa: E402  (also writes nothing on import: guarded below)

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
M = 'google/owlv2-base-patch16-ensemble'
proc = Owlv2Processor.from_pretrained(M)
owl = Owlv2ForObjectDetection.from_pretrained(M).eval()

# per photo: (query, colour, score threshold, at most k boxes)
PANELS = {
    'pears': [('a pear', '#5fd38d', 0.3, 5), ('an apple', '#ff6b6b', 0.3, 5)],
    'crowd': [('an umbrella', '#5ab0ff', 0.4, 1), ('a man in a floral shirt', '#f2b134', 0.3, 1), ('a hat', '#b28dff', 0.29, 3)],
    'street': [('a crane', '#f2b134', 0.4, 1), ('a hotel sign', '#5fd38d', 0.3, 1)],
}


def nms(boxes, thr=0.5):
    keep = []
    for b in boxes:
        x0, y0, x1, y1 = b['b']
        ok = True
        for k in keep:
            a0, b0, a1, b1 = k['b']
            iw, ih = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
            inter = iw * ih
            if inter / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter) > thr:
                ok = False
        if ok:
            keep.append(b)
    return keep


panels, times = {}, []
for name, qs in PANELS.items():
    img = Image.open(ROOT / f'assets/lecture07/{name}.jpg').convert('RGB')
    W, H = img.size
    t = time.perf_counter()
    inp = proc(text=[[q for q, *_ in qs]], images=img, return_tensors='pt')
    with torch.no_grad():
        out = owl(**inp)
    times.append(time.perf_counter() - t)
    S = max(W, H)   # OWLv2 pads the image to a square (bottom / right), so boxes are in padded coordinates
    res = proc.image_processor.post_process_object_detection(out, threshold=0.1, target_sizes=torch.tensor([[S, S]]))[0]
    boxes = []
    for qi, (q, col, thr, k) in enumerate(qs):
        cand = sorted(({'b': [round(max(0, float(v))) for v in b], 'score': round(float(s), 2), 'label': q, 'color': col}
                       for s, l, b in zip(res['scores'], res['labels'], res['boxes']) if int(l) == qi and float(s) >= thr),
                      key=lambda c: -c['score'])
        for c in cand:
            c['b'] = [min(c['b'][0], W), min(c['b'][1], H), min(c['b'][2], W), min(c['b'][3], H)]
        boxes += nms(cand)[:k]
    panels[name] = {'src': f'{name}.jpg', 'size': [W, H], 'prompts': [{'text': q, 'color': c} for q, c, *_ in qs], 'boxes': boxes}

# text → boxes → masks: the OWLv2 boxes on the pears as SAM box prompts
sam = sam_model_registry['vit_b'](checkpoint=str(ROOT / 'experiments/data/sam/sam_vit_b_01ec64.pth')).eval()
pred = SamPredictor(sam)
pred.set_image(np.array(Image.open(ROOT / 'assets/lecture07/pears.jpg').convert('RGB')))
masks = [pred.predict(box=np.array(b['b']), multimask_output=False)[0][0] for b in panels['pears']['boxes']]
panels['pears']['overlay'] = overlay(masks, [b['color'] for b in panels['pears']['boxes']], 'ov_pears.png')

json.dump({'model': 'OWLv2 base/16 ensemble + SAM ViT-B', 'owl_s': round(float(np.median(times)), 1), 'panels': panels},
          open(OUT / 'open_vocab.json', 'w'), indent=1)
for n, p in panels.items():
    print(n, [(b['label'], b['score'], b['b']) for b in p['boxes']])
print('OWLv2 per image', round(float(np.median(times)), 1), 's')
