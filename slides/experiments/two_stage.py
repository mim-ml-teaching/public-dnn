"""The two stages of Faster R-CNN (torchvision, ResNet-50-FPN v2, COCO) on the street photo, for the two-stage slide:
stage 1 (region proposal network): class-agnostic proposals; stage 2: for each proposal, class probabilities (softmax
over background + COCO classes) and a refined box; final detections after NMS.

Output: assets/lecture07/data/street_two_stage.json
Run:    .venv/bin/python experiments/two_stage.py
"""
import json
import pathlib

import torch
from PIL import Image
from torchvision.models import detection as D
from torchvision.transforms import functional as F

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
W8 = D.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = D.fasterrcnn_resnet50_fpn_v2(weights=W8).eval()
cats = W8.meta['categories']
img = Image.open(ROOT / 'assets/lecture07/street.jpg').convert('RGB')
W, H = img.size
with torch.no_grad():
    images, _ = model.transform([F.to_tensor(img)])
    feats = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, feats)                       # stage 1: boxes in the resized image, sorted by objectness
    rh = model.roi_heads
    bf = rh.box_head(rh.box_roi_pool(feats, proposals, images.image_sizes))
    logits, _ = rh.box_predictor(bf)                              # stage 2: per proposal
    probs = logits.softmax(-1)
    dets, _ = rh(feats, proposals, images.image_sizes)
sy, sx = H / images.image_sizes[0][0], W / images.image_sizes[0][1]
scale = lambda b: [round(float(b[0]) * sx, 1), round(float(b[1]) * sy, 1), round(float(b[2]) * sx, 1), round(float(b[3]) * sy, 1)]
P = proposals[0]
props = []
for b, p in zip(P, probs):
    k = int(p.argmax())
    props.append({'b': scale(b), 'c': 'background' if k == 0 else cats[k], 'p': round(float(p[k]), 3), 'bg': round(float(p[0]), 3)})
final = [{'b': scale(b), 'c': cats[int(l)], 's': round(float(s), 3)} for b, l, s in zip(dets[0]['boxes'], dets[0]['labels'], dets[0]['scores']) if float(s) >= 0.7]
json.dump({'size': [W, H], 'model': 'torchvision fasterrcnn_resnet50_fpn_v2 (COCO)', 'n_proposals': len(props),
           'proposals': props, 'final': final}, open(ROOT / 'assets/lecture07/data/street_two_stage.json', 'w'))
print('proposals', len(props), 'final (score ≥ 0.7)', len(final))
from collections import Counter
print(Counter(p['c'] for p in props).most_common(8))
