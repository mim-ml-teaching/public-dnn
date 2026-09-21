"""Boxes on the street photo for the grid / anchor slides: torchvision Faster R-CNN (ResNet-50-FPN v2, COCO),
its usual NMS, score ≥ 0.6. These boxes stand in for ground-truth annotations on the slides.

Output: assets/lecture07/data/street_boxes.json
Run:    .venv/bin/python experiments/street_boxes.py
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
with torch.no_grad():
    out = model([F.to_tensor(img)])[0]
boxes = [{'b': [round(float(v), 1) for v in b], 's': round(float(s), 3), 'c': cats[int(l)]}
         for b, s, l in zip(out['boxes'], out['scores'], out['labels']) if float(s) >= 0.6]
json.dump({'size': list(img.size), 'model': 'torchvision fasterrcnn_resnet50_fpn_v2 (COCO), score ≥ 0.6', 'boxes': boxes},
          open(ROOT / 'assets/lecture07/data/street_boxes.json', 'w'), indent=1)
for b in boxes:
    print(b['c'], b['s'], b['b'])
