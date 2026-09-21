"""The raw output of Mask R-CNN's mask head (torchvision maskrcnn_resnet50_fpn_v2, COCO) for one person on the crowd
photo: a 28×28 grid of probabilities (sigmoid) inside the person's box, before it is resized into the box.
For the Mask R-CNN architecture diagram.

Output: assets/lecture07/data/mask28.png (28×28, grey = probability) and mask28.json (the box, score)
Run:    .venv/bin/python experiments/mask_head.py
"""
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from torchvision.models import detection as D
from torchvision.transforms import functional as F

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
W8 = D.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = D.maskrcnn_resnet50_fpn_v2(weights=W8).eval()
img = Image.open(ROOT / 'assets/lecture07/crowd.jpg').convert('RGB')
W, H = img.size
with torch.no_grad():
    det = model([F.to_tensor(img)])[0]
    # the most confident large person
    area = (det['boxes'][:, 2] - det['boxes'][:, 0]) * (det['boxes'][:, 3] - det['boxes'][:, 1])
    ok = (det['labels'] == 1) & (det['scores'] > 0.98)
    i = int(torch.argmax(torch.where(ok, area, torch.zeros_like(area))))
    box = det['boxes'][i]
    images, _ = model.transform([F.to_tensor(img)])
    s = images.image_sizes[0][1] / W
    feats = model.backbone(images.tensors)
    rh = model.roi_heads
    m = rh.mask_predictor(rh.mask_head(rh.mask_roi_pool(feats, [box[None] * s], images.image_sizes)))  # 1 × 91 × 28 × 28
    p = m[0, 1].sigmoid().numpy()                                                                       # class 1: person
Image.fromarray((p * 255).astype(np.uint8)).save(OUT / 'mask28.png')
json.dump({'box': [round(float(v), 1) for v in box], 'score': round(float(det['scores'][i]), 3), 'size': [W, H], 'mask': list(p.shape)},
          open(OUT / 'mask28.json', 'w'))
print('box', [round(float(v)) for v in box], 'score', float(det['scores'][i]), 'mask', p.shape, 'mean', p.mean().round(2))
