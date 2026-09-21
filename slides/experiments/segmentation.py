"""Semantic segmentation of the street photo with torchvision's pretrained DeepLabV3-ResNet50 (21 Pascal VOC
classes, trained on a COCO subset). Also a coarse version: the same logits average-pooled to 1/32 of the input
(like FCN-32s, which predicts at the resolution of the last pooling layer) and upsampled bilinearly; and at 1/16.

Output: assets/lecture07/data/street_seg.json  (class names and colours of the classes present)
        assets/lecture07/data/street_seg_fine.png, street_seg_mid.png, street_seg_coarse.png  (class index per pixel, 8-bit grey)
Run:    .venv/bin/python experiments/segmentation.py
"""
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms import functional as F

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
OUT.mkdir(parents=True, exist_ok=True)

w = DeepLabV3_ResNet50_Weights.DEFAULT
net = deeplabv3_resnet50(weights=w).eval()
cats = w.meta['categories']

img = Image.open(ROOT / 'assets/lecture07/street.jpg').convert('RGB')
W, H = img.size
x = F.normalize(F.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[None]
with torch.no_grad():
    feats = net.backbone(x)['out']                    # stride 8 (dilated ResNet)
    logits8 = net.classifier(feats)                   # 21 × H/8 × W/8
fine = Fn.interpolate(logits8, size=(H, W), mode='bilinear', align_corners=False).argmax(1)[0]
logits32 = Fn.avg_pool2d(logits8, 4, ceil_mode=True)  # 1/32 of the input
coarse = Fn.interpolate(logits32, size=(H, W), mode='bilinear', align_corners=False).argmax(1)[0]
logits16 = Fn.avg_pool2d(logits8, 2, ceil_mode=True)  # 1/16 of the input
mid = Fn.interpolate(logits16, size=(H, W), mode='bilinear', align_corners=False).argmax(1)[0]

COLORS = {'background': None, 'person': '#ff6b6b', 'car': '#5ab0ff', 'bus': '#f2b134', 'bicycle': '#5fd38d',
          'motorbike': '#b28dff', 'train': '#5fd38d', 'pottedplant': '#5fd38d'}
present = sorted(set(np.unique(fine.numpy())) | set(np.unique(coarse.numpy())) | set(np.unique(mid.numpy())))
classes = []
for c in present:
    name = cats[c]
    frac = float((fine == c).float().mean())
    classes.append({'id': int(c), 'name': name, 'color': COLORS.get(name, '#9aa1ae'), 'frac_fine': round(frac, 4)})
    print(f'{name:12s} fine {frac:6.2%}  coarse {float((coarse == c).float().mean()):6.2%}')

Image.fromarray(fine.numpy().astype(np.uint8)).save(OUT / 'street_seg_fine.png')
Image.fromarray(coarse.numpy().astype(np.uint8)).save(OUT / 'street_seg_coarse.png')
Image.fromarray(mid.numpy().astype(np.uint8)).save(OUT / 'street_seg_mid.png')
json.dump({'size': [W, H], 'model': 'torchvision deeplabv3_resnet50 (COCO_WITH_VOC_LABELS_V1)',
           'logits_size': list(logits8.shape[-2:]), 'coarse_size': list(logits32.shape[-2:]),
           'classes': classes}, open(OUT / 'street_seg.json', 'w'), indent=1)
print('logits', tuple(logits8.shape), 'coarse', tuple(logits32.shape))
