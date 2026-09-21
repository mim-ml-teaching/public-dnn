"""Test-time augmentation as in AlexNet: average the predictions over 10 crops (4 corners + centre, and their
mirror images) of one photo. Uses torchvision's pretrained AlexNet and the bison photo of the lecture.

Output: assets/lecture06/data/tta_bison.json (crop boxes in pixels of the 256-px resized photo, per-crop
        probability of the true class and the top-1 class of each crop, the averaged prediction)
Run:    .venv/bin/python experiments/tta.py
"""
import json
import pathlib

import torch
from PIL import Image, ImageOps
from torchvision import models
from torchvision.transforms import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture06/data/tta_bison.json'
w = models.AlexNet_Weights.IMAGENET1K_V1
net = models.alexnet(weights=w).eval()
cats = w.meta['categories']
TRUE = cats.index('bison')

img = Image.open(ROOT / 'assets/lecture06/bison.jpg').convert('RGB')
img = F.resize(img, 256)                              # shorter side 256, as in AlexNet
W, H = img.size
S = 224
boxes = {'top-left': (0, 0), 'top-right': (W - S, 0), 'bottom-left': (0, H - S), 'bottom-right': (W - S, H - S),
         'centre': ((W - S) // 2, (H - S) // 2)}
crops = []
for name, (x, y) in boxes.items():
    c = img.crop((x, y, x + S, y + S))
    for flip in (False, True):
        crops.append({'name': name, 'flip': flip, 'box': [x, y, S, S], 'img': ImageOps.mirror(c) if flip else c})
x = torch.stack([F.normalize(F.to_tensor(c['img']), w.transforms().mean, w.transforms().std) for c in crops])
with torch.no_grad():
    p = net(x).softmax(1)
avg = p.mean(0)
res = {'size': [W, H], 'true': 'bison', 'crops': [], 'avg_true': avg[TRUE].item(),
       'avg_top': [(cats[i], round(avg[i].item(), 3)) for i in avg.topk(3).indices]}
for c, pi in zip(crops, p):
    top = pi.argmax().item()
    res['crops'].append({'name': c['name'], 'flip': c['flip'], 'box': c['box'], 'p_true': round(pi[TRUE].item(), 3),
                         'top1': cats[top], 'p_top1': round(pi[top].item(), 3)})
    print(f"{c['name']:13s} flip={c['flip']!s:5s} p(bison)={pi[TRUE]:.3f}  top-1 {cats[top]} ({pi[top]:.3f})")
print('average p(bison) =', round(res['avg_true'], 3), res['avg_top'])
OUT.write_text(json.dumps(res, indent=1))
