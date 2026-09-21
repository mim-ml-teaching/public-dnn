"""Why a detector cannot say "pear": the classification head of Mask R-CNN v2 (torchvision, COCO) is a softmax over
91 outputs (background + the COCO categories). We give it the true box of each fruit (from SAM, pears_sam.json) as
the region proposal and read the whole softmax: the probability has to go to a class it knows, or to background.

Output: assets/lecture07/data/pears_softmax.json
Run:    .venv/bin/python experiments/pear_softmax.py
"""
import json
import pathlib

import torch
from PIL import Image
from torchvision.models import detection as D
from torchvision.transforms import functional as F

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
W8 = D.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = D.maskrcnn_resnet50_fpn_v2(weights=W8).eval()
cats = W8.meta['categories']
img = Image.open(ROOT / 'assets/lecture07/pears.jpg').convert('RGB')
objs = json.load(open(OUT / 'pears_sam.json'))['objects']
with torch.no_grad():
    images, _ = model.transform([F.to_tensor(img)])
    sy, sx = images.image_sizes[0][0] / img.size[1], images.image_sizes[0][1] / img.size[0]
    feats = model.backbone(images.tensors)
    props = [torch.tensor([[o['b'][0] * sx, o['b'][1] * sy, o['b'][2] * sx, o['b'][3] * sy] for o in objs], dtype=torch.float32)]
    bf = model.roi_heads.box_head(model.roi_heads.box_roi_pool(feats, props, images.image_sizes))
    logits, _ = model.roi_heads.box_predictor(bf)
    P = logits.softmax(-1)
named = [i for i, c in enumerate(cats) if c != 'N/A']
res = []
for o, p in zip(objs, P):
    order = sorted(named, key=lambda i: -float(p[i]))
    top = [{'c': 'background' if cats[i] == '__background__' else cats[i], 'p': round(float(p[i]), 3)} for i in order[:5]]
    res.append({'name': o['name'], 'b': o['b'], 'top': top})
    print(o['name'], top)
json.dump({'model': 'torchvision maskrcnn_resnet50_fpn_v2 (COCO), box head softmax on the true boxes',
           'n_outputs': int(P.shape[1]), 'n_classes': len(named) - 1, 'objects': res}, open(OUT / 'pears_softmax.json', 'w'), indent=1)
