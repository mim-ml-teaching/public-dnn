"""Object detection and instance segmentation with torchvision's pretrained COCO models.

1. Faster R-CNN (ResNet-50-FPN v2) on the crowd photo with non-maximum suppression switched off: the raw,
   overlapping boxes for the NMS slide (the widget runs NMS in the browser), plus the anchor count.
2. Mask R-CNN (ResNet-50-FPN v2) on the pears photo: boxes, labels and masks ("pear" is not a COCO class).
3. Mask R-CNN on the crowd photo (instance masks).
4. A comparison of torchvision detectors: COCO box AP (torchvision's published numbers for the weights),
   parameters, and our CPU time on the crowd photo (median of 3 runs, 16 threads).

Output: assets/lecture07/data/crowd_raw.json, pears_maskrcnn.json, crowd_maskrcnn.json, detectors.json
Run:    .venv/bin/python experiments/detection.py
"""
import json
import pathlib
import statistics
import time

import numpy as np
import torch
from PIL import Image
from torchvision.models import detection as D
from torchvision.transforms import functional as F

from lecture07_common import mask_json

torch.set_num_threads(16)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture07/data'
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    img = Image.open(ROOT / f'assets/lecture07/{name}.jpg').convert('RGB')
    return img, F.to_tensor(img)


def r(v, d=1):
    return [round(float(a), d) for a in v]


crowd, xc = load('crowd')
pears, xp = load('pears')

# ---- 1. raw boxes, NMS off ----
w = D.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
cats = w.meta['categories']
net = D.fasterrcnn_resnet50_fpn_v2(weights=w, box_nms_thresh=1.0, box_score_thresh=0.3, box_detections_per_img=1000).eval()
with torch.no_grad():
    out = net([xc])[0]
    images, _ = net.transform([xc])
    feats = net.backbone(images.tensors)
    anchors = net.rpn.anchor_generator(images, list(feats.values()))[0]
scale = crowd.size[0] / images.image_sizes[0][1]
boxes = [{'b': r(b), 's': round(float(s), 3), 'c': cats[l]} for b, s, l in zip(out['boxes'], out['scores'], out['labels'])]
per_level = [int(f.shape[-2] * f.shape[-1] * 3) for f in feats.values()]
json.dump({'size': list(crowd.size), 'model': 'torchvision fasterrcnn_resnet50_fpn_v2 (COCO), box_nms_thresh=1.0, score ≥ 0.3',
           'n_anchors': int(anchors.shape[0]), 'anchors_per_level': per_level,
           'feature_sizes': [list(f.shape[-2:]) for f in feats.values()], 'resize_scale': round(scale, 4),
           'boxes': boxes}, open(OUT / 'crowd_raw.json', 'w'))
print(f'crowd: {len(boxes)} raw boxes (score ≥ 0.3), {anchors.shape[0]} anchors {per_level}, net input {images.image_sizes[0]}')

# ---- 2./3. Mask R-CNN ----
w = D.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
mnet = D.maskrcnn_resnet50_fpn_v2(weights=w).eval()


def maskrcnn(img, x, name, thr):
    with torch.no_grad():
        o = mnet([x])[0]
    keep = o['scores'] >= thr
    objs = []
    for b, s, l, m in zip(o['boxes'][keep], o['scores'][keep], o['labels'][keep], o['masks'][keep]):
        objs.append({'b': r(b), 's': round(float(s), 3), 'c': cats[l], 'mask': mask_json((m[0] > 0.5).numpy())})
    json.dump({'size': list(img.size), 'model': 'torchvision maskrcnn_resnet50_fpn_v2 (COCO)', 'score_min': thr,
               'objects': objs}, open(OUT / f'{name}_maskrcnn.json', 'w'))
    print(name, [(o_['c'], o_['s']) for o_ in objs])


maskrcnn(pears, xp, 'pears', 0.5)
maskrcnn(crowd, xc, 'crowd', 0.7)

# ---- 4. detector comparison ----
MODELS = [
    ('SSDlite (MobileNetV3)', 'one-stage', D.ssdlite320_mobilenet_v3_large, D.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    ('SSD300 (VGG-16)', 'one-stage', D.ssd300_vgg16, D.SSD300_VGG16_Weights.DEFAULT),
    ('RetinaNet (R50-FPN)', 'one-stage', D.retinanet_resnet50_fpn, D.RetinaNet_ResNet50_FPN_Weights.DEFAULT),
    ('FCOS (R50-FPN)', 'anchor-free', D.fcos_resnet50_fpn, D.FCOS_ResNet50_FPN_Weights.DEFAULT),
    ('Faster R-CNN (R50-FPN)', 'two-stage', D.fasterrcnn_resnet50_fpn, D.FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    ('Faster R-CNN v2 (R50-FPN)', 'two-stage', D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    ('Mask R-CNN v2 (R50-FPN)', 'two-stage + masks', D.maskrcnn_resnet50_fpn_v2, D.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
]
rows = []
for name, kind, fn, wt in MODELS:
    m = fn(weights=wt).eval()
    with torch.no_grad():
        m([xc])
        ts = []
        for _ in range(3):
            t = time.perf_counter(); o = m([xc])[0]; ts.append(time.perf_counter() - t)
    metrics = wt.meta['_metrics']['COCO-val2017']
    row = {'name': name, 'kind': kind, 'box_map': metrics['box_map'], 'mask_map': metrics.get('mask_map'),
           'params_M': round(sum(p.numel() for p in m.parameters()) / 1e6, 1), 'gflops': wt.meta.get('_ops'),
           'cpu_ms': round(statistics.median(ts) * 1000), 'n_det_05': int((o['scores'] >= 0.5).sum())}
    rows.append(row)
    print(row)
json.dump({'image': 'crowd.jpg', 'note': 'box_map/mask_map/gflops: torchvision weight metadata (COCO val2017); cpu_ms: our run, 16 threads',
           'torch': torch.__version__, 'models': rows}, open(OUT / 'detectors.json', 'w'), indent=1)
