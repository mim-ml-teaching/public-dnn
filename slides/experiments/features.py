"""What do the layers of a trained convnet compute? Two figures from torchvision's pretrained ImageNet models.

1. The 64 first-layer filters of AlexNet (11x11x3 each), shown as small RGB images.
2. Feature maps of ResNet-18 for one photo (Chelsea the cat, CC0, from scikit-image): after the first conv
   and after each of the four stages. For every depth the channels with the largest spatial contrast are
   shown; the maps get coarser and more selective with depth.

Output: assets/lecture06/alexnet_conv1.png, assets/lecture06/fmap_<layer>.png, assets/lecture06/data/fmaps.json
Run:    .venv/bin/python experiments/features.py      (downloads the pretrained weights, ~280 MB, to ~/.cache/torch)
"""
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from torchvision import models

ROOT = pathlib.Path(__file__).resolve().parents[1]
A = ROOT / 'assets/lecture06'

# ---- 1. AlexNet first-layer filters ----
alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
w = alex.features[0].weight.detach().numpy()              # (64, 3, 11, 11)
tiles = []
for f in w:
    f = f.transpose(1, 2, 0)
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)          # each filter scaled to [0, 1] separately
    tiles.append(np.pad(f, ((1, 1), (1, 1), (0, 0)), constant_values=0.08))
rows = [np.concatenate(tiles[r * 16:(r + 1) * 16], axis=1) for r in range(4)]
grid = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
im = Image.fromarray(grid)
im.resize((im.width * 8, im.height * 8), Image.NEAREST).save(A / 'alexnet_conv1.png')

# ---- 2. ResNet-18 feature maps for one photo ----
weights = models.ResNet18_Weights.IMAGENET1K_V1
net = models.resnet18(weights=weights).eval()
img = Image.open(A / 'chelsea.jpg').convert('RGB')
x = weights.transforms(crop_size=224, resize_size=224)(img).unsqueeze(0)
acts = {}
for name in ('relu', 'layer1', 'layer2', 'layer3', 'layer4'):
    getattr(net, name).register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o[0].detach()))
with torch.no_grad():
    logits = net(x)
top = logits.softmax(1)[0].topk(3)
labels = [(weights.meta['categories'][i], round(p.item(), 3)) for p, i in zip(top.values, top.indices)]

meta = {'labels': labels, 'layers': []}
for name, a in acts.items():
    C, H, W = a.shape
    score = a.flatten(1).std(1)                             # channels with the most spatial contrast
    ch = score.topk(6).indices.tolist()
    tiles = []
    for c in ch:
        m = a[c].numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        t = Image.fromarray((m * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST)
        tiles.append(np.pad(np.asarray(t), 3, constant_values=40))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(A / f'fmap_{name}.png')
    meta['layers'].append({'name': name, 'shape': [C, H, W], 'channels': ch})
    print(name, (C, H, W), ch)
(A / 'data/fmaps.json').write_text(json.dumps(meta, indent=1))
print('top-3:', labels)
