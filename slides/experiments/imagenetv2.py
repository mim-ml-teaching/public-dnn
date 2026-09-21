"""Do ImageNet classifiers generalize to a *new* ImageNet test set? (Recht et al., 2019, on a small scale.)

ImageNetV2 (matched-frequency, 10,000 images) was collected years later with the original procedure.
We evaluate torchvision's pretrained models on it and compare with their accuracy on the original
validation set (torchvision's published numbers, `weights.meta`).

Output: assets/lecture06/data/imagenetv2.json
Run:    .venv/bin/python experiments/imagenetv2.py
        (downloads ImageNetV2, 1.26 GB, to experiments/data/, and pretrained weights, ~1 GB, to ~/.cache/torch)
"""
import json
import pathlib
import tarfile
import time
import urllib.request

import torch
from PIL import Image
from torchvision import models

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture06/data/imagenetv2.json'
DATA = ROOT / 'experiments/data'
URL = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'
TAR = DATA / 'imagenetv2-matched-frequency.tar.gz'
DIR = DATA / 'imagenetv2-matched-frequency-format-val'

if not DIR.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    if not TAR.exists():
        print('downloading', URL, flush=True)
        urllib.request.urlretrieve(URL, TAR)
    with tarfile.open(TAR) as t:
        t.extractall(DATA, filter='data')
files = sorted((int(p.parent.name), p) for p in DIR.glob('*/*'))   # folder name = class index 0..999
print(len(files), 'images', flush=True)

MODELS = {
    'AlexNet (2012)': (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    'VGG-16 (2014)': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    'GoogLeNet (2014)': (models.googlenet, models.GoogLeNet_Weights.IMAGENET1K_V1),
    'ResNet-18 (2015)': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'ResNet-50 (2015)': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    'ResNet-152 (2015)': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}
res = OUT.exists() and json.loads(OUT.read_text()) or {'models': {}}
for name, (make, w) in MODELS.items():
    if name in res['models']:
        continue
    net = make(weights=w).eval()
    tf = w.transforms()
    correct, t0 = 0, time.time()
    with torch.no_grad():
        for i in range(0, len(files), 100):
            chunk = files[i:i + 100]
            x = torch.stack([tf(Image.open(p).convert('RGB')) for _, p in chunk])
            y = torch.tensor([c for c, _ in chunk])
            correct += (net(x).argmax(1) == y).sum().item()
    acc = w.meta['_metrics']['ImageNet-1K']
    res['models'][name] = {'orig_top1': acc['acc@1'] / 100, 'v2_top1': correct / len(files)}
    print(f'{name}: original {acc["acc@1"]:.1f}%  v2 {100 * correct / len(files):.1f}%  ({time.time() - t0:.0f}s)', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
print('wrote', OUT)
