"""Shuffle the pixels of every image with one fixed permutation: does the network care?

An MLP and a small convnet are trained on the original images and on the shuffled images (the same
permutation for every image, train and test). For the MLP the two problems are the same problem up to a
reordering of the first layer's weights; the convnet relies on neighbouring pixels being related.
Records test accuracy per epoch, and saves a few example images (original and shuffled).

Output: assets/lecture06/data/shuffled_pixels_<dataset>.json, assets/lecture06/shuffled_<dataset>.png
Run:    .venv/bin/python experiments/shuffled_pixels.py [mnist|cifar10]
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'
OUT = ROOT / f'assets/lecture06/data/shuffled_pixels_{DATASET}.json'
EPOCHS, BATCH, SEED = (8 if DATASET == 'mnist' else 15), 128, 0
torch.set_num_threads(max(1, torch.get_num_threads()))

data = ROOT / 'experiments/data'
if DATASET == 'mnist':
    tr, te = datasets.MNIST(data, train=True, download=True), datasets.MNIST(data, train=False, download=True)
    to_t = lambda ds: (ds.data.float().div(255).unsqueeze(1), ds.targets.clone())
else:
    tr, te = datasets.CIFAR10(data, train=True, download=True), datasets.CIFAR10(data, train=False, download=True)
    to_t = lambda ds: (torch.tensor(ds.data).float().div(255).permute(0, 3, 1, 2), torch.tensor(ds.targets))
Xtr, ytr = to_t(tr)
Xte, yte = to_t(te)
mean, std = Xtr.mean(dim=(0, 2, 3), keepdim=True), Xtr.std(dim=(0, 2, 3), keepdim=True)
Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
C, H, W = Xtr.shape[1:]

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(H * W, generator=g)          # one fixed permutation of pixel positions


def shuffle(X):
    return X.flatten(2)[:, :, perm].reshape(X.shape)


def mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(C * H * W, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                         nn.Linear(512, 10))


def cnn():
    s = H // 4                                      # spatial size after two 2x2 poolings: 7 (MNIST) or 8 (CIFAR)
    return nn.Sequential(
        nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * s * s, 256), nn.ReLU(), nn.Linear(256, 10))


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return sum((model(X[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(X), 1000)) / len(X)


def run(make, shuffled):
    torch.manual_seed(SEED)
    model = make()
    Xa, Xb = (shuffle(Xtr), shuffle(Xte)) if shuffled else (Xtr, Xte)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    hist = {'epoch': [0], 'test_acc': [accuracy(model, Xb, yte)]}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xa))
        for i in range(0, len(Xa), BATCH):
            b = idx[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(Xa[b]), ytr[b]).backward()
            opt.step()
        hist['epoch'].append(ep)
        hist['test_acc'].append(accuracy(model, Xb, yte))
        print(f'{make.__name__} shuffled={shuffled} epoch {ep}: test {hist["test_acc"][-1]:.4f} ({time.time() - t0:.0f}s)', flush=True)
    hist['params'] = sum(p.numel() for p in model.parameters())
    return hist


res = {'dataset': DATASET, 'epochs': EPOCHS, 'optimizer': 'Adam lr=1e-3', 'batch': BATCH}
for make in (mlp, cnn):
    res[make.__name__] = {'original': run(make, False), 'shuffled': run(make, True)}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print('wrote', OUT)

# example images: top row original, bottom row shuffled (same permutation), 8 test images, upscaled
raw = (te.data if DATASET == 'mnist' else torch.tensor(te.data))
raw = raw.numpy() if hasattr(raw, 'numpy') else np.asarray(raw)
if raw.ndim == 3:
    raw = raw[..., None]
pick = [int(np.where(np.asarray(yte) == c)[0][0]) for c in range(8)]
rows = []
for shuffled in (False, True):
    tiles = []
    for i in pick:
        img = raw[i].reshape(H * W, -1)
        if shuffled:
            img = img[perm.numpy()]
        tiles.append(img.reshape(H, W, -1))
    rows.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=0) for t in tiles], axis=1))
grid = np.concatenate(rows, axis=0).astype(np.uint8)
im = Image.fromarray(grid.squeeze(-1) if grid.shape[-1] == 1 else grid)
im = im.resize((im.width * 6, im.height * 6), Image.NEAREST)
png = ROOT / f'assets/lecture06/shuffled_{DATASET}.png'
im.save(png)
print('wrote', png)
