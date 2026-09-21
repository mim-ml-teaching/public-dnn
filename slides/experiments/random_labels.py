"""Zhang et al. (2017) on a small scale: an MLP fits true labels AND randomly shuffled labels.

Same network, same optimizer, same data; only the labels differ. Records train/test accuracy per epoch.
Output: assets/lecture01/data/random_labels.json

Run: .venv/bin/python experiments/random_labels.py
"""
import json
import pathlib
import time

import torch
import torch.nn as nn
from torchvision import datasets

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture01/data/random_labels.json'
N_TRAIN, EPOCHS, SEED = 5000, 300, 0

torch.manual_seed(SEED)
data = ROOT / 'experiments/data'
train = datasets.MNIST(data, train=True, download=True)
test = datasets.MNIST(data, train=False, download=True)
prep = lambda ds: (ds.data.float().div(255).sub(0.1307).div(0.3081).flatten(1), ds.targets.clone())
Xtr, ytr = prep(train)
Xte, yte = prep(test)
idx = torch.randperm(len(Xtr))[:N_TRAIN]
Xtr, ytr = Xtr[idx], ytr[idx]


def run(labels, name):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    hist = {'epoch': [], 'train_acc': [], 'test_acc': [], 'train_loss': []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), labels[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xtr)
            tr_loss = loss_fn(out, labels).item()
            tr_acc = (out.argmax(1) == labels).float().mean().item()
            te_acc = (model(Xte).argmax(1) == yte).float().mean().item()
        hist['epoch'].append(ep); hist['train_acc'].append(round(tr_acc, 4))
        hist['test_acc'].append(round(te_acc, 4)); hist['train_loss'].append(round(tr_loss, 4))
        if ep % 25 == 0 or ep == 1:
            print(f'{name:>6} ep {ep:3d}  train {tr_acc:.3f}  test {te_acc:.3f}  loss {tr_loss:.4f}  ({time.time() - t0:.0f}s)', flush=True)
    return hist, n_params


true_hist, n_params = run(ytr, 'true')
g = torch.Generator().manual_seed(1)
random_hist, _ = run(torch.randint(0, 10, (N_TRAIN,), generator=g), 'random')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    'setup': {'dataset': 'MNIST', 'n_train': N_TRAIN, 'n_test': len(Xte), 'model': 'MLP 784-512-512-10 (ReLU)',
              'n_params': n_params, 'optimizer': 'SGD lr=0.01 momentum=0.9, batch 128', 'epochs': EPOCHS},
    'true_labels': true_hist, 'random_labels': random_hist,
}, indent=1))
print('wrote', OUT)
