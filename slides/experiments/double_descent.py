"""Double descent (Belkin et al. 2019) with random ReLU features on MNIST.

Model: f(x) = W2 · relu(W1 x), W1 random and fixed (p hidden units), W2 fitted by least squares on one-hot
targets; when p > n we take the minimum-norm interpolating solution (what gradient descent from zero finds).
We sweep the width p across the interpolation threshold p = n and record train/test error.
Output: assets/lecture01/data/double_descent.json

Run: .venv/bin/python experiments/double_descent.py
"""
import json
import pathlib

import torch
from torchvision import datasets

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture01/data/double_descent.json'
N_TRAIN, N_TEST, SEEDS = 1000, 10000, 5
WIDTHS = sorted({int(round(10 ** (k / 16))) for k in range(16, 16 * 4 + 6)})  # 10 … ~20000, log-spaced

torch.set_default_dtype(torch.float64)
data = ROOT / 'experiments/data'
train = datasets.MNIST(data, train=True, download=True)
test = datasets.MNIST(data, train=False, download=True)
prep = lambda ds: (ds.data.double().div(255).flatten(1), ds.targets.clone())
Xall, yall = prep(train)
Xte, yte = prep(test)
Xte, yte = Xte[:N_TEST], yte[:N_TEST]
onehot = lambda y: torch.nn.functional.one_hot(y, 10).double()


def errors(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(Xall), generator=g)[:N_TRAIN]
    Xtr, ytr = Xall[idx], yall[idx]
    Wbig = torch.randn(784, max(WIDTHS), generator=g) / 784 ** 0.5
    res = []
    for p in WIDTHS:
        F, Fte = torch.relu(Xtr @ Wbig[:, :p]), torch.relu(Xte @ Wbig[:, :p])
        W2 = torch.linalg.pinv(F, rcond=1e-10) @ onehot(ytr)   # least squares; minimum norm when p > n
        tr, te = F @ W2, Fte @ W2
        res.append({
            'train_err': (tr.argmax(1) != ytr).double().mean().item(),
            'test_err': (te.argmax(1) != yte).double().mean().item(),
            'train_mse': ((tr - onehot(ytr)) ** 2).sum(1).mean().item(),
            'test_mse': ((te - onehot(yte)) ** 2).sum(1).mean().item(),
            'w_norm': W2.norm().item(),
            'cond': (lambda sv: (sv[0] / sv[-1]).item())(torch.linalg.svdvals(F)),
        })
        print(f'seed {seed} p {p:6d}  train err {res[-1]["train_err"]:.3f}  test err {res[-1]["test_err"]:.3f}  test mse {res[-1]["test_mse"]:.3f}', flush=True)
    return res


runs = [errors(s) for s in range(SEEDS)]
avg = lambda key: [round(sum(r[i][key] for r in runs) / SEEDS, 5) for i in range(len(WIDTHS))]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    'setup': {'dataset': 'MNIST', 'n_train': N_TRAIN, 'n_test': N_TEST, 'seeds': SEEDS,
              'model': 'random ReLU features (fixed first layer) + min-norm least squares on one-hot targets'},
    'width': WIDTHS, 'n_params': [10 * p for p in WIDTHS],
    'train_err': avg('train_err'), 'test_err': avg('test_err'),
    'train_mse': avg('train_mse'), 'test_mse': avg('test_mse'),
    'w_norm': avg('w_norm'), 'cond': avg('cond'),
}, indent=1))
print('wrote', OUT)
