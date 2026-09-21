"""Layer-by-layer sizes, parameter counts and activation memory of the classic ImageNet architectures.

Models are built by torchvision without pretrained weights (nothing is downloaded); the ImageNet accuracies
are torchvision's published numbers for its pretrained weights (`weights.meta`). LeNet-5 is defined here
(the 1998 layout with max pooling and ReLU instead of subsampling layers and tanh).

Output: assets/lecture06/data/arch_stats.json
Run:    .venv/bin/python experiments/arch_stats.py
"""
import json
import pathlib

import torch
import torch.nn as nn
from torchvision import models

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/lecture06/data/arch_stats.json'


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def leaf_table(model, x):
    """Output shape and parameter count of every conv / linear / pooling layer, in forward order."""
    rows = []

    def hook(mod, inp, out):
        rows.append({'type': type(mod).__name__, 'out': list(out.shape[1:]), 'params': n_params(mod),
                     'act': out[0].numel(),
                     # multiply-adds: every output value of a conv / linear layer costs one weight-row dot product
                     'macs': (out[0].numel() * (mod.weight[0].numel()) if isinstance(mod, (nn.Conv2d, nn.Linear)) else 0),
                     'kernel': getattr(mod, 'kernel_size', None), 'stride': getattr(mod, 'stride', None)})
    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return rows


def flops(model, x):
    """Multiply-accumulates of conv and linear layers."""
    tot = 0

    def hook(mod, inp, out):
        nonlocal tot
        if isinstance(mod, nn.Conv2d):
            tot += out[0].numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            tot += mod.in_features * mod.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return tot


lenet = nn.Sequential(
    nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

res = {'lenet5': {'input': [1, 32, 32], 'layers': leaf_table(lenet, torch.zeros(1, 1, 32, 32)),
                  'params': n_params(lenet), 'macs': flops(lenet, torch.zeros(1, 1, 32, 32))}}

x = torch.zeros(1, 3, 224, 224)
specs = {
    'alexnet': (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
    'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    'googlenet': (lambda: models.googlenet(aux_logits=False, init_weights=False), models.GoogLeNet_Weights.IMAGENET1K_V1),
    'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
}
for name, (make, w) in specs.items():
    m = make().eval()
    acc = w.meta['_metrics']['ImageNet-1K']
    res[name] = {'params': n_params(m), 'macs': flops(m, x), 'top1': acc['acc@1'], 'top5': acc['acc@5'],
                 'layers': leaf_table(m, x) if name in ('alexnet', 'vgg16') else None}
    if name.startswith('resnet'):
        stages = {'stem': n_params(m.conv1) + n_params(m.bn1)}
        for k in ('layer1', 'layer2', 'layer3', 'layer4'):
            stages[k] = {'blocks': len(getattr(m, k)), 'params': n_params(getattr(m, k))}
        stages['fc'] = n_params(m.fc)
        res[name]['stages'] = stages
    print(f"{name:10s} params {res[name]['params'] / 1e6:6.1f}M  GMACs {res[name]['macs'] / 1e9:5.2f}  "
          f"top-1 {res[name]['top1']:.1f}  top-5 {res[name]['top5']:.1f}")

# bottleneck vs plain: the 5x5 branch of an inception module (28x28x256 input, 32 outputs)
res['inception_example'] = {'direct_5x5': 5 * 5 * 256 * 32, 'reduce_1x1_then_5x5': 256 * 16 + 5 * 5 * 16 * 32}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print('wrote', OUT)
