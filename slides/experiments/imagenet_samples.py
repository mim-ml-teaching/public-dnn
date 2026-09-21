"""Sample images for the ImageNet slide (lecture 6).

Taken from ImageNetV2 (Recht et al., 2019; matched-frequency), a test set collected from Flickr with the
original ImageNet procedure and the same 1,000 classes; already downloaded by `imagenetv2.py`.
Each image is centre-cropped to a square and resized to 192 px.

Output: assets/lecture06/imagenet/<class>.jpg
Run:    .venv/bin/python experiments/imagenet_samples.py   (after imagenetv2.py)
"""
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / 'experiments/data/imagenetv2-matched-frequency-format-val'
OUT = ROOT / 'assets/lecture06/imagenet'
SIZE = 192

# (ImageNet class index, file name, position in the sorted folder of that class)
PICKS = [
    (1, 'goldfish', 0), (281, 'tabby', 2), (340, 'zebra', 3), (417, 'balloon', 2), (779, 'school_bus', 6),
    (510, 'container_ship', 6), (967, 'espresso', 3),
    (250, 'siberian_husky', 6), (249, 'malamute', 2), (248, 'eskimo_dog', 1),
]

OUT.mkdir(parents=True, exist_ok=True)
for idx, name, pos in PICKS:
    f = sorted((DIR / str(idx)).iterdir())[pos]
    im = Image.open(f).convert('RGB')
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    im.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / f'{name}.jpg', quality=88)
    print(name, f.name)
