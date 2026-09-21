"""Helpers shared by the lecture 7 experiments: run-length encoding of binary masks for the slides.

A mask is stored as {"h": H, "w": W, "rle": [n0, n1, n2, ...]}: row-major runs, alternating 0s and 1s,
starting with a (possibly empty) run of 0s. Decoded in shared/widgets/detview.js and sam.js.
"""
import numpy as np


def rle(mask):
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size == 0:
        return []
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    bounds = np.concatenate([[0], change, [m.size]])
    runs = np.diff(bounds).tolist()
    if m[0]:
        runs = [0] + runs
    return runs


def mask_json(mask):
    h, w = mask.shape
    return {'h': int(h), 'w': int(w), 'rle': rle(mask)}
