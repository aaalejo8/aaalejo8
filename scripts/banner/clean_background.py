#!/usr/bin/env python3
"""Remove the title/subtitle baked into the original Salta landscape.

The original banner (assets/source/salta-original.png) has "Alejo Almada" and
the subtitle painted into the pixels. The animated banner draws its own text,
so we need a clean plate. This script:

1. builds a mask of the bright glyph pixels inside the two text boxes
   (pixel brighter than its local median background),
2. fills them with multi-scale normalised convolution (smooth, halo-free),
3. re-adds film grain so the patch matches the photo texture.

Run once from the repository root (output is committed):
    python scripts/banner/clean_background.py
Requires: pip install -r scripts/banner/requirements.txt  (opencv-python-headless)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "assets/source/salta-original.png"
OUT = ROOT / "assets/source/salta-clean.png"

# Text boxes in original-image pixels (2172 x 724): name, subtitle.
BOXES = [(735, 232, 1440, 352), (610, 358, 1565, 410)]


def text_mask(img: np.ndarray) -> np.ndarray:
    value = img.max(axis=2).astype(np.int16)
    background = cv2.medianBlur(img.max(axis=2), 41).astype(np.int16)
    raw = ((value - background) > 10).astype(np.uint8) * 255
    mask = np.zeros_like(raw)
    for x0, y0, x1, y1 in BOXES:
        mask[y0:y1, x0:x1] = raw[y0:y1, x0:x1]
    # Close glyph interiors and swallow anti-aliased halos.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask


def normalized_fill(img: np.ndarray, keep: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Blur only the known pixels (normalised convolution) to estimate the hidden ones."""
    w = keep.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)[..., None], den


def main() -> None:
    img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Missing {SRC}")
    mask = text_mask(img)
    keep = mask == 0

    # Multi-scale normalised convolution: fine scale where real pixels are
    # close, wider scales deeper inside thick glyphs. Smooth, halo-free fill.
    filled = np.zeros(img.shape, np.float32)
    weight_left = np.ones(img.shape[:2], np.float32)
    for sigma in (4, 9, 20, 45):
        est, den = normalized_fill(img, keep, sigma)
        trust = np.clip(den * 2.2, 0, 1) * weight_left
        filled += est * trust[..., None]
        weight_left -= trust
    filled += est * weight_left[..., None]

    # Grain matched to the surrounding photo noise.
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.normal(0, 4.0, img.shape[:2]).astype(np.float32), (0, 0), 0.7)
    filled = filled + grain[..., None]

    soft = cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32)[..., None] / 255.0
    soft = np.maximum(soft, (mask > 0)[..., None].astype(np.float32))
    out = img.astype(np.float32) * (1 - soft) + filled * soft
    cv2.imwrite(str(OUT), np.clip(out, 0, 255).astype(np.uint8))
    print(f"wrote {OUT.relative_to(ROOT)} (masked {int((mask > 0).sum()):,} px)")


if __name__ == "__main__":
    main()
