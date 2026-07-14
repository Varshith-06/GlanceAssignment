"""Shared helpers: config loading and the named-colour palette in LAB space."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths relative to the project root so scripts work from any CWD.
    cfg["paths"] = {k: str(PROJECT_ROOT / v) for k, v in cfg["paths"].items()
                    if k != "collection_name"} | {
                        "collection_name": cfg["paths"]["collection_name"]}
    return cfg


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) uint8/float 0-255 -> LAB (skimage convention)."""
    from skimage.color import rgb2lab

    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    return rgb2lab(arr.reshape(1, -1, 3)).reshape(arr.shape)


@functools.lru_cache(maxsize=1)
def palette_lab() -> dict[str, np.ndarray]:
    """Named palette from config.yaml converted once to LAB."""
    cfg = load_config()
    return {name: rgb_to_lab(np.array(rgb, dtype=np.float64))
            for name, rgb in cfg["colors"].items()}


def nearest_palette_color(lab: np.ndarray) -> tuple[str, float]:
    """Nearest named colour by Euclidean distance in LAB (delta-E CIE76)."""
    best_name, best_d = "", float("inf")
    for name, ref in palette_lab().items():
        d = float(np.linalg.norm(np.asarray(lab) - ref))
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


# Extended colour-word lexicon (CSS/xkcd-style RGB). The query parser resolves
# ANY of these to a continuous LAB target, so an out-of-palette word like
# "maroon" or "teal" becomes an approximate colour constraint instead of being
# silently dropped (which would give full credit to any colour). Index-side
# colour NAMING still uses the compact config palette — captions and metadata
# stay stable; only query-side scoring is continuous.
EXTENDED_COLORS: dict[str, tuple[int, int, int]] = {
    "maroon": (128, 0, 32), "burgundy": (101, 0, 17), "crimson": (220, 20, 60),
    "scarlet": (255, 36, 0), "brick": (178, 34, 34), "rust": (183, 65, 14),
    "salmon": (250, 128, 114), "coral": (255, 127, 80), "peach": (255, 203, 164),
    "teal": (0, 128, 128), "turquoise": (64, 224, 208), "cyan": (0, 180, 200),
    "mint": (152, 224, 172), "olive": (128, 128, 0), "lime": (160, 220, 50),
    "forest": (34, 100, 34), "emerald": (80, 200, 120),
    "indigo": (75, 0, 130), "violet": (143, 0, 255), "lavender": (181, 156, 220),
    "magenta": (200, 30, 140), "fuchsia": (255, 0, 255), "plum": (142, 69, 133),
    "mustard": (225, 173, 1), "gold": (212, 175, 55), "khaki": (195, 176, 145),
    "tan": (210, 180, 140), "cream": (255, 253, 208), "ivory": (255, 255, 240),
    "charcoal": (54, 69, 79), "silver": (192, 192, 192),
}


def resolve_color_lab(word: str) -> np.ndarray | None:
    """Colour word -> LAB target. Palette names use the config palette (so an
    exact-name match with indexed metadata stays consistent); anything else
    falls back to the extended lexicon. None if the word isn't a colour."""
    if word in palette_lab():
        return palette_lab()[word]
    if word in EXTENDED_COLORS:
        return rgb_to_lab(np.array(EXTENDED_COLORS[word], dtype=np.float64))
    return None
