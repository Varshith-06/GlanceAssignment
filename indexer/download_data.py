"""Subsample the local Fashionpedia val_test2020 image pool into data/raw.

The assignment dataset ships as a flat folder of images (no annotation JSON),
so "download" means: deterministically sample `index.num_images` files into
data/raw with stable ids. Idempotent — existing files are skipped.

Spec §4.1 requires the subsample to be *spread across the three axes*
(environment, clothing type, colour). A purely random 600-sample of this pool
contains e.g. zero red ties and zero yellow raincoats, which would make two of
the five eval queries unmeasurable. So sampling is diversity-aware:

  random base (seeded)  ∪  top-M zero-shot FashionCLIP hits per axis probe

The probes cover the assignment's axes generically (colour×garment, formality,
environment) — they are corpus curation, not query-time logic; retrieval never
sees them. Full-pool embeddings are cached (data/pool_fashionclip.npz) so this
costs one CPU pass ever.
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config

# One probe per corpus axis cell we must cover (spec: environment ×
# clothing type × colour). Top-M images per probe join the sample.
DIVERSITY_PROBES = [
    "a person wearing a bright yellow raincoat or yellow jacket",
    "a person wearing a red necktie and a white dress shirt",
    "a person wearing a blue shirt",
    "formal business attire in a modern office",
    "a person sitting on a park bench outdoors",
    "casual weekend streetwear in the city",
    "a person relaxing indoors at home",
    "a person wearing a colorful winter coat outdoors",
]
PROBE_TOP_M = 15


def _pool_embeddings(pool: list[Path], cache: Path) -> np.ndarray:
    """FashionCLIP embeddings of the full pool, cached to disk."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["names"]) == [p.name for p in pool]:
            return data["embs"]
    from indexer.embed import ClipEmbedder

    cfg = load_config()
    print(f"Embedding full pool ({len(pool)} images) for diversity probes "
          f"(one-time, cached)...")
    embedder = ClipEmbedder(cfg["models"]["fashion_clip"])
    embs = embedder.embed_images(pool, batch_size=cfg["index"]["batch_size"])
    np.savez(cache, names=np.array([p.name for p in pool]), embs=embs)
    return embs


def _probe_hits(pool: list[Path], embs: np.ndarray) -> list[Path]:
    from indexer.embed import ClipEmbedder

    cfg = load_config()
    embedder = ClipEmbedder(cfg["models"]["fashion_clip"])
    text = embedder.embed_text(DIVERSITY_PROBES)
    hits: list[Path] = []
    sims = embs @ text.T  # (pool, n_probes)
    for j in range(sims.shape[1]):
        for i in np.argsort(-sims[:, j])[:PROBE_TOP_M]:
            hits.append(pool[int(i)])
    return hits


def subsample() -> list[Path]:
    cfg = load_config()
    src_dir = Path(cfg["paths"]["dataset_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    pool = sorted(src_dir.glob("*.jpg"))
    if not pool:
        raise FileNotFoundError(f"No images found in {src_dir}")

    rng = random.Random(cfg["index"]["seed"])
    n = min(cfg["index"]["num_images"], len(pool))
    selected = dict.fromkeys(rng.sample(pool, n))  # ordered, unique

    cache = Path(cfg["paths"]["raw_dir"]).parent / "pool_fashionclip.npz"
    embs = _pool_embeddings(pool, cache)
    selected.update(dict.fromkeys(_probe_hits(pool, embs)))

    copied = []
    for src in selected:
        dst = raw_dir / src.name  # keep original hash-names as stable ids
        if not dst.exists():
            try:
                os.link(src, dst)  # hardlink: no duplicate bytes on same volume
            except OSError:
                shutil.copy2(src, dst)
        copied.append(dst)
    print(f"data/raw ready: {len(copied)} images "
          f"({n} random + diversity probes, pool={len(pool)}, "
          f"seed={cfg['index']['seed']})")
    return sorted(copied)


if __name__ == "__main__":
    subsample()
