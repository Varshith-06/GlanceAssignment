"""Build pooled candidate contact-sheets for hand-labelling relevance.

Standard pooled evaluation: for each of the 5 queries, take the union of
top-10 results from all three ablation systems (vanilla CLIP dense,
FashionCLIP dense, hybrid), and render one numbered contact sheet per query
to eval/pool/. A human then writes the relevant image_ids into
eval/relevance.json.

Run:  python eval/label_pool.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ablation import SWAP_QUERIES, load_records, vanilla_clip_rank
from eval.run_eval import EVAL_QUERIES
from retriever.search import Searcher

ALL_QUERIES = EVAL_QUERIES | SWAP_QUERIES

POOL_DIR = Path(__file__).resolve().parent / "pool"
N = 10


def contact_sheet(ids: list[str], paths: dict[str, str], title: str,
                  out_path: Path) -> None:
    cols = 5
    rows = -(-len(ids) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.6 * rows))
    axes = axes.flatten()
    for ax in axes[len(ids):]:
        ax.axis("off")
    for ax, image_id in zip(axes, ids):
        ax.imshow(Image.open(paths[image_id]).convert("RGB"))
        ax.set_title(image_id[:12], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    POOL_DIR.mkdir(exist_ok=True)
    records = load_records()
    paths = {r["image_id"]: r["path"] for r in records}
    searcher = Searcher()

    pools = {}
    for name, query in ALL_QUERIES.items():
        ids: list[str] = []
        for results in (vanilla_clip_rank(query, records, N),
                        searcher.dense.search(query, N),
                        searcher.search(query, k=N)):
            for r in results:
                if r["image_id"] not in ids:
                    ids.append(r["image_id"])
        pools[name] = ids
        contact_sheet(ids, paths, f"[{name}] {query}",
                      POOL_DIR / f"pool_{name}.png")
        print(f"{name}: pooled {len(ids)} candidates")

    (POOL_DIR / "pools.json").write_text(json.dumps(pools, indent=2),
                                         encoding="utf-8")
    print(f"Contact sheets + pools.json in {POOL_DIR}")


if __name__ == "__main__":
    main()
