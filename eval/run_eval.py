"""Run the 5 assignment queries, save top-k image grids, report P@k / R@k.

Relevance labels live in eval/relevance.json — a hand-labelled mapping
{query: [relevant image_ids]}. Grids go to eval/results/.

Run:  python eval/run_eval.py
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

from eval.metrics import precision_at_k, recall_at_k
from retriever.search import Searcher

EVAL_QUERIES = {
    "attribute": "A person in a bright yellow raincoat.",
    "contextual": "Professional business attire inside a modern office.",
    "complex": "Someone wearing a blue shirt sitting on a park bench.",
    "style": "Casual weekend outfit for a city walk.",
    "compositional": "A red tie and a white shirt in a formal setting.",
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RELEVANCE_FILE = Path(__file__).resolve().parent / "relevance.json"


def save_grid(results: list[dict], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 4.2))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ax.imshow(Image.open(r["record"]["path"]).convert("RGB"))
        ax.set_title(f"{r['image_id'][:8]}\nfinal={r['final_score']:.2f} "
                     f"cos={r['cosine']:.2f} attr={r['attr_score']:.2f}",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(k: int = 5) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    relevance = (json.loads(RELEVANCE_FILE.read_text(encoding="utf-8"))
                 if RELEVANCE_FILE.exists() else {})

    searcher = Searcher()
    rows = []
    for name, query in EVAL_QUERIES.items():
        results = searcher.search(query, k=k)
        save_grid(results, f"[{name}] {query}", RESULTS_DIR / f"{name}.png")

        retrieved = [r["image_id"] for r in results]
        line = f"{name:14s} | {query}"
        if name in relevance:
            rel = set(relevance[name])
            p = precision_at_k(retrieved, rel, k)
            rc = recall_at_k(retrieved, rel, k)
            line += f"  ->  P@{k}={p:.2f}  R@{k}={rc:.2f}"
            rows.append((name, p, rc))
        print(line)
        for r in results:
            print(f"    {r['image_id']}  final={r['final_score']:.3f}  "
                  f"{r['record']['caption']}")

    if rows:
        mp = sum(r[1] for r in rows) / len(rows)
        mr = sum(r[2] for r in rows) / len(rows)
        print(f"\nMean P@{k}={mp:.2f}  Mean R@{k}={mr:.2f}  "
              f"({len(rows)} labelled queries)")
    print(f"\nGrids saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
