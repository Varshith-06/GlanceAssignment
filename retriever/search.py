"""Public entrypoint: search(query, k) — wires Stage 1 -> 2 -> 3.

Run:  python retriever/search.py "a red tie and a white shirt in a formal setting"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from retriever.dense_search import DenseSearcher
from retriever.query_parser import fuse_soft_axis, parse
from retriever.rerank import rerank


class Searcher:
    """Holds the loaded model + DB connection so repeated queries are cheap."""

    def __init__(self):
        self.cfg = load_config()
        self.dense = DenseSearcher()

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg["search"]["top_k"]
        q_emb = self.dense.embedder.embed_text([query])[0]  # encoded once
        candidates = self.dense.search(query, self.cfg["search"]["top_n"],
                                       query_emb=q_emb)                     # Stage 1
        structured = parse(query)                                           # Stage 2
        self._soften_labels(structured, q_emb)
        return rerank(structured, candidates, k=k, cfg=self.cfg)            # Stage 3

    def _soften_labels(self, structured, q_emb) -> None:
        """Classify the query embedding against the same prompt banks used at
        index time, then fuse with the keyword parse via the trust ladder."""
        qs = self.cfg["query_soft"]
        for axis in ("formality", "environment"):
            dist = self.dense.embedder.zero_shot_scores(
                q_emb.reshape(1, -1), self.cfg[f"{axis}_prompts"],
                temperature=qs["temperature"])[0]
            label, soft = fuse_soft_axis(getattr(structured, axis), dist,
                                         qs[axis]["drop"], qs[axis]["trust"],
                                         qs.get("below_drop", "keyword"))
            setattr(structured, axis, label)
            setattr(structured, f"{axis}_dist", soft)


def search(query: str, k: int = 5) -> list[dict]:
    return Searcher().search(query, k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    structured = parse(args.query)
    print(f"Parsed: garments={[(g.item, g.color) for g in structured.garments]} "
          f"formality={structured.formality} environment={structured.environment}\n")
    for i, r in enumerate(search(args.query, args.k), 1):
        rec = r["record"]
        print(f"{i}. {r['image_id']}  final={r['final_score']:.3f} "
              f"(cos={r['cosine']:.3f}, attr={r['attr_score']:.3f})")
        print(f"   {rec['caption']}")
