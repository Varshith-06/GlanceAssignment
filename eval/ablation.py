"""Ablation on the compositional query — the money shot.

Three systems, identical corpus:
  A. Vanilla CLIP (openai/clip-vit-base-patch32), dense-only
  B. FashionCLIP, dense-only
  C. Full hybrid (FashionCLIP dense recall + structured attribute rerank)

A vs B isolates the fine-grained-fashion gain; B vs C isolates the
compositional-binding gain. P@k is computed against the same hand-labelled
relevance set used by run_eval. Vanilla CLIP embeddings are computed once and
cached (data/vanilla_clip_embeddings.npz).

Run:  python eval/ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config
from eval.metrics import precision_at_k, recall_at_k
from eval.run_eval import EVAL_QUERIES, RELEVANCE_FILE, RESULTS_DIR, save_grid
from indexer.embed import ClipEmbedder
from retriever.search import Searcher

QUERY_NAME = "compositional"
K = 10
SWAP_RANKS: dict[str, dict[str, str]] = {}  # filled by main() for reporting

# Swap-pair experiment. The assignment's compositional query cannot fully
# separate the systems on this corpus (it contains no red necktie at all — see
# eval/relevance_criteria.md), so we add a binding probe built from colour
# combinations that DO exist, with a strong natural asymmetry: the corpus has
# ~23 white-top+black-pants images but only ~3 black-top+white-pants ones.
# Querying the RARE direction floods a bag-of-concepts retriever with swapped
# matches (same words, wrong binding); only binding-aware scoring can demote
# them. The common direction is the control — everyone should do well.
SWAP_QUERIES = {
    "swap_rare": "a person wearing a black shirt and white pants",
    "swap_common": "a person wearing a white shirt and black pants",
}


def load_records() -> list[dict]:
    cfg = load_config()
    with open(cfg["paths"]["metadata_file"], "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def vanilla_clip_rank(query: str, records: list[dict], k: int) -> list[dict]:
    """Dense-only ranking with vanilla CLIP; embeddings cached to disk."""
    cfg = load_config()
    cache = Path(cfg["paths"]["vanilla_cache"])
    ids = [r["image_id"] for r in records]

    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        cached_ids, embs = list(data["ids"]), data["embs"]
    else:
        cached_ids, embs = [], np.empty((0, 512), np.float32)
    missing = [r for r in records if r["image_id"] not in set(cached_ids)]

    embedder = ClipEmbedder(cfg["models"]["vanilla_clip"])
    if missing:
        new = embedder.embed_images([r["path"] for r in missing],
                                    cfg["index"]["batch_size"])
        cached_ids += [r["image_id"] for r in missing]
        embs = np.concatenate([embs, new]) if embs.size else new
        np.savez(cache, ids=np.array(cached_ids), embs=embs)

    order = {img_id: i for i, img_id in enumerate(cached_ids)}
    mat = embs[[order[i] for i in ids]]
    sims = mat @ embedder.embed_text([query])[0]
    by_id = {r["image_id"]: r for r in records}
    top = np.argsort(-sims)[:k]
    return [{"image_id": ids[i], "cosine": float(sims[i]),
             "attr_score": 0.0, "final_score": float(sims[i]),
             "record": by_id[ids[i]]} for i in top]


def rank_all(searcher: Searcher, records: list[dict], query: str) -> dict[str, list[dict]]:
    return {
        "vanilla_clip_dense": vanilla_clip_rank(query, records, K),
        "fashionclip_dense": [{**c, "attr_score": 0.0,
                               "final_score": c["cosine"]}
                              for c in searcher.dense.search(query, K)],
        "hybrid_full": searcher.search(query, k=K),
    }


def main() -> None:
    records = load_records()
    relevance = (json.loads(RELEVANCE_FILE.read_text(encoding="utf-8"))
                 if RELEVANCE_FILE.exists() else {})
    searcher = Searcher()  # loads FashionCLIP once; reused by systems B and C
    RESULTS_DIR.mkdir(exist_ok=True)

    # Per-query P@5 for all three systems (the compositional row is the
    # headline; the others show the reranker doesn't sacrifice easy queries).
    per_query: dict[str, dict[str, float]] = {}
    compositional_table = []
    for qname, query in (EVAL_QUERIES | SWAP_QUERIES).items():
        relevant = set(relevance.get(qname, []))
        systems = rank_all(searcher, records, query)
        per_query[qname] = {}
        swap_ranks: dict[str, dict[str, str]] = {}
        for sname, results in systems.items():
            ids = [r["image_id"] for r in results]
            per_query[qname][sname] = precision_at_k(ids, relevant, 5)
            if qname in SWAP_QUERIES:
                swap_ranks.setdefault(qname, {})[sname] = ", ".join(
                    f"#{ids.index(r) + 1}" if r in ids else f">{K}"
                    for r in sorted(relevant))
            if qname == QUERY_NAME:
                compositional_table.append({
                    "system": sname,
                    "p@5": precision_at_k(ids, relevant, 5),
                    "p@10": precision_at_k(ids, relevant, 10),
                    "r@10": recall_at_k(ids, relevant, K),
                })
                save_grid(results[:5], f"[{sname}] {query}",
                          RESULTS_DIR / f"ablation_{sname}.png")
        if qname in SWAP_QUERIES:
            SWAP_RANKS[qname] = swap_ranks[qname]

    names = ["vanilla_clip_dense", "fashionclip_dense", "hybrid_full"]
    print("P@5 per query:")
    print(f"{'query':16s}" + "".join(f"{n:>20s}" for n in names))
    for qname in EVAL_QUERIES:
        print(f"{qname:16s}" + "".join(f"{per_query[qname][n]:20.2f}" for n in names))
    means = {n: sum(per_query[q][n] for q in EVAL_QUERIES) / len(EVAL_QUERIES)
             for n in names}
    print(f"{'MEAN (5 queries)':16s}" + "".join(f"{means[n]:20.2f}" for n in names))
    print("\nSwap-pair binding probe (P@5, and rank of each labelled match):")
    for qname in SWAP_QUERIES:
        print(f"{qname:16s}" + "".join(f"{per_query[qname][n]:20.2f}" for n in names))
        print(f"{'  match ranks':16s}" + "".join(
            f"{SWAP_RANKS[qname][n]:>20s}" for n in names))

    print(f'\nCompositional query detail: "{EVAL_QUERIES[QUERY_NAME]}"')
    print(f"{'system':22s} {'P@5':>6s} {'P@10':>6s} {'R@10':>6s}")
    for row in compositional_table:
        print(f"{row['system']:22s} {row['p@5']:6.2f} {row['p@10']:6.2f} "
              f"{row['r@10']:6.2f}")

    out = RESULTS_DIR / "ablation_table.json"
    out.write_text(json.dumps({"per_query_p5": per_query, "mean_p5": means,
                               "compositional": compositional_table,
                               "swap_match_ranks": SWAP_RANKS}, indent=2),
                   encoding="utf-8")
    print(f"\nGrids + table saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
