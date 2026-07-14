"""Stage 3 — attribute-match rerank. This is where precision is won.

Pure scoring logic: candidates (with their structured records) come in as
plain data, ranked results go out. No DB, no model, no I/O — fully unit-testable.

The core idea — the *binding score*: a parsed clause like (tie, red) earns
credit only from an image garment that matches BOTH the category and the
colour on the SAME detection. An image with a blue tie + red shirt gets almost
nothing for "red tie", even though CLIP's bag-of-concepts cosine can't tell
the difference. That asymmetry is exactly the compositionality fix.

Scoring per clause:
  cat_score   : 1.0 exact category, partial for near-substitutes (coat~jacket)
  color_score : 1.0 on exact name match, else a continuous LAB delta-E ramp
                against the query word's own LAB target (any word the colour
                lexicon can ground — 'maroon', 'teal', 'burgundy' — not just
                the 14 palette names). 'navy' earns ~0.6 of a 'blue' query;
                'white' bottoms out at the `category_only_credit` floor
                (garment seen, colour contradicts — strong counter-evidence,
                deliberately below `absent_credit`).
  clause      = max over image garments of (cat_score * color_score),
                or absent_credit when no category-compatible garment exists

Formality/environment use the stored zero-shot probability of the queried
label (soft), so "office at p=0.45 vs street p=0.40" is not treated the same
as a confident mismatch. Axes the query doesn't mention are excluded from the
weight normalisation — unspecified never penalises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, resolve_color_lab
from retriever.query_parser import GarmentClause, StructuredQuery

# Near-substitutable categories: (query item -> {image category: credit}).
CATEGORY_COMPAT = {
    "shirt": {"t-shirt": 0.7, "blouse": 1.0},
    "t-shirt": {"shirt": 0.7, "sweater": 0.5},
    "coat": {"jacket": 0.8, "cape": 0.5},
    "jacket": {"coat": 0.8, "cardigan": 0.5, "vest": 0.4},
    "sweater": {"cardigan": 0.7, "t-shirt": 0.4},
    "cardigan": {"sweater": 0.7, "jacket": 0.4},
    "vest": {"jacket": 0.5},
    "cape": {"coat": 0.6, "jacket": 0.4},
    "pants": {"tights": 0.3, "shorts": 0.3},
    "shorts": {"pants": 0.4},
    "skirt": {"dress": 0.5, "shorts": 0.3},
    "dress": {"jumpsuit": 0.5, "skirt": 0.4},
    "jumpsuit": {"dress": 0.5},
    "tights": {"socks": 0.3, "pants": 0.3},
    "shoes": {"socks": 0.2},
}


def _category_score(query_item: str, image_category: str) -> float:
    if query_item == image_category:
        return 1.0
    return CATEGORY_COMPAT.get(query_item, {}).get(image_category, 0.0)


def _color_score(clause: GarmentClause, garment_lab: list[float],
                 garment_color: str, soft_max: float, floor: float,
                 gate: dict | None = None) -> float:
    if clause.color is None:
        return 1.0  # colour unspecified -> category alone decides
    targets = clause.color_targets()
    if not targets:
        return 1.0  # colour word we cannot ground -> don't fabricate a penalty

    # A clause may accept several colours ("a red and white shirt"): take the
    # BEST match, never demand both — the detector reports one dominant colour
    # per garment, so requiring both would make the garment unmatchable.
    best = 0.0
    for name, ref in targets:
        if name == garment_color:
            return 1.0  # user's word == the name the indexer gave this garment
        # Achromatic gate: a near-neutral garment (black/white/gray fabric,
        # chroma below the gate) is evidence of NO colour — it earns only the
        # floor toward a chromatic query, never partial soft credit.
        if gate:
            q_chroma = float(np.hypot(ref[1], ref[2]))
            g_chroma = float(np.hypot(garment_lab[1], garment_lab[2]))
            if (q_chroma > gate["query_chroma_min"]
                    and g_chroma < gate["garment_chroma_max"]):
                best = max(best, floor)
                continue
        # Perceptual soft match: measured garment LAB vs the query's LAB
        # target. Continuous, so "maroon" sits near red fabrics without
        # equalling them, and "navy" earns most of a "blue" query's credit.
        de = float(np.linalg.norm(np.asarray(garment_lab) - np.asarray(ref)))
        best = max(best, 1.0 - de / soft_max)
    return max(best, floor)


def binding_score(query: StructuredQuery, garments: list[dict],
                  soft_max: float, floor: float, absent_credit: float,
                  gate: dict | None = None) -> float:
    """Mean over parsed clauses of the best same-garment (category AND colour)
    match. Each image garment may satisfy at most one clause (greedy) so a
    single white shirt can't pay for both 'white shirt' and 'white pants'.

    Evidence hierarchy per clause:
      correct binding (1.0) >> category absent (absent_credit — detector
      misses are common, absence is weak evidence) > confirmed wrong colour
      (cat * floor — the garment was seen and its colour contradicts)."""
    if not query.garments:
        return 0.0
    available = list(garments)
    clause_scores = []
    for clause in query.garments:
        best_i, best_s = -1, -1.0
        for i, g in enumerate(available):
            cat = _category_score(clause.item, g["category"])
            if cat == 0.0:
                continue
            s = cat * _color_score(clause, g["color_lab"], g["color"],
                                   soft_max, floor, gate)
            if s > best_s:
                best_i, best_s = i, s
        if best_i >= 0:
            available.pop(best_i)
            clause_scores.append(best_s)
        else:
            clause_scores.append(absent_credit)
    return float(np.mean(clause_scores))


def _soft_label_score(query_label: str | None, query_dist: dict | None,
                      argmax_label: str, scores: dict | None) -> float | None:
    if query_label is None:
        return None  # axis unspecified -> excluded from normalisation
    if query_dist and scores:
        # Agreement between the query's and the image's label distributions.
        # Collapses to scores[query_label] when the query dist is one-hot.
        return float(sum(p * scores.get(label, 0.0)
                         for label, p in query_dist.items()))
    if scores:
        return float(scores.get(query_label, 0.0))
    return 1.0 if query_label == argmax_label else 0.0


def rerank(query: StructuredQuery, candidates: list[dict], k: int = 5,
           cfg: dict | None = None) -> list[dict]:
    """candidates: [{image_id, cosine, record}] -> top-k with score breakdown."""
    cfg = cfg or load_config()
    rc, sc = cfg["rerank"], cfg["search"]

    results = []
    for cand in candidates:
        rec = cand["record"]
        parts, weights = [], []

        if query.garments:
            parts.append(binding_score(query, rec.get("garments", []),
                                       rc["color_soft_max"],
                                       rc["category_only_credit"],
                                       rc["absent_credit"],
                                       rc.get("achromatic_gate")))
            weights.append(rc["w_binding"])
        f = _soft_label_score(query.formality, getattr(query, "formality_dist", None),
                              rec.get("formality", ""), rec.get("formality_scores"))
        if f is not None:
            parts.append(f)
            weights.append(rc["w_formality"])
        e = _soft_label_score(query.environment, getattr(query, "environment_dist", None),
                              rec.get("environment", ""), rec.get("environment_scores"))
        if e is not None:
            parts.append(e)
            weights.append(rc["w_environment"])

        attr = float(np.dot(parts, weights) / np.sum(weights)) if weights else 0.0
        final = sc["alpha"] * cand["cosine"] + sc["beta"] * attr
        results.append({**cand, "attr_score": round(attr, 4),
                        "final_score": round(final, 4)})

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:k]
