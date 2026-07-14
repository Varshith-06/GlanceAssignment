"""The swap test — the assignment's explicit unit test.

"red shirt with blue pants" and "blue shirt with red pants" must:
  1. parse to DIFFERENT colour->garment bindings, and
  2. rank a matching image differently under the reranker.
Runs fully offline (rule parser + pure rerank), no index or model needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import palette_lab
from retriever.query_parser import fuse_soft_axis, parse_rules
from retriever.rerank import rerank


def _lab(name):
    return [float(v) for v in palette_lab()[name]]


def _image(image_id, shirt_color, pants_color):
    """Synthetic indexed record with a known shirt/pants colouring."""
    return {
        "image_id": image_id,
        "cosine": 0.30,  # identical cosine for all -> only binding can separate
        "record": {
            "garments": [
                {"category": "shirt", "color": shirt_color, "color_lab": _lab(shirt_color)},
                {"category": "pants", "color": pants_color, "color_lab": _lab(pants_color)},
            ],
            "formality": "casual",
            "environment": "street",
        },
    }


def test_swap_queries_parse_to_different_bindings():
    q1 = parse_rules("red shirt with blue pants")
    q2 = parse_rules("blue shirt with red pants")
    b1 = {(g.item, g.color) for g in q1.garments}
    b2 = {(g.item, g.color) for g in q2.garments}
    assert b1 == {("shirt", "red"), ("pants", "blue")}
    assert b2 == {("shirt", "blue"), ("pants", "red")}
    assert b1 != b2


def test_swap_images_rank_differently():
    red_shirt_blue_pants = _image("correct", "red", "blue")
    blue_shirt_red_pants = _image("swapped", "blue", "red")
    candidates = [blue_shirt_red_pants, red_shirt_blue_pants]

    ranked = rerank(parse_rules("red shirt with blue pants"), candidates, k=2)
    assert ranked[0]["image_id"] == "correct"
    assert ranked[0]["final_score"] > ranked[1]["final_score"]

    # And the mirror query prefers the mirror image.
    ranked = rerank(parse_rules("blue shirt with red pants"), candidates, k=2)
    assert ranked[0]["image_id"] == "swapped"


def test_one_garment_cannot_satisfy_two_clauses():
    # A single white shirt must not pay for both 'white shirt' and 'white pants'.
    img = {
        "image_id": "one_shirt",
        "cosine": 0.30,
        "record": {"garments": [
            {"category": "shirt", "color": "white", "color_lab": _lab("white")}],
            "formality": "casual", "environment": "home"},
    }
    both = {
        "image_id": "both",
        "cosine": 0.30,
        "record": {"garments": [
            {"category": "shirt", "color": "white", "color_lab": _lab("white")},
            {"category": "pants", "color": "white", "color_lab": _lab("white")}],
            "formality": "casual", "environment": "home"},
    }
    ranked = rerank(parse_rules("white shirt and white pants"), [img, both], k=2)
    assert ranked[0]["image_id"] == "both"


def test_color_does_not_leak_across_conjunction():
    q = parse_rules("a red tie and a white shirt in a formal setting")
    bindings = {(g.item, g.color) for g in q.garments}
    assert bindings == {("tie", "red"), ("shirt", "white")}
    assert q.formality == "formal"


def test_near_miss_color_scores_between_wrong_and_exact():
    # 'blue shirt' vs a NAVY shirt: partial credit — far above a white shirt,
    # below a true blue one. This is the LAB soft match actually firing.
    navy = _image("navy_shirt", "navy", "black")
    blue = _image("blue_shirt", "blue", "black")
    white = _image("white_shirt", "white", "black")
    ranked = rerank(parse_rules("a blue shirt"), [white, navy, blue], k=3)
    assert [r["image_id"] for r in ranked] == ["blue_shirt", "navy_shirt", "white_shirt"]
    scores = {r["image_id"]: r["attr_score"] for r in ranked}
    assert scores["navy_shirt"] > scores["white_shirt"] + 0.2  # meaningfully apart


def test_out_of_palette_color_word_is_grounded_not_dropped():
    # 'maroon' is not a palette name; it must resolve to a LAB target near
    # red — NOT parse to color=None, which would grant any colour full credit.
    q = parse_rules("a maroon shirt")
    assert q.garments[0].color == "maroon"
    assert q.garments[0].color_lab is not None

    red = _image("red_shirt", "red", "black")
    black = _image("black_shirt", "black", "white")
    ranked = rerank(q, [black, red], k=2)
    assert ranked[0]["image_id"] == "red_shirt"
    assert ranked[0]["attr_score"] > ranked[1]["attr_score"] + 0.2


def test_expanded_vocabulary_and_compounds():
    # Compound: "dress shirt" is one shirt, not a dress + a shirt.
    q = parse_rules("a white dress shirt and navy chinos")
    assert {(g.item, g.color) for g in q.garments} == {("shirt", "white"),
                                                       ("pants", "navy")}
    # Adjacent same-category nouns merge: "flannel shirt" = one shirt,
    # and the colour still binds through the merge.
    q = parse_rules("a red flannel shirt")
    assert [(g.item, g.color) for g in q.garments] == [("shirt", "red")]
    # tank + top both map to t-shirt -> one clause, not two.
    q = parse_rules("a black tank top and sneakers")
    assert [(g.item, g.color) for g in q.garments] == [("t-shirt", "black"),
                                                       ("shoes", None)]
    # A sweep of new synonyms lands on canonical categories.
    q = parse_rules("beige trench, loafers, a fedora and a tote")
    assert [(g.item, g.color) for g in q.garments] == [
        ("coat", "beige"), ("shoes", None), ("hat", None), ("bag", None)]
    # "suit" acts as garment (jacket) AND formality signal.
    q = parse_rules("a navy suit")
    assert [(g.item, g.color) for g in q.garments] == [("jacket", "navy")]
    assert q.formality == "formal"


def test_soft_axis_trust_ladder():
    drop, trust = 0.60, 0.90

    # Band 1: embedding unsure -> keyword result alone (axis drops if none).
    unsure = {"office": 0.4, "street": 0.3, "park": 0.2, "home": 0.1}
    assert fuse_soft_axis(None, unsure, drop, trust) == (None, None)
    assert fuse_soft_axis("park", unsure, drop, trust) == ("park", None)
    # ... and the "drop" variant discards the axis even over a keyword hit.
    assert fuse_soft_axis("park", unsure, drop, trust, below_drop="drop") == (None, None)

    # Band 2: embedding plausible but below trust -> keyword wins conflicts,
    # soft distribution used when they agree (or no keyword fired).
    plausible = {"office": 0.7, "street": 0.2, "park": 0.05, "home": 0.05}
    assert fuse_soft_axis("street", plausible, drop, trust) == ("street", None)
    label, dist = fuse_soft_axis("office", plausible, drop, trust)
    assert label == "office" and dist == plausible
    label, dist = fuse_soft_axis(None, plausible, drop, trust)
    assert label == "office" and dist == plausible

    # Band 3: embedding confident -> it wins even over a keyword hit.
    confident = {"office": 0.95, "street": 0.03, "park": 0.01, "home": 0.01}
    label, dist = fuse_soft_axis("street", confident, drop, trust)
    assert label == "office" and dist == confident


def test_soft_dist_scoring_collapses_to_label_lookup():
    # With a one-hot query dist, agreement scoring == the plain label lookup,
    # so soft labels change nothing when the query is unambiguous.
    img = _image("a", "red", "blue")
    img["record"]["environment_scores"] = {"office": 0.0, "street": 0.1,
                                           "park": 0.9, "home": 0.0}
    q_hard = parse_rules("red shirt in the park")
    ranked_hard = rerank(q_hard, [img], k=1)

    q_soft = parse_rules("red shirt in the park")
    q_soft.environment_dist = {"office": 0.0, "street": 0.0, "park": 1.0, "home": 0.0}
    ranked_soft = rerank(q_soft, [img], k=1)
    assert abs(ranked_hard[0]["final_score"] - ranked_soft[0]["final_score"]) < 1e-9


def test_achromatic_gate():
    # A black shirt must earn only the floor toward "blue" — colourless
    # fabric is not weak evidence of blue. Navy (real chroma) keeps its
    # soft credit, so it must rank strictly above black, which ties the
    # floor with any other wrong colour.
    navy = _image("navy", "navy", "black")
    black = _image("black", "black", "white")
    q = parse_rules("a blue shirt")
    ranked = rerank(q, [black, navy], k=2)
    assert ranked[0]["image_id"] == "navy"
    assert ranked[0]["attr_score"] > ranked[1]["attr_score"] + 0.25
    # Achromatic QUERIES bypass the gate: "white shirt" still scores exactly
    # 1.0 against a white garment and near-floor against black.
    q_white = parse_rules("a white shirt")
    white = _image("white", "white", "black")
    ranked = rerank(q_white, [black, white], k=2)
    assert ranked[0]["image_id"] == "white"


def test_unspecified_axes_do_not_penalize():
    # Query mentions only a garment; two images differing only in environment
    # must tie — environment is excluded from normalisation, not scored as 0.
    a = _image("park_img", "red", "blue")
    a["record"]["environment"] = "park"
    b = _image("home_img", "red", "blue")
    b["record"]["environment"] = "home"
    ranked = rerank(parse_rules("red shirt"), [a, b], k=2)
    assert abs(ranked[0]["final_score"] - ranked[1]["final_score"]) < 1e-9
