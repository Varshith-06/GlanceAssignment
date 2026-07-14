"""Binding extraction across English constructions — the suite the 5 eval
queries cannot see (they are all pre-nominal, so they measure none of this).

parse() = LLM -> dependency parse -> adjacency rule. These tests exercise the
offline path (dependency parse), which is what gets graded.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retriever.query_parser import parse_deps, parse_rules

spacy_missing = parse_deps("a red shirt") is None
requires_spacy = pytest.mark.skipif(
    spacy_missing, reason="spaCy en_core_web_sm not installed")


def bindings(q):
    """{(item, frozenset_of_acceptable_colours)} from the dependency parse."""
    sq = parse_deps(q)
    return {(g.item, frozenset([g.color] + g.alt_colors) if g.color
             else frozenset()) for g in sq.garments}


def one(item, *colors):
    return (item, frozenset(colors))


# --- pre-nominal: the case adjacency already handled -------------------------

@requires_spacy
@pytest.mark.parametrize("query,expected", [
    ("a red shirt", {one("shirt", "red")}),
    ("a red tie and a white shirt", {one("tie", "red"), one("shirt", "white")}),
    ("a bright yellow raincoat", {one("coat", "yellow")}),
    # The colour must not leak across an intervening noun phrase.
    ("a red bag next to a white shirt", {one("bag", "red"), one("shirt", "white")}),
    ("a red tie and a white shirt in a formal setting",
     {one("tie", "red"), one("shirt", "white")}),
])
def test_prenominal(query, expected):
    assert bindings(query) == expected


# --- post-nominal: adjacency DROPPED the colour entirely ----------------------

@requires_spacy
@pytest.mark.parametrize("query,expected", [
    ("a shirt that's red", {one("shirt", "red")}),
    ("a shirt that is red", {one("shirt", "red")}),
    ("a tie which is red", {one("tie", "red")}),
    ("the shirt is white", {one("shirt", "white")}),
    ("a shirt in red", {one("shirt", "red")}),
    ("a coat in bright yellow", {one("coat", "yellow")}),
    ("pants in navy", {one("pants", "navy")}),
    ("a jacket coloured black", {one("jacket", "black")}),
    ("a shirt, red, with black pants", {one("shirt", "red"), one("pants", "black")}),
])
def test_postnominal_colour_is_bound_not_dropped(query, expected):
    assert bindings(query) == expected


# --- coordination: adjacency OVERWROTE or dropped the colour -----------------

@requires_spacy
def test_two_colours_one_garment():
    # One shirt, either colour acceptable — NOT two shirt clauses (which would
    # demand two separate shirt detections in the image).
    assert bindings("a red and white shirt") == {one("shirt", "red", "white")}
    assert bindings("a black and yellow jacket") == {one("jacket", "black", "yellow")}


@requires_spacy
def test_one_colour_distributes_over_coordinated_garments():
    assert bindings("a red shirt and tie") == {one("shirt", "red"), one("tie", "red")}
    assert bindings("black shirt and pants") == {one("shirt", "black"),
                                                 one("pants", "black")}


@requires_spacy
def test_distribution_does_not_override_an_explicit_colour():
    # "white shirt" must keep white — it must NOT inherit red from the tie.
    assert bindings("a red tie and a white shirt") == {one("tie", "red"),
                                                       one("shirt", "white")}


# --- the swap test must still hold at the syntax tier -------------------------

@requires_spacy
def test_swap_still_distinguished():
    assert bindings("red shirt with blue pants") == {one("shirt", "red"),
                                                     one("pants", "blue")}
    assert bindings("blue shirt with red pants") == {one("shirt", "blue"),
                                                     one("pants", "red")}
    assert bindings("red shirt with blue pants") != bindings("blue shirt with red pants")


# --- the adjacency fallback keeps working when spaCy is absent ---------------

def test_adjacency_fallback_still_handles_the_common_case():
    got = {(g.item, g.color) for g in parse_rules("a red tie and a white shirt").garments}
    assert got == {("tie", "red"), ("shirt", "white")}
