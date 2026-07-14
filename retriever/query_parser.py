"""Query -> StructuredQuery. This is where compositionality is won.

CLIP smears "red tie and white shirt" into one vector where 'red' is not bound
to 'tie'. We instead lift the binding into explicit structure *before* scoring:
each colour is attached to a specific garment noun, and the reranker only
rewards images where that exact (category, colour) pair co-occurs on one
detected garment.

Three parsers, tried in order, each a strict fallback for the next:
  * LLM (optional, needs ANTHROPIC_API_KEY): strict-JSON extraction.
  * Dependency parse (spaCy, offline): reads the binding off the SYNTAX tree,
    so it handles constructions adjacency cannot — post-nominal colour ("a
    shirt that's red", "a shirt in red"), copulas ("the shirt is white"),
    colour coordination ("a red and white shirt" = one shirt, two acceptable
    colours) and modifier distribution ("a red shirt and tie" = both red).
  * Adjacency rule (always available, zero dependencies): "a colour binds to
    the nearest FOLLOWING garment noun" — the English pre-nominal adjective
    rule. Correct on the common case and on the swap test, but silently drops
    the colour on every construction listed above; it is the last resort.

Why not use CLIP's own attention to recover the binding? Measured: its text
encoder is causal (a colour token cannot attend forward to its noun at all)
and its backward attention is *positional*, not lexical — swapping the colours
in "a shirt in red and pants in blue" moves the weights by <0.003. It would
reproduce the adjacency heuristic, including its failures. Syntax is the right
tool for syntax; that is what the dependency tier is for.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, resolve_color_lab


@dataclass
class GarmentClause:
    item: str                 # canonical category (matches indexer vocabulary)
    color: str | None = None  # primary colour word as the user said it
    # Continuous LAB target for the colour word. This is what the reranker
    # scores against, so "maroon" is a point near red — not red itself, and
    # never a dropped constraint. Filled automatically from `color` if omitted.
    color_lab: list[float] | None = None
    # Additional ACCEPTABLE colours for the same garment, from coordination:
    # "a red and white shirt" is ONE shirt that may be measured as either.
    # The reranker takes the best match, never demands both — the detector
    # only reports one dominant colour per garment, so demanding both would
    # make a multicoloured garment unmatchable.
    alt_colors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.color is not None and self.color_lab is None:
            lab = resolve_color_lab(self.color)
            self.color_lab = [float(v) for v in lab] if lab is not None else None

    def color_targets(self) -> list[tuple[str, list[float]]]:
        """Every (name, LAB) the garment may match. Empty if the colour word
        exists but cannot be grounded — callers treat that as unconstrained."""
        out = []
        for name in ([self.color] if self.color else []) + list(self.alt_colors):
            lab = resolve_color_lab(name)
            if lab is not None:
                out.append((name, [float(v) for v in lab]))
        return out


@dataclass
class StructuredQuery:
    garments: list[GarmentClause] = field(default_factory=list)
    formality: str | None = None    # formal | casual | outerwear | None
    environment: str | None = None  # office | street | park | home | None
    # Query-side zero-shot distributions (filled by fuse_soft_axis when the
    # embedder is available). When set, the reranker scores the axis as
    # agreement between this distribution and the image's stored one instead
    # of looking up a single label.
    formality_dist: dict[str, float] | None = None
    environment_dist: dict[str, float] | None = None
    raw: str = ""


def fuse_soft_axis(keyword_label: str | None, dist: dict[str, float],
                   drop: float, trust: float,
                   below_drop: str = "keyword") -> tuple[str | None, dict[str, float] | None]:
    """Three-band fusion of the keyword parser and the embedding classifier.

    conf = max(dist):
      conf < drop           -> embedding too unsure to assert anything:
                               below_drop="keyword" falls back to the keyword
                               result alone (axis drops when keywords found
                               nothing); below_drop="drop" drops the axis
                               unconditionally, treating embedding doubt as a
                               veto over the keyword hit.
      drop <= conf < trust  -> embedding is plausible: on a CONFLICT the
                               explicit keyword wins (it is literal evidence);
                               otherwise use the soft distribution.
      conf >= trust         -> embedding wins outright — this is what corrects
                               keyword false-positives like "smart casual"
                               matching the 'formal' keyword "smart".

    Returns (label, dist_or_None); dist None means score by label lookup."""
    argmax = max(dist, key=dist.get)
    conf = dist[argmax]
    if conf < drop:
        return (None, None) if below_drop == "drop" else (keyword_label, None)
    if conf < trust and keyword_label is not None and keyword_label != argmax:
        return keyword_label, None
    return argmax, dict(dist)


# ---------------------------------------------------------------- lexicons --
# A colour token is any word that resolves to a LAB target: the 14 config
# palette names, the extended CSS/xkcd lexicon in common.py (maroon, teal,
# burgundy, ...), plus spelling variants. The word itself is kept as the
# clause colour; scoring is continuous in LAB, so no lossy mapping to the
# nearest palette name happens on the query side.
SPELLING_VARIANTS = {"grey": "gray"}


def _color_word(token: str) -> str | None:
    token = SPELLING_VARIANTS.get(token, token)
    return token if resolve_color_lab(token) is not None else None


# Intensity modifiers carry no binding information — strip them.
COLOR_MODIFIERS = {"bright", "dark", "light", "deep", "pale", "vivid", "neon"}

# Surface garment nouns -> canonical category used by the indexer. Broad on
# purpose: an unrecognised garment word silently weakens the query to
# dense-only, so coverage is cheap insurance. Grouped by canonical category
# (the 24 categories the detector actually emits on this corpus).
GARMENT_SYNONYMS = {
    # -- shirt (woven / collared tops) --
    "shirt": "shirt", "blouse": "shirt", "dress-shirt": "shirt",
    "button-down": "shirt", "button-up": "shirt", "oxford": "shirt",
    "polo": "shirt", "henley": "shirt", "flannel": "shirt", "tunic": "shirt",
    # -- t-shirt (knit / casual tops) --
    "t-shirt": "t-shirt", "tshirt": "t-shirt", "tee": "t-shirt", "top": "t-shirt",
    "sweatshirt": "t-shirt", "hoodie": "t-shirt", "tank": "t-shirt",
    "tank-top": "t-shirt", "camisole": "t-shirt", "cami": "t-shirt",
    "crop-top": "t-shirt", "jersey": "t-shirt", "singlet": "t-shirt",
    # -- sweater --
    "sweater": "sweater", "pullover": "sweater", "jumper": "sweater",
    "turtleneck": "sweater", "knit": "sweater", "knitwear": "sweater",
    "fleece": "sweater",
    # -- cardigan --
    "cardigan": "cardigan", "cardi": "cardigan",
    # -- jacket --
    "jacket": "jacket", "blazer": "jacket", "windbreaker": "jacket",
    "bomber": "jacket", "sportcoat": "jacket", "sportscoat": "jacket",
    "suit": "jacket", "tuxedo": "jacket", "tux": "jacket",
    # -- coat --
    "coat": "coat", "raincoat": "coat", "overcoat": "coat", "trenchcoat": "coat",
    "trench": "coat", "parka": "coat", "anorak": "coat", "peacoat": "coat",
    "puffer": "coat", "duffle": "coat", "duster": "coat", "mackintosh": "coat",
    "slicker": "coat", "windcheater": "coat",
    # -- vest --
    "vest": "vest", "waistcoat": "vest", "gilet": "vest", "bodywarmer": "vest",
    # -- pants --
    "pants": "pants", "trousers": "pants", "jeans": "pants", "slacks": "pants",
    "chinos": "pants", "khakis": "pants", "joggers": "pants",
    "sweatpants": "pants", "cargos": "pants", "cargo-pants": "pants",
    "corduroys": "pants", "denims": "pants", "jeggings": "pants",
    "culottes": "pants", "palazzos": "pants",
    # -- shorts --
    "shorts": "shorts", "bermudas": "shorts", "hotpants": "shorts",
    # -- skirt --
    "skirt": "skirt", "miniskirt": "skirt", "pencil-skirt": "skirt",
    "midi-skirt": "skirt", "maxi-skirt": "skirt", "sarong": "skirt",
    # -- dress --
    "dress": "dress", "gown": "dress", "frock": "dress", "sundress": "dress",
    "maxi-dress": "dress", "minidress": "dress", "kaftan": "dress",
    "caftan": "dress", "shift-dress": "dress",
    # -- jumpsuit --
    "jumpsuit": "jumpsuit", "romper": "jumpsuit", "playsuit": "jumpsuit",
    "overalls": "jumpsuit", "dungarees": "jumpsuit", "boilersuit": "jumpsuit",
    "onesie": "jumpsuit",
    # -- cape (0 detections on this corpus; kept for parser completeness) --
    "cape": "cape", "poncho": "cape", "cloak": "cape",
    # -- tie --
    "tie": "tie", "necktie": "tie", "bowtie": "tie", "bow-tie": "tie",
    "cravat": "tie", "ascot": "tie",
    # -- scarf --
    "scarf": "scarf", "shawl": "scarf", "muffler": "scarf", "bandana": "scarf",
    "pashmina": "scarf", "stole": "scarf",
    # -- hat --
    "hat": "hat", "cap": "hat", "beanie": "hat", "fedora": "hat",
    "beret": "hat", "snapback": "hat", "sunhat": "hat", "bonnet": "hat",
    "panama": "hat", "trilby": "hat", "bucket-hat": "hat", "visor": "hat",
    # -- shoes --
    "shoes": "shoes", "shoe": "shoes", "sneakers": "shoes", "trainers": "shoes",
    "boots": "shoes", "heels": "shoes", "pumps": "shoes", "loafers": "shoes",
    "sandals": "shoes", "flats": "shoes", "brogues": "shoes",
    "stilettos": "shoes", "slippers": "shoes", "moccasins": "shoes",
    "espadrilles": "shoes", "wedges": "shoes", "mules": "shoes",
    "kicks": "shoes", "footwear": "shoes", "oxfords": "shoes",
    # -- bag --
    "bag": "bag", "handbag": "bag", "purse": "bag", "backpack": "bag",
    "tote": "bag", "satchel": "bag", "clutch": "bag", "wallet": "bag",
    "duffel": "bag", "crossbody": "bag", "knapsack": "bag",
    "pocketbook": "bag", "messenger-bag": "bag",
    # -- glasses --
    "glasses": "glasses", "sunglasses": "glasses", "spectacles": "glasses",
    "shades": "glasses", "eyewear": "glasses", "sunnies": "glasses",
    # -- belt / tights / socks / gloves / umbrella / watch --
    "belt": "belt",
    "tights": "tights", "stockings": "tights", "leggings": "tights",
    "pantyhose": "tights", "hosiery": "tights",
    "socks": "socks", "sock": "socks",
    "gloves": "gloves", "glove": "gloves", "mittens": "gloves", "mitts": "gloves",
    "umbrella": "umbrella", "parasol": "umbrella", "brolly": "umbrella",
    "watch": "watch", "wristwatch": "watch", "smartwatch": "watch",
    "timepiece": "watch",
}

# Two-word garment compounds whose FIRST word is (or could be) a garment noun
# of a different category — without these, "dress shirt" would parse as a
# dress plus a shirt. Checked before single-token lookup.
GARMENT_COMPOUNDS = {
    ("dress", "shirt"): "shirt",
    ("polo", "shirt"): "shirt",
    ("tank", "top"): "t-shirt",
    ("crop", "top"): "t-shirt",
    ("sweater", "vest"): "vest",
    ("suit", "jacket"): "jacket",
    ("bomber", "jacket"): "jacket",
    ("puffer", "jacket"): "coat",
    ("trench", "coat"): "coat",
    ("rain", "jacket"): "coat",
    ("pencil", "skirt"): "skirt",
    ("cargo", "pants"): "pants",
    ("bow", "tie"): "tie",
    ("bucket", "hat"): "hat",
    ("baseball", "cap"): "hat",
    ("messenger", "bag"): "bag",
    ("shoulder", "bag"): "bag",
}
# Some canonical categories are near-substitutable at retrieval time; the
# reranker treats these as acceptable category matches (full list in rerank.py).

FORMALITY_KEYWORDS = {
    "formal": ["formal", "business", "professional", "suit", "tuxedo", "tux",
               "office attire", "smart", "elegant", "work attire"],
    "casual": ["casual", "everyday", "weekend", "relaxed", "streetwear",
               "laid-back", "comfortable"],
    "outerwear": ["raincoat", "outerwear", "winter", "rain"],
}
ENVIRONMENT_KEYWORDS = {
    "office": ["office", "workplace", "meeting", "desk", "boardroom"],
    "street": ["street", "city", "urban", "sidewalk", "downtown", "crosswalk"],
    "park": ["park", "bench", "garden", "outdoors", "nature", "picnic", "lawn"],
    "home": ["home", "indoors", "couch", "sofa", "living room", "bedroom"],
}


def _keyword_label(text: str, keyword_map: dict[str, list[str]]) -> str | None:
    """First label whose keyword appears in the text (word-boundary match)."""
    for label, words in keyword_map.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", text):
                return label
    return None


def parse_rules(query: str) -> StructuredQuery:
    """Deterministic parser. Colour binding = nearest following garment noun."""
    text = query.lower()
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", text)

    garments: list[GarmentClause] = []
    pending_color: str | None = None
    pending_dist = 0
    prev_garment_end = -1  # token index just past the last garment noun
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in COLOR_MODIFIERS:
            i += 1
            continue  # "bright yellow" -> treat as "yellow"
        color_word = _color_word(tok)
        if color_word is not None:
            pending_color = color_word
            pending_dist = 0
            i += 1
            continue
        # Garment lookup: two-token compounds first ("dress shirt" is one
        # shirt, not a dress + a shirt), then single nouns.
        if i + 1 < len(tokens) and (tok, tokens[i + 1]) in GARMENT_COMPOUNDS:
            item, consumed = GARMENT_COMPOUNDS[(tok, tokens[i + 1])], 2
        elif tok in GARMENT_SYNONYMS:
            item, consumed = GARMENT_SYNONYMS[tok], 1
        else:
            if pending_color is not None:
                pending_dist += 1
            i += 1
            continue
        # Adjacent nouns of the SAME category are one garment, not two
        # ("flannel shirt", "tank top" via synonyms) — merge instead of
        # emitting a duplicate clause that would demand a second garment.
        if garments and i == prev_garment_end and garments[-1].item == item:
            if garments[-1].color is None and pending_color and pending_dist <= 2:
                garments[-1] = GarmentClause(item=item, color=pending_color)
        else:
            # Bind the pending colour only if it is close enough to plausibly
            # modify this noun ("red tie and a white shirt": 'red' must not
            # leak past 'tie' onto 'shirt').
            color = pending_color if pending_color and pending_dist <= 2 else None
            garments.append(GarmentClause(item=item, color=color))
        pending_color = None
        prev_garment_end = i + consumed
        i += consumed

    formality = _keyword_label(text, FORMALITY_KEYWORDS)
    environment = _keyword_label(text, ENVIRONMENT_KEYWORDS)

    # Outerwear garments imply the outerwear style axis if nothing else set it.
    if formality is None and any(g.item in ("coat", "jacket") for g in garments):
        formality = "outerwear"

    return StructuredQuery(garments=garments, formality=formality,
                           environment=environment, raw=query)


# ------------------------------------------------------- dependency parse --
# Dependency relations that attach a colour to the garment it modifies. Each
# was verified against spaCy's actual output (not assumed) — see the mapping
# in the comments below.
_SPACY = None
_SPACY_TRIED = False


def _load_spacy():
    """Lazy, cached, failure-tolerant load. Missing model => tier is skipped
    and parse() falls back to the adjacency rule, so the system still runs."""
    global _SPACY, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy

            name = load_config()["models"].get("spacy", "en_core_web_sm")
            _SPACY = spacy.load(name, disable=["ner", "lemmatizer", "textcat"])
        except Exception:
            _SPACY = None
    return _SPACY


def _canon_color(token) -> str | None:
    return _color_word(token.text)


def _canon_garment(token) -> str | None:
    return GARMENT_SYNONYMS.get(token.text)


def _colors_from(token, seen: set[int]) -> list[str]:
    """Colour at `token`, plus any coordinated with it ('red and white')."""
    out = []
    c = _canon_color(token)
    if c and token.i not in seen:
        seen.add(token.i)
        out.append(c)
        for child in token.children:          # red --conj--> white
            if child.dep_ == "conj":
                out.extend(_colors_from(child, seen))
    return out


def parse_deps(query: str) -> StructuredQuery | None:
    """Read colour->garment bindings off the syntax tree. None if unavailable
    or if it finds no garment at all (caller then falls back to adjacency)."""
    nlp = _load_spacy()
    if nlp is None:
        return None
    try:
        doc = nlp(query.lower())
    except Exception:
        return None

    # Garment nouns, minus compound modifiers ("dress shirt" is one shirt, not
    # a dress and a shirt). The compound table keeps its canonical mapping.
    items: dict[int, str] = {}
    for tok in doc:
        item = _canon_garment(tok)
        if item is None:
            continue
        if tok.dep_ in ("compound", "amod") and _canon_garment(tok.head):
            continue  # modifier half of a compound — the head carries it
        head_pair = None
        for child in tok.children:
            if child.dep_ == "compound":
                head_pair = GARMENT_COMPOUNDS.get((child.text, tok.text))
        items[tok.i] = head_pair or item

    if not items:
        return None

    seen: set[int] = set()
    colors: dict[int, list[str]] = {i: [] for i in items}
    for i in items:
        tok = doc[i]
        for child in tok.children:
            # "a red shirt" / "a shirt, red, ..."  red --amod--> shirt
            # Also compound/nmod: several colour words are tagged NOUN or
            # PROPN rather than ADJ ("a navy tie" -> navy --compound--> tie),
            # so amod alone silently loses them.
            if child.dep_ in ("amod", "compound", "nmod"):
                colors[i] += _colors_from(child, seen)
            # "a shirt in red"  shirt --prep--> in --pobj--> red
            elif child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        colors[i] += _colors_from(pobj, seen)
            # "a shirt that's red"  shirt --relcl--> 's --acomp--> red
            # "a jacket coloured black"  jacket --acl--> coloured --oprd--> black
            elif child.dep_ in ("relcl", "acl"):
                for gc in child.children:
                    if gc.dep_ in ("acomp", "oprd", "attr", "dobj"):
                        colors[i] += _colors_from(gc, seen)
        # "the shirt is white"  shirt --nsubj--> is --acomp--> white
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.head.pos_ in ("AUX", "VERB"):
            for gc in tok.head.children:
                if gc.dep_ in ("acomp", "attr", "oprd"):
                    colors[i] += _colors_from(gc, seen)

    # Modifier distribution over coordinated garments: "a red shirt and tie"
    # -> the tie is red too. Only when the conjunct has no colour of its own,
    # so "a red tie and a white shirt" keeps its two distinct bindings.
    for i in items:
        tok = doc[i]
        if not colors[i] and tok.dep_ == "conj" and tok.head.i in items:
            colors[i] = list(colors[tok.head.i])

    garments = [
        GarmentClause(item=items[i],
                      color=(colors[i][0] if colors[i] else None),
                      alt_colors=colors[i][1:])
        for i in sorted(items)
    ]

    text = query.lower()
    formality = _keyword_label(text, FORMALITY_KEYWORDS)
    environment = _keyword_label(text, ENVIRONMENT_KEYWORDS)
    if formality is None and any(g.item in ("coat", "jacket") for g in garments):
        formality = "outerwear"
    return StructuredQuery(garments=garments, formality=formality,
                           environment=environment, raw=query)


def parse_llm(query: str) -> StructuredQuery | None:
    """Optional LLM parser (strict JSON). Returns None on any failure so the
    caller falls back to rules — the system must never depend on network."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        garments = ", ".join(sorted(set(GARMENT_SYNONYMS.values())))
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=300,
            messages=[{"role": "user", "content":
                f"Extract fashion search attributes as strict JSON with keys "
                f'"garments" (list of {{"item","color"}}), "formality" '
                f'(formal|casual|outerwear|null), "environment" '
                f"(office|street|park|home|null). item must be one of: {garments}. "
                f"color is the single colour word from the query (e.g. maroon, "
                f"teal, red) or null. Bind each color to the garment it "
                f"modifies. Query: {query!r}. JSON only."}])
        data = json.loads(msg.content[0].text)

        def _clause(g: dict) -> GarmentClause:
            color = g.get("color")
            # A colour the lexicon can't ground in LAB falls back to
            # unconstrained — matching the rule parser, which never treats
            # such a word as a colour in the first place.
            if color is not None and resolve_color_lab(str(color).lower()) is None:
                color = None
            return GarmentClause(item=g["item"],
                                 color=str(color).lower() if color else None)

        return StructuredQuery(
            garments=[_clause(g) for g in data.get("garments", [])],
            formality=data.get("formality"),
            environment=data.get("environment"),
            raw=query)
    except Exception:
        return None


def _arbitrate(deps: StructuredQuery, rules: StructuredQuery) -> StructuredQuery:
    """Choose between the syntax parse and the adjacency parse.

    Neither dominates, and this is the whole reason a router exists:

    * The CNN dependency parser (en_core_web_sm) assumes GRAMMATICAL input.
      Real search queries are often telegraphic fragments — "red shirt blue
      pants", "navy blazer grey trousers" — with no verb and no determiners.
      It reads those as a single compound noun chain (shirt --compound-->
      pants), collapsing the garments and re-attaching colours to the wrong
      nouns: 8/12 on such inputs, where the adjacency rule scores 9/12.
    * The adjacency rule assumes PRE-NOMINAL colour, and silently drops it on
      "a shirt that's red", "a shirt in red", "a red and white shirt" (5/19 on
      the construction suite, where the syntax tier scores 19/19).

    Routing recovers most of both: 18/19 overall, versus 15/19 for the syntax
    tier alone and 14/19 for adjacency alone. (A transformer parser —
    en_core_web_trf — needs no router at all and scores 19/19, but costs 440 MB
    and ~40 ms/query against 12 MB and ~4 ms. Set `models.spacy` in config.yaml
    to trade up; measured comparison is in the write-up.)

    Signature of a collapsed tree: it recovers FEWER garments than a plain
    lexical scan, or the same garments with FEWER colours bound. In either
    case the tree is untrustworthy and adjacency wins.
    """
    if len(deps.garments) < len(rules.garments):
        return rules
    if len(deps.garments) == len(rules.garments):
        d_colors = sum(1 for g in deps.garments if g.color)
        r_colors = sum(1 for g in rules.garments if g.color)
        if r_colors > d_colors:
            return rules
    return deps


def parse(query: str) -> StructuredQuery:
    """Public entrypoint. LLM first when available; otherwise run BOTH offline
    parsers and arbitrate — they fail on disjoint inputs (see _arbitrate)."""
    llm = parse_llm(query)
    if llm is not None:
        return llm
    rules = parse_rules(query)
    deps = parse_deps(query)
    return _arbitrate(deps, rules) if deps is not None else rules


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "a red tie and a white shirt in a formal setting"
    print(parse(q))
