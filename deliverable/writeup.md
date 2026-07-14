# Multimodal Fashion & Context Retrieval — Write-up

*Glance ML internship assignment. Code: see GitHub link in §4.*

---

## 1. Approaches considered & trade-offs

The task looks like "just use CLIP", but vanilla CLIP has two documented failure
modes that this assignment's queries deliberately probe:

1. **Compositionality blindness.** CLIP embeds an image as a *bag of concepts*:
   "red shirt with blue pants" and "blue shirt with red pants" land at nearly
   the same point because attribute→object *binding* is not preserved in a
   single pooled vector.
2. **Fine-grained fashion weakness.** Generic CLIP training data underspecifies
   fashion: blazer vs cardigan, fabric, fit and formality nuances blur together.

Every candidate architecture trades off along **recall vs precision vs zero-shot
generality**:

| # | Approach | Recall | Binding precision | Zero-shot | Verdict |
|---|---|---|---|---|---|
| a | Vanilla CLIP, dense-only | high | **poor** — bag-of-concepts | high | Baseline; fails queries 1 & 5 |
| b | FashionCLIP, dense-only | high | poor — same pooling, better features | high | Fixes fine-grained, not binding |
| c | CLIP + hard metadata filter | low — a single wrong detection/label removes a relevant image irrecoverably | high when metadata is right | medium | Brittle; detector errors are fatal |
| d | **Hybrid retrieve-then-rerank (chosen)** | high (dense stage untouched) | high (structured binding score) | high (falls back to cosine when attributes are silent) | Best of both; soft scores degrade gracefully |
| e | Fully structured / tag search (no embeddings) | very low — only what the tag vocabulary covers | high | none — "casual weekend outfit" has no tag | Cannot do style inference |

**When each wins:** (a)/(b) win when queries are single-concept ("a dress");
(c) wins when metadata is human-curated and exact filtering is a product
requirement; (e) wins in closed catalogues with clean structured data. For
open-ended natural-language queries over noisy images — this assignment — (d)
dominates because errors in either signal are *soft*: a missed detection costs
some rerank score instead of eliminating the image, and a fuzzy query still
gets full dense recall.

---

## 2. Chosen architecture

```
INDEXER (Part A)
  image ──► FashionCLIP image embedding ─────────────┐
        └─► YOLOS-Fashionpedia garment detection      ├──► ChromaDB {vector, metadata}
            + per-garment K-means colour in LAB        │    + data/metadata.jsonl
            + zero-shot formality & environment        ┘

RETRIEVER (Part B)
  query ──► Stage 1: FashionCLIP text embed → cosine top-50 (recall)
        ──► Stage 2: parse → {garments:[{item,color}], formality, environment}
        ──► Stage 3: rerank — final = α·cosine + β·attribute_match → top-k
```

**Why FashionCLIP** (`patrickjohncyh/fashion-clip`): CLIP fine-tuned on 800K
fashion image-text pairs. It directly attacks failure mode #2 (fine-grained
garments) at zero engineering cost, and keeps the zero-shot interface we need
for formality/environment prompts. It does *not* fix binding — that is the
reranker's job.

**Why each stage exists:**

- **Stage 1 — dense recall.** Cheap wide net; anything semantically close
  survives to the rerank. Zero-shot generality lives here: "casual weekend
  outfit" needs no tag or label to recall the right neighbourhood.
- **Stage 2 — structured parsing.** *This is where compositionality is solved.*
  Binding is lifted out of the smeared embedding into explicit structure:
  `[{item: tie, color: red}, {item: shirt, color: white}]`. Three tiers, each a
  strict fallback for the last: a strict-JSON **LLM** call; an offline
  **dependency parse** (spaCy) that reads the binding off the syntax tree; and
  a zero-dependency **adjacency rule** ("a colour binds to the nearest
  *following* garment noun" — the English pre-nominal adjective rule). The
  system passes all five eval queries and the swap test with the LLM disabled.

  *Why the syntax tier exists (and why not attention).* The adjacency rule is
  right on pre-nominal colour and silently wrong everywhere else: measured on a
  19-construction suite it scored **5/19**, dropping the colour entirely on
  "a shirt that's red" / "a shirt in red" / "the shirt is white" (the clause
  degrades to category-only, so *any*-coloured shirt scores 1.0 — the binding
  machinery quietly switches off) and mis-binding coordination ("a red and
  white shirt" → white; "a red shirt and tie" → colourless tie). The obvious
  fix is to read the binding out of CLIP's own attention — but that is
  *circular*, and measurably useless: CLIP's text encoder is causal (a colour
  token cannot attend forward to its noun at all — weight 0.000), and its
  backward attention is **positional, not lexical**: swapping the colours in
  "a shirt in red and pants in blue" moves the weights by <0.003. It would
  reproduce the adjacency heuristic, failures included. Syntax is the right
  instrument for syntax: a dependency parse maps `amod`/`acomp`/`relcl`/
  `prep`+`pobj`/`conj` onto exactly the bindings we need, offline and
  deterministically — **19/19** on the construction suite. Note this bug was
  invisible to the benchmark (all five eval queries are pre-nominal), which is
  why that suite is now a regression test in its own right.

  *…but the syntax tier alone is not enough either, and this is why the parser
  routes.* Dependency parsers assume **grammatical** input. Real search queries
  often are not: "red shirt blue pants", "navy blazer grey trousers" have no
  verb and no determiners, and the small spaCy model reads them as a single
  compound noun-chain (`shirt --compound--> pants`), collapsing the garments and
  re-attaching colours to the wrong nouns. On such inputs the syntax tier scores
  **8/12 — worse than the adjacency rule's 9/12**, and its errors are *active
  mis-bindings* rather than silent drops. So the two offline parsers fail on
  **disjoint** inputs, and `parse()` arbitrates: if the tree recovers fewer
  garments (or fewer bound colours) than a plain lexical scan, the tree has
  collapsed and adjacency wins. Routing scores **18/19** overall, beating either
  parser alone (15/19 syntax, 14/19 adjacency).

  *Would a transformer parser be better?* Measured, rather than assumed — and
  the answer is yes, at a price. Swapping spaCy's convolutional encoder for its
  RoBERTa one (`en_core_web_trf`) scores **19/19 with no router at all**,
  including 12/12 on the telegraphic fragments the small model mangles. It costs
  **440 MB and ~40 ms/query against 12 MB and ~4 ms**. The default stays on the
  small model, because the router buys back all but one case for 3% of the disk;
  `models.spacy` in `config.yaml` trades up in one line. This is the honest shape
  of the trade — a cheap model plus a cheap fallback rule beats a model 37× its
  size on 18 of 19 cases.

  Formality/environment additionally get
  a **query-side zero-shot classification** (the query embedding scored
  against the same prompt banks as images), fused with the keyword parse via a
  three-band trust ladder: unconfident → keyword only (axis drops if keywords
  found nothing, so unspecified never locks on); moderately confident →
  explicit keywords win conflicts; highly confident → the embedding wins,
  which corrects keyword false-positives ("smart *casual*" no longer reads as
  formal via the keyword "smart") and recovers axes keywords miss entirely
  ("hiking by a lake" → park at 0.97 confidence).
- **Stage 3 — attribute rerank.** *This is where precision is won.* Per parsed
  clause, the **binding score** rewards a candidate only if some single detected
  garment matches **both** category and colour: an image with a blue tie and a
  red shirt earns almost nothing for "red tie". Colour matching is continuous
  in LAB space (perceptually uniform, so delta-E ≈ human colour difference):
  the query word is grounded to its own LAB point via a ~45-word colour
  lexicon, so *maroon* is a target near-but-not-equal-to red, *navy* earns
  ~0.6 of a *blue* query's credit, and *white* bottoms out near zero. An
  out-of-palette word (*burgundy*, *teal*) therefore stays a constraint
  instead of silently vanishing. Formality and
  environment use the stored zero-shot **probabilities**, not argmax labels, so
  a near-tie classification is not punished like a confident mismatch. Axes the
  query never mentions are excluded from the weight normalisation — unspecified
  never penalises. Final score: `α·cosine + β·attr` (α=0.45, β=0.55, weights in
  `config.yaml`).

Two design details worth calling out:

- **Per-garment colour, not per-image colour.** Dominant colour is K-means
  (k=3) over the *detected garment's crop* in LAB. Whole-image colour would
  reintroduce exactly the bag-of-concepts failure we are fixing.
- **Greedy one-to-one clause matching.** One white shirt cannot satisfy both
  "white shirt" and "white pants" — each detection pays for at most one clause.

**How the five assignment queries flow through it:**

| Query | Dense stage contributes | Structured stage contributes |
|---|---|---|
| 1. "bright yellow raincoat" | raincoat-like images | parser: (coat, yellow) + outerwear; binding score demands yellow *on the coat* |
| 2. "business attire inside a modern office" | suits/formal looks | formality=formal (prob.), environment=office (prob.) |
| 3. "blue shirt sitting on a park bench" | people outdoors in shirts | binding (shirt, blue) + environment=park |
| 4. "casual weekend outfit for a city walk" | **carries the query** — pure style inference, no literal labels | formality=casual, environment=street nudge the top |
| 5. "red tie and white shirt, formal" | ties/shirts/formal wear | **the differentiator**: two clauses, each must bind on its own garment |

---

## 3. Results

*Corpus: the **full 3,200-image** Fashionpedia val_test2020 pool, fully
indexed (the system was developed on a 693-image subsample, then scaled up
and re-tuned against full-corpus error analysis). Relevance: hand-labelled by
pooled evaluation — for each query the union of top-10 results from all three
ablation systems was judged visually; criteria per query in
`eval/relevance_criteria.md`. Grids: `eval/results/*.png`.*

### Hybrid system on the five assignment queries (k=5)

| Query | P@5 | R@5 | Notes |
|---|---|---|---|
| 1 Attribute (yellow raincoat) | 0.80 | 0.67 | binding demands yellow *on the coat* |
| 2 Contextual (office) | 0.80 | 0.20 | formality+environment soft scores |
| 3 Complex (blue shirt, park bench) | 0.40 | 0.33 | see failure analysis below |
| 4 Style (casual weekend) | 1.00 | 0.14 | dense zero-shot carries it |
| 5 Compositional (red tie, white shirt) | 0.80 | 0.40 | corpus caveat below |
| **Mean** | **0.76** | **0.35** | |

### Ablation — P@5, all three systems, all five queries (full corpus)

| Query | Vanilla CLIP dense | FashionCLIP dense | **Hybrid (ours)** |
|---|---|---|---|
| attribute | 0.60 | 0.60 | **0.80** |
| contextual | 0.40 | 1.00 | 0.80 |
| complex | 0.20 | 0.40 | 0.40 |
| style | 0.80 | 1.00 | 1.00 |
| compositional | 0.80 | 0.40 | 0.80 |
| **Mean** | 0.56 | 0.68 | **0.76** |

Vanilla→FashionCLIP isolates the fine-grained-fashion gain (+0.12);
FashionCLIP→hybrid isolates the structured-attribute gain (+0.08 mean, +0.20
on the attribute query where colour binding decides, and +0.40 on the
compositional query, where FashionCLIP's dense ranking actually *trails
vanilla* — better fashion features do not fix binding; structure does).

**Corpus caveat on query 5, and how we measured binding anyway.** An
exhaustive FashionCLIP probe over the full 3,200-image pool found **no genuine
red necktie** — vanilla CLIP's best "red tie" matches pool-wide are red
*dresses* (the attribute leaks to the wrong garment — the bag-of-concepts
failure in one picture, `eval/pool/` shows it). Relevance for query 5 was
therefore relaxed to "necktie + white shirt + formal", which removes the
colour-binding signal and lets a dense suit-retriever tie the hybrid. So we
added a **swap-pair binding probe** on a colour pair the corpus *does*
contain, with a strong natural asymmetry: ~23 white-top+black-pants images vs
exactly **one** verified black-top+white-pants image. Querying the rare
direction floods a bag-of-concepts retriever with swapped matches:

*(At full corpus the rare direction has 4 verified true matches among ~26
white-top+black-pants images.)*

| System | "black shirt + white pants" (rare): P@5 and true-match ranks | "white shirt + black pants" (control): P@5 |
|---|---|---|
| Vanilla CLIP dense | 0.00 — best true match at #10 | 0.40 |
| FashionCLIP dense | 0.40 — ranks #1, #2, #8 | 0.60 |
| **Hybrid (ours)** | **0.60 — ranks #1, #3, #4** | **1.00** |

This is the compositionality result in its purest measurable form on this
corpus: same words, opposite binding — only the hybrid reliably separates
them. The swap unit test (`tests/test_compositionality.py`) proves the same
mechanism analytically: identical-cosine synthetic images with swapped
shirt/pants colours rank in opposite orders under the two swapped queries.

**Failure analysis — query 3 (blue shirt + park bench).** The hardest query:
a garment clause AND a scene clause, multiplying both subsystems' error
rates. Full-corpus error analysis drove three measured fixes (each A/B'd,
two other candidate fixes were rejected by the data — see below): after them,
the four best-detected relevant images rank #1, #3, #6, #8. What remains is
instructive: the top false positives are a real navy shirt failing only the
un-modelled *sitting* posture, and denim tees on the street; the missed
relevant images are detector failures (a blue jersey whose K-means colour
reads brown, a plaid shirt averaging to gray). Pose modelling,
segmentation-based colour, and a real scene classifier (future work) target
exactly these residuals.

**What full-corpus error analysis changed — and what it refused to change.**
Scaling from 693 to 3,200 images was used as a tuning pass, one lever at a
time with the eval re-run after each:
1. **Achromatic gate (added):** black shirts (measured chroma 5-8) were
   earning 0.30 credit toward "blue" — as much as a genuinely bluish-gray
   tee. A garment with no chroma now earns only the floor toward a chromatic
   query. Black-shirt false positives left the complex top-5; no blue-hued
   garment (navy 24, bluish-gray 15) was touched.
2. **Absent-credit 0.3 → 0.2 (adjusted):** garmentless park photos were
   riding the environment score past true matches because "no garment
   detected" scored the same as real-but-imperfect colour evidence.
3. **Colour ramp tightening (rejected):** narrowing `color_soft_max` 100→80
   killed the beige→yellow leak but dropped complex P@5 from 0.40 to 0.20 —
   the query's true matches are navy/denim tops living exactly in the
   punished delta-E band. Reverted.
4. **Wider dense recall (rejected):** doubling top-N to 100 *hurt* (complex
   0.40→0.20). The misses were never recall failures — dense already ranked
   them #2/#5/#6 — they were reranker demotions of detector-blind images,
   and the wider pool only imported extra distractors.

**Error trade-off we accepted.** The garment detector runs at a deliberately
low confidence threshold (0.25): a missed garment silently destroys binding
evidence, while a spurious detection is absorbed by soft scoring (wrong-colour
matches floor out near zero). The reranker encodes the matching evidence
hierarchy explicitly: correct binding (1.0) ≫ perceptual near-miss (e.g. navy
for "blue" ≈ 0.6) ≫ category never detected (0.30, weak evidence) > detected
with contradicting colour (0.05, strong counter-evidence). The colour ramp is
calibrated against palette-anchor spacing: its decay length (delta-E 100) is
about twice the typical distance between neighbouring named colours, so
neighbours score partial credit while opposites hit the floor — with a decay
length below that spacing, soft matching would degenerate to
exact-name-or-floor.

---

## 4. Code

GitHub: **https://github.com/Varshith-06/GlanceAssignment** — `README.md` has
setup, run commands and the architecture diagram. Runs CPU-only, offline (LLM
parser optional). `pip install -r requirements.txt && python
indexer/build_index.py` rebuilds everything from scratch; all stages are
idempotent.

---

## 5. Future work

**Locations & weather.** Add a dedicated scene classifier (Places365) rather
than CLIP prompts → richer environment embedding + city metadata as filterable
fields; add weather tags (rainy/sunny/snowy) either from image classification
or paired EXIF/caption data; extend the query parser to emit `{location,
weather}` slots that feed the same reranker — the scoring framework already
generalises to new axes by adding a weight. (We prototyped the weather axis:
it wires in exactly as described, but on this corpus visual weather cues
appear in <1% of photos, and weather *inferred* from garment words — "yellow
raincoat" → rainy at 0.92 confidence — penalised correct garment matches shot
in dry conditions. The axis pays off only with data that actually encodes
weather, e.g. EXIF/caption pairing, so it stays future work.)

**Precision.** (1) Fine-tune FashionCLIP with **hard negatives**: auto-generate
colour-swapped caption pairs ("red tie, white shirt" ↔ "white tie, red shirt")
and train contrastively — this attacks compositionality *at the embedding
level* instead of patching it post-hoc. (2) Add a cross-encoder reranker
(e.g. BLIP-2 ITM) over the final top-k — joint attention over image+text sees
binding natively; affordable because k is small. (3) Replace box crops with
garment *segmentation* masks before LAB colour naming — removes background
bleed, the main colour-extraction error source.

**Scaling to 1M images.** Swap Chroma's flat search for FAISS IVF/HNSW —
sub-linear ANN keeps the dense stage ~O(log n); pre-filter candidates by
structured metadata (e.g. `environment=office`) before ranking to shrink the
candidate set; batch/GPU embedding for indexing throughput. The key
architectural property is already in place: **only the top ~50 candidates are
ever reranked, so rerank cost is constant regardless of corpus size** — the
expensive precision machinery never touches the full corpus.
