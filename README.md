# Multimodal Fashion & Context Retrieval

Natural-language image search over a fashion corpus, built to beat vanilla CLIP
on the failure it is known for: **compositionality**. CLIP embeds an image as a
bag of concepts — it cannot reliably tell *"red tie and white shirt"* from
*"white tie and red shirt"*, because attribute→object **binding** does not
survive the pooled vector.

This system fixes that with a **hybrid retrieve-then-rerank** pipeline: dense
recall for coverage, then a structured reranker that scores each candidate on
whether the right colour actually sits on the right garment.

Full write-up (approach, trade-offs, results, future work):
**[`deliverable/writeup.pdf`](deliverable/writeup.pdf)**

---

## Results

Corpus: the full **3,200-image** Fashionpedia `val_test2020` pool. Relevance
labels produced by pooled evaluation — for each query, the union of the top-10
results from all three systems was judged by hand
([criteria](eval/relevance_criteria.md)).

**Precision@5, all three systems, the five assignment queries:**

| Query | Vanilla CLIP | FashionCLIP | **Hybrid (this system)** |
|---|---|---|---|
| Attribute — *"a bright yellow raincoat"* | 0.60 | 0.60 | **0.80** |
| Contextual — *"business attire in a modern office"* | 0.40 | 1.00 | 0.80 |
| Complex — *"blue shirt sitting on a park bench"* | 0.20 | 0.40 | 0.40 |
| Style — *"casual weekend outfit for a city walk"* | 0.80 | 1.00 | 1.00 |
| Compositional — *"a red tie and a white shirt"* | 0.80 | 0.40 | 0.80 |
| **Mean P@5** | 0.56 | 0.68 | **0.76** |

**The compositionality result.** The assignment's compositional query cannot
fully separate the systems on this corpus — an exhaustive search found it
contains *no red necktie at all*. So binding is measured directly with a
**swap-pair probe** built from a colour combination the corpus does contain,
with a strong natural asymmetry (~26 white-top+black-pants images vs 4
black-top+white-pants). Querying the **rare** direction floods a
bag-of-concepts retriever with swapped matches — same words, wrong binding:

| System | Rare direction (P@5) | Control direction (P@5) |
|---|---|---|
| Vanilla CLIP | 0.00 — best true match at rank #10 | 0.40 |
| FashionCLIP | 0.40 | 0.60 |
| **Hybrid** | **0.60** — true matches at #1, #3, #4 | **1.00** |

Note FashionCLIP-dense actually *trails vanilla* on the compositional query
(0.40 vs 0.80): **better fashion features do not fix binding — structure does.**

Result grids for every query: [`eval/results/`](eval/results).

---

## Architecture

```
INDEXER (offline, once)
  image ──► FashionCLIP image embedding ─────────────┐
        └─► YOLOS-Fashionpedia garment detection      ├──► ChromaDB {vector, metadata}
            + per-garment K-means colour in LAB        │    + data/metadata.jsonl
            + zero-shot formality & environment        ┘

RETRIEVER (per query)
  query ──► Stage 1  Dense recall — FashionCLIP text embed → top-50 candidates
        ──► Stage 2  Parse into structure — {garments:[{item,color}], formality, environment}
        ──► Stage 3  Rerank — final = α·cosine + β·attribute_match → top-k
```

**Stage 1 — recall.** Deliberately dumb: cast a wide semantic net. Zero-shot
generality lives here (*"casual weekend outfit"* needs no label to work).

**Stage 2 — structured parsing.** *Where compositionality is solved.* Binding
is lifted out of the smeared embedding into explicit structure. Three tiers,
each a strict fallback for the last: an **LLM** (optional) → an offline
**dependency parse** that reads bindings off the syntax tree → an **adjacency
rule** with zero dependencies. Colour words are grounded to continuous LAB
targets, so *maroon* is a point near red rather than a dropped constraint.

**Stage 3 — attribute rerank.** *Where precision is won.* Per parsed clause,
the **binding score** rewards a candidate only if a *single detected garment*
matches **both** the category and the colour. An image with a blue tie and a
red shirt earns almost nothing for *"red tie"* — even though CLIP's cosine
cannot tell it from the correct image. The evidence hierarchy is explicit:

| Evidence | Clause score |
|---|---|
| Right garment, right colour | **1.0** |
| Right garment, perceptual near-miss (navy for *blue*) | ~0.5–0.9 |
| Garment never detected — weak evidence | 0.20 |
| Garment detected in a contradicting colour — strong counter-evidence | 0.05 |

Every weight and threshold lives in [`config.yaml`](config.yaml) and was chosen
by A/B measurement on the full corpus, not by hand.

---

## Two findings worth reading

**CLIP's own attention cannot recover binding — measured, not assumed.** The
obvious shortcut for parsing *"a shirt that's red"* is to read the binding out
of CLIP's text-encoder attention. It doesn't work, for two independent reasons:
the encoder is **causal** (a colour token cannot attend forward to its noun —
weight is structurally 0.000), and its backward attention is **positional, not
lexical** — swapping the colours in *"a shirt in red and pants in blue"* moves
the weights by <0.003. It would reproduce the adjacency heuristic, failures
included. Syntax is the right instrument for syntax; hence the dependency tier.

**The benchmark was blind to a whole bug class.** All five assignment queries
put the colour *before* the noun. Measured against a 19-construction suite, the
original adjacency rule scored **5/19** — silently dropping the colour on *"a
shirt that's red"* and mis-binding *"a red and white shirt"* — yet cost exactly
0.00 P@5 on every reported number. The dependency parser takes it to **19/19**,
and that suite is now a regression test
([`tests/test_query_parsing.py`](tests/test_query_parsing.py)). A clean
scoreboard is not evidence of a clean system.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows  (source .venv/bin/activate on Unix)
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # dependency parser for query binding
```

Dataset: place the Fashionpedia `val_test2020/test` image folder at the repo
root (path configurable in `config.yaml`).

Runs **CPU-only and fully offline** — no API keys. Setting `ANTHROPIC_API_KEY`
enables the LLM query parser as an enhancement; every result above was produced
with it disabled.

## Run

```bash
# 1. Build the index  (idempotent; embeddings cached)
python indexer/build_index.py

# 2. Search
python retriever/search.py "a red tie and a white shirt in a formal setting"

# 3. Evaluation — 5 assignment queries → grids + P@k/R@k in eval/results/
python eval/run_eval.py

# 4. Ablation — vanilla CLIP vs FashionCLIP vs hybrid, plus the swap-pair probe
python eval/ablation.py

# 5. Tests — includes the swap test and the 19-construction parser suite
python -m pytest tests -q
```

## Layout

| Path | Role |
|---|---|
| `config.yaml` | every knob: models, N/k, α/β, rerank weights, colour palette |
| `indexer/download_data.py` | diversity-aware subsample → `data/raw/` |
| `indexer/embed.py` | FashionCLIP image/text embeddings (L2-normalised) |
| `indexer/attributes.py` | garment detection, LAB colour, formality, environment |
| `indexer/build_index.py` | orchestrates → ChromaDB + `metadata.jsonl` |
| `retriever/query_parser.py` | text → structure (LLM → dependency parse → adjacency) |
| `retriever/dense_search.py` | Stage 1 recall |
| `retriever/rerank.py` | Stage 3 binding/attribute scoring — pure, no I/O |
| `retriever/search.py` | public `search(query, k)` entrypoint |
| `eval/` | 5-query eval, metrics, 3-system ablation, labelling tools |
| `deliverable/writeup.md` → `.pdf` | the full write-up |

Modules are decoupled by design: `search.py` never touches embedding internals,
and `rerank.py` is pure data-in/data-out — no DB, no model — so the scoring
logic is unit-testable in isolation.
