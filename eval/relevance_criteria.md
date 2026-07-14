# Evaluation criteria

How `eval/relevance.json` was produced, and how to read the numbers it yields.

## Method: pooled evaluation

There is no ground truth shipped with this corpus, so relevance was
hand-labelled using **pooling** (the standard TREC construction):

1. Run all three ablation systems — vanilla CLIP dense, FashionCLIP dense, and
   the full hybrid — on each query.
2. Take the **union of their top-10** results. This is the *pool*
   (`eval/pool/pools.json`; ~20–26 images per query).
3. Judge every pooled image visually against fixed, written criteria (below)
   and record the relevant ids.

Pooling from **all three** systems is what keeps the comparison fair: if the
pool were built from the hybrid's results alone, the hybrid would be scored
against a relevance set biased toward what it happens to return, and the
ablation would be meaningless.

Labels are **binary** (relevant / not), and the relevance set is the union of
judgments from two rounds — the 693-image development corpus and the final
3,200-image corpus. Every id in it was judged by eye; none were inferred.

## Per-query criteria

| Query | Relevant iff | Labelled | Notes |
|---|---|---|---|
| **attribute** — *"a person in a bright yellow raincoat"* | a yellow **outer layer** (raincoat / coat / jacket) is worn | 6 | yellow *dresses, pants or under-layers* do **not** count — the point is that the colour binds to the coat |
| **contextual** — *"professional business attire inside a modern office"* | unambiguous business attire (suit, or blazer + formal trousers/skirt) | 20 | the corpus has almost no true indoor-office photos, so the *setting* was not required; jeans and styled-edgy looks excluded |
| **complex** — *"someone wearing a blue shirt sitting on a park bench"* | blue top **and** seated outdoors (bench / park) | 6 | the conjunction *is* the test: blue shirts standing, or bench-sitters in other colours, do **not** count |
| **style** — *"casual weekend outfit for a city walk"* | casual outfit plausible for a city walk | 36 | a street-fashion corpus, so most pooled candidates qualify; dress shirts and heels excluded |
| **compositional** — *"a red tie and a white shirt in a formal setting"* | a necktie worn with a white/light shirt, formal attire | 10 | see the corpus caveat below; a tie over a dark or checked shirt does **not** count |
| **swap_rare** — *"a black shirt and white pants"* | dark top **and** white/light trousers | 4 | the binding probe — see below |
| **swap_common** — *"a white shirt and black pants"* | light top **and** dark trousers | 16 | the control direction of the same probe |

## Two caveats that shape the numbers

**1. The corpus contains no red necktie.** An exhaustive FashionCLIP probe over
all 3,200 images found none — the best pool-wide matches for *"red tie"* are red
*dresses* (which is itself a neat illustration of the bag-of-concepts failure:
the attribute leaks onto the wrong garment). The compositional query's relevance
therefore **relaxes the tie's colour** while keeping every other constraint. That
weakens it as a binding test, because a dense suit-retriever can score well
without binding anything.

That is exactly why the **swap-pair probe** exists. It is built from a colour
combination the corpus *does* contain, with a strong natural asymmetry: ~26
white-top+black-pants images versus only 4 of the reverse. Querying the **rare**
direction floods a bag-of-concepts retriever with swapped matches — the same
words, the wrong binding — so only a system that actually binds colour to
garment can rank the true matches first. The common direction is the control:
every system should do well there.

**2. Recall@5 is capped by the size of the relevance set.** With 36 relevant
images for the style query, the best achievable R@5 is 5/36 = 0.14 — which is
precisely the score the hybrid gets, on a query where it returns **5 of 5**
correct. Read **P@5 as the headline metric** and R@5 as secondary; recall is only
meaningfully comparable *between systems on the same query*, never across queries
with different pool sizes.

**3. Pooled recall is relative to the pool.** Standard TREC caveat: an image no
system ever retrieved was never judged, so it cannot be counted relevant. Recall
is therefore measured against the pooled set, not against the whole corpus.

## Metrics

- **Precision@k** — of the top *k* returned, the fraction that are relevant.
  The headline number.
- **Recall@k** — of all labelled-relevant images, the fraction that appear in
  the top *k*. Subject to caveats 2 and 3 above.
- **Rank of true matches** — reported for the swap probe, where the relevance
  sets are small and *where* a match lands matters more than how many appear.

Implementation: `eval/metrics.py`. Runners: `eval/run_eval.py` (five assignment
queries) and `eval/ablation.py` (three systems + the swap probe).
