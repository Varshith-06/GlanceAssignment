# Hand-labelling criteria for eval/relevance.json

Labels were produced by pooled evaluation: for each query, the union of top-10
results from all three ablation systems (vanilla CLIP dense, FashionCLIP dense,
full hybrid) was judged visually (contact sheets in `eval/pool/`). Binary
relevance, criteria per query:

| Query | Relevant iff | Notes |
|---|---|---|
| attribute | yellow raincoat / yellow outer jacket or coat is worn | yellow dresses, pants or under-layers do NOT count |
| contextual | unambiguous professional business attire (suit / blazer + formal trousers or skirt) | the corpus has almost no true indoor-office photos, so setting was not required; jeans or styled-edgy looks excluded |
| complex | blue top AND seated outdoors (bench/park context) | blue shirts standing, or bench-sitters in other colours, do NOT count — the conjunction is the point |
| style | casual outfit plausible for a city walk | street-fashion corpus, so most pooled candidates qualify; dress shirts / heels excluded |
| compositional | necktie worn with a white/light shirt in formal attire | **the full 3,200-image pool contains no genuine red necktie** (verified by an exhaustive FashionCLIP probe — best "red tie" matches pool-wide are red *dresses*). Relevance therefore relaxes the tie colour while keeping every other binding constraint; images with a tie on a dark/checked shirt do NOT count |

Caveat: pooled labelling only judges images some system retrieved; recall is
relative to the pooled set, as in standard TREC-style pooling.
