# Limitations

Moved out of the README so the front page stays short. Back to the [README](../README.md).

## Known limitations

The golden set has 10 questions. That is enough to catch gross breakage and not enough to
trust a two-point recall difference. Target is 60, added when a real retrieval failure turns
up that no existing question covers.

The chunker special-cases HTML tables but not markdown pipe tables. A few markdown rows are
written as one very long physical line and only survive because the oversized-block
windower falls back to splitting on spaces. That is a safety net, not real markdown-table
handling. It is on the list.

A heading path long enough to eat the token budget will produce a chunk over that budget.
`pack()` repeats the heading path on every chunk and floors the room left for content at 96
tokens, so content never gets squeezed to nothing. The cost is an overrun. That overrun is
not hidden. `emit()` flags `oversized` against the budget and never against the floored room,
so one of these chunks is counted in the over-budget total like any other, and
`tests/test_chunk.py` pins that. It does not fire on this corpus. The longest heading path
here leaves 448 tokens of room by the approximate counter, well clear of the floor.

BM25 does no stemming. "partition" and "partitions" are distinct terms. It did not cost a
golden hit on this corpus, but a query using the singular against a doc that only uses the
plural would miss. A real stemmer is a dependency and a source of surprising matches, so it
is deferred rather than added blind.

The reranker is an off-the-shelf cross-encoder trained on MS MARCO web search, not on Spark
docs. q002 shows the cost. It reads shuffle-tuning prose and cannot separate the right fix
from the wrong one when both use the same words. A domain-tuned reranker would likely close
that gap. Fine-tuning one is out of scope for this project and noted rather than attempted.

At roughly 3.5 seconds a query on 2 CPU cores, reranking every query is not something you
would ship as is. The pool size is a knob (`--pool`) and the honest production answer is a
GPU or a shorter pool. The point here is to measure the tradeoff, not to hide it.

The smoke checks report source-doc overlap at top-5, not recall@k on gold chunks. They
are kept as a fast sanity proxy for spotting where the retrievers disagree.
`harness/run_eval.py` is the metric that counts.

The gold chunk labels are minimal-sufficient rather than exhaustive. A chunk is gold
only if it would let a reader answer the question on its own. Chunks that are relevant
but not sufficient are scored as misses, so every recall number here is a floor. q009 is
the clearest case. Three systems return the history server configuration table, which is
the same section as the labelled chunk and arguably useful, and score zero for it. The
labels were fixed before any system was run and have not been touched since. Moving a
label after seeing a score is how an eval set stops meaning anything. Full method in
`docs/labelling.md`.

q005's reference answer claims a broadcast join needs the smaller relation to fit in
driver and executor memory. No chunk in either cited source doc says that.
`evalset/validate.py` missed it because its grounding check verifies config keys and
declared spans, and q005 has no spans while its one config key is present. The answer
has been left as written rather than quietly edited to match the corpus.

`python -m evalset.validate --claims` now scores every prose clause against the
best-matching paragraph in its source docs and lists the weakest first. **It is advisory
and it does not cleanly find the defect.** q005's fabricated claim scores 0.50 and lands
second on a list of ten, below a correct q006 claim at 0.40. So it narrows ten claims to
a shortlist worth reading by hand and it is not a gate. Two sharper approaches were
measured and thrown away first. Anchoring each sentence to a config key flagged 8
clauses of which 1 was the real defect. Content bigram coverage scored the fabricated
clause at 0.00 and scored three correct clauses at 0.00 as well. Neither discriminates,
and shipping either would have looked rigorous while being noise.

Faithfulness is not measured, because there is no generator in this repo to be faithful
or unfaithful. What `support@5` measures is whether the retrieved context even contains
the evidence a correct answer would need. That is a ceiling on faithfulness rather than
a measurement of it, and it is named accordingly. That ceiling is known not to bind. It
returns 1.0 on 30 of 40 rows, 7 of the 10 questions rest on a single required string,
and on q005 it reports 1.0 while every system misses the answer. Treat it as a smoke
alarm for missing evidence and not as a quality score. `harness/diagnose.py` is the
check that catches it.

Every quality comparison in the ablation table is underpowered, and `p_floor()`
quantifies it rather than leaving it as a hedge. Below six questions moving, no gap can
reach p = 0.05 at any effect size. Seven of eight metric comparisons move three
questions or fewer. This is the single strongest argument for growing the golden set and
it is why "target is 60" above is a plan rather than a nice-to-have.

Ten questions, one corpus, one machine. These numbers are for catching a regression in this
project. They are not a benchmark of BM25 against dense retrieval in general, and a corpus of
API documentation dense with exact config keys is unusually kind to a lexical retriever. 11 of
the 20 gold chunks literally contain a config key from their own reference answer, so some of
BM25's advantage is the corpus and some of it is real.
