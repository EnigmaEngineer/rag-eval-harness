# How the gold chunks were labelled

Day 5 needs recall@k and MRR. Both need to know which chunk is the right one. The golden set
only ever carried `source_docs`, a filename. That is a doc-level label. Every smoke test up to
day 4 scored "did an expected source doc appear in the top 5", which is a proxy and was always
described as one.

A doc-level label cannot tell a good retriever from a lucky one. `sql-performance-tuning.md` is
19 chunks. Returning any of them counts as a hit. Three of the ten questions cite that one file,
so a retriever that has learned nothing except "shuffle questions live in the tuning doc" scores
well on the proxy. That is the gap this file closes.

## The rule

A chunk is gold if, read on its own with no other context, it would let a competent reader
answer the question.

Supporting context is not gold. A chunk that mentions the topic, or that you would need
alongside another chunk to get to the answer, does not qualify. The rule is applied the same way
to all ten questions and the per-question reasoning is in the `why` field of
`evalset/gold_chunks.jsonl`.

## How the candidates were found

Not with the retrievers. Labelling by asking the system under test what the answer is would make
the metric circular and would bake in whatever the retriever already prefers.

Instead every chunk of every cited source doc was listed by heading path and read. Those docs run
17 to 62 chunks each, which is small enough to read directly. Exact-string searches for the
config keys in the reference answers were used to locate candidates faster, then each candidate
was read before it was accepted or rejected. String search narrowed where to look. It never
decided the label.

## What this measurement is honest about

**The labels are minimal-sufficient, not exhaustive.** For q004 the chunks on state cleanup and
on drop guarantees are genuinely relevant, and they are deliberately not gold because neither
defines a watermark. If the retriever returns one of them it is scored as a miss. So the recall
numbers here are a floor. The true relevance-based recall is somewhat higher and this harness
cannot see it.

**Gold set sizes differ, so recall@k is not comparable across questions.** q001 has one gold
chunk and q003 has three, so a perfect retriever gets recall@1 of 1.0 on the first and 0.33 on
the second. That is arithmetic, not a retrieval failure. The harness reports macro-averaged
recall@k across questions and also `hit@k`, which asks only whether at least one gold chunk made
the cut. Read the two together.

**Ten questions is a small sample.** One question moving changes any of these numbers by ten
percentage points. Treat a single-question difference between two systems as noise. This set
exists to catch a regression, not to rank retrievers to two decimal places.

## The q005 defect

Labelling turned up a problem in the golden set itself. q005 asks whether a broadcast join works
when one side does not fit in driver memory. The reference answer says the smaller relation must
fit in driver and executor memory. No chunk in either cited source doc says that. The threshold
config is well grounded and the memory constraint is not.

`evalset/validate.py` did not catch it because its grounding check only verifies config keys and
explicit `answer_spans`, and q005 has no spans while its one config key is present. The check is
doing what it was written to do. It just cannot see a prose claim.

The answer has been left alone rather than quietly rewritten to match the docs. Editing the
eval set so the numbers behave is the failure mode this whole project is about. It is recorded
here and in the README limitations instead.
