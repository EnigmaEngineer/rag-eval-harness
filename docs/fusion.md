# Why reciprocal rank fusion, not weighted score fusion

The retriever has two rankers. A dense one over `bge-small-en-v1.5` cosine similarity, and
a BM25 sparse one. They disagree, and the disagreement is the value. Dense catches paraphrase
(q002 asks about "tiny tasks after a shuffle", the doc says "partitions"). BM25 catches exact
tokens dense smooths over (a specific config key, an error class name). Fusion is how you keep
both strengths in one ranked list.

There are two common ways to fuse. I picked reciprocal rank fusion. Here is why.

## Weighted score fusion, and why it fights you

The obvious approach is `final = a * dense_score + (1 - a) * bm25_score`. It fails on a detail
that is easy to miss. The two scores are not on the same scale.

Dense cosine here is a similarity in roughly 0 to 1. BM25 is an unbounded sum of per-term
idf-weighted contributions. A BM25 score of 18 and a cosine of 0.62 are not comparable numbers.
To add them you first have to normalise, and every normalisation has a problem.

Min-max scaling maps each list to 0 to 1, but the min and max come from the candidate set you
happened to retrieve for this query. The same document gets a different normalised score
depending on what else came back. Z-score assumes a distribution the scores do not follow.
Either way the mixing weight `a` becomes a hyperparameter you have to tune, and you cannot tune
it honestly until there is a metric to tune against. That metric is the day-5 harness. Setting
`a` before then would be guessing dressed up as a number.

## What RRF does instead

RRF throws the scores away and keeps only the ranks.

```
score(d) = sum over rankers of  1 / (k + rank_r(d))
```

`rank` starts at 1. A document at rank 3 in either list contributes `1 / (k + 3)` regardless of
whether its raw score was 0.9 or 0.55 or 18.2. There is nothing to normalise because there is
nothing left on a raw scale. The two rankers become directly comparable by construction.

`k` damps the head of each list so one ranker's rank-1 cannot swamp everything below it. Small
`k` trusts the top ranks hard. Large `k` flattens the contribution curve toward uniform. 60 is
the value from the original RRF paper (Cormack et al. 2009) and the common default.
It is a parameter in `fuse.py`, so the day-5 harness can sweep it if the metrics ask for it,
rather than it being baked in.

## The honest cost

RRF discards magnitude. A document the dense ranker is wildly confident about and a document
it barely preferred over the next one look identical if they share a rank. In a case where the
score gap is real signal, weighted fusion could in principle beat RRF. The bet here is that
across a query set the rank information is robust and the score scales are not, which is the
same bet the RRF paper made and won against tuned weighted combinations.

This is a bet, not a proof. The day-5 ablation measures dense only, BM25 only, and fused on
recall@k and MRR. If fused does not beat both inputs there, this decision gets revisited with
weighted fusion as the alternative and a real number deciding it.
