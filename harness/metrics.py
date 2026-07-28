"""Retrieval metrics. Pure functions over ranked id lists, so they are testable without
loading an index or a model.

Everything here takes `ranked` (chunk ids, best first, no duplicates) and `gold` (the set of
chunk ids from evalset/gold_chunks.jsonl). Nothing here knows what a retriever is.

Three numbers, because one is not enough:

- `recall_at_k` is the fraction of the gold set that made the cut. It is the metric that
  answers "did we get the evidence". It is also the one that punishes a question for having
  three gold chunks instead of one, which is why it is never read alone.
- `hit_at_k` asks only whether at least one gold chunk made the cut. Insensitive to gold set
  size, so it is comparable across questions. This is the closest thing to the doc-overlap
  proxy the day-2 to day-4 smokes used, but at chunk level.
- `reciprocal_rank` is about ordering. It is the metric a reranker is supposed to move, and
  the reason day 4 could not show whether the reranker was worth its 3.5 seconds.

See docs/labelling.md for what the gold sets do and do not cover.
"""


def _cut(ranked, k):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return ranked[:k]


def recall_at_k(ranked, gold, k):
    """Fraction of gold chunks appearing in the top k.

    Undefined with no gold chunks, so that raises rather than returning 0.0 and quietly
    dragging a mean down.
    """
    gold = set(gold)
    if not gold:
        raise ValueError("recall is undefined with an empty gold set")
    return len(gold & set(_cut(ranked, k))) / len(gold)


def hit_at_k(ranked, gold, k):
    """True if any gold chunk is in the top k. Gold set size cannot affect this one."""
    return bool(set(gold) & set(_cut(ranked, k)))


def reciprocal_rank(ranked, gold, k=None):
    """1 / rank of the first gold chunk, ranks starting at 1. Zero if none is found.

    `k` optionally truncates first, which is how you get MRR@k rather than MRR over the whole
    returned list. Worth being explicit about: MRR@10 and MRR@5 differ, and a paper that does
    not say which one it used is not reproducible.
    """
    gold = set(gold)
    considered = ranked if k is None else _cut(ranked, k)
    for rank, _id in enumerate(considered, start=1):
        if _id in gold:
            return 1.0 / rank
    return 0.0


def mean(values):
    """Macro average. Every question counts once regardless of how many gold chunks it has."""
    values = list(values)
    if not values:
        raise ValueError("nothing to average")
    return sum(values) / len(values)


# Evidence support, and why it is not called faithfulness.
#
# The blueprint line for today says "faithfulness scoring". Faithfulness is a property of a
# generated answer: does the text the model produced stay inside the evidence it was given.
# This repo has no generator. There is no answer to score. Writing something called
# faithfulness_score() that never sees a generated answer would be a fake metric with a
# credible name, which is worse than not having it.
#
# What can be measured today is the upstream half. Does the retrieved context actually contain
# the evidence a faithful answer would need. If it does not, the generator has no way to be
# faithful and no amount of prompting fixes it. That is the number below. It is a ceiling on
# faithfulness, not a measurement of it.

def evidence_support(retrieved_texts, required):
    """Fraction of required evidence strings present anywhere in the retrieved context.

    `required` is the exact strings a correct answer has to be able to point at: the Spark
    config keys named in the reference answer, plus any answer_spans. Matching is a
    case-insensitive substring test over the concatenated top-k text.

    Substring matching is crude and it is the right crudeness here. These strings are config
    keys and fixed phrases, not paraphrasable claims. `spark.sql.adaptive.skewJoin.enabled`
    either appears or it does not. Nothing is gained by embedding it.

    Returns 1.0 when a question requires no specific evidence, because there is nothing to
    fail to support.
    """
    if not required:
        return 1.0
    blob = "\n".join(retrieved_texts).lower()
    found = sum(1 for token in required if token.lower() in blob)
    return found / len(required)
