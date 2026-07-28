"""Unit tests for the eval metrics.

    python -m tests.test_metrics

The metrics decide whether hybrid retrieval stays in this project, so a quiet bug here would
be worse than a quiet bug in a retriever. A retriever that is wrong looks wrong. A metric that
is wrong makes everything look fine.

Hand-built ranked lists throughout. No index, no model.
"""

import sys

from harness import metrics
from harness.run_eval import required_evidence, parse_slice


def test_recall_counts_the_gold_set_not_the_hits():
    ranked = ["a", "b", "c", "d", "e"]
    # two of three gold chunks in the top 5. the denominator is the gold set size, so more
    # gold chunks make the same k harder. that is the q003 case in the real eval set.
    assert metrics.recall_at_k(ranked, {"a", "c", "z"}, 5) == 2 / 3
    assert metrics.recall_at_k(ranked, {"a"}, 5) == 1.0
    assert metrics.recall_at_k(ranked, {"z"}, 5) == 0.0


def test_recall_respects_the_cut():
    ranked = ["a", "b", "c", "d", "e"]
    assert metrics.recall_at_k(ranked, {"d"}, 3) == 0.0
    assert metrics.recall_at_k(ranked, {"d"}, 5) == 1.0


def test_recall_refuses_an_empty_gold_set():
    # returning 0.0 here would silently drag a mean down and look like a retrieval failure.
    try:
        metrics.recall_at_k(["a"], set(), 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an empty gold set")


def test_hit_ignores_gold_set_size():
    ranked = ["a", "b", "c"]
    assert metrics.hit_at_k(ranked, {"a", "y", "z"}, 5) is True
    assert metrics.hit_at_k(ranked, {"a"}, 5) is True
    assert metrics.hit_at_k(ranked, {"z"}, 5) is False


def test_reciprocal_rank_uses_the_first_gold_chunk():
    ranked = ["a", "b", "c", "d"]
    assert metrics.reciprocal_rank(ranked, {"a"}) == 1.0
    assert metrics.reciprocal_rank(ranked, {"c"}) == 1 / 3
    # two gold chunks, the earlier one wins. rr is about how far the user scrolls.
    assert metrics.reciprocal_rank(ranked, {"b", "d"}) == 1 / 2
    assert metrics.reciprocal_rank(ranked, {"zz"}) == 0.0


def test_reciprocal_rank_truncates_before_scoring():
    ranked = ["a", "b", "c", "d", "e", "f", "g"]
    # the day-4 q002 shape: the right chunk sits at rank 6, so mrr@5 and mrr@10 disagree.
    # a harness that does not say which k it used is not reproducible.
    assert metrics.reciprocal_rank(ranked, {"f"}, k=10) == 1 / 6
    assert metrics.reciprocal_rank(ranked, {"f"}, k=5) == 0.0


def test_evidence_support_is_a_fraction_of_required_strings():
    texts = ["set spark.sql.adaptive.enabled to true", "nothing relevant here"]
    assert metrics.evidence_support(texts, ["spark.sql.adaptive.enabled"]) == 1.0
    assert metrics.evidence_support(
        texts, ["spark.sql.adaptive.enabled", "spark.serializer"]) == 0.5
    # case insensitive, because the docs write config keys inside code tags and prose does not
    assert metrics.evidence_support(["Spark.Serializer"], ["spark.serializer"]) == 1.0
    # no requirement means nothing to fail
    assert metrics.evidence_support(texts, []) == 1.0


def test_required_evidence_pulls_config_keys_and_spans():
    row = {
        "answer": "Enable it with spark.sql.adaptive.coalescePartitions.enabled and check "
                  "spark.sql.shuffle.partitions.",
        "answer_spans": ["adaptive query execution"],
    }
    assert required_evidence(row) == [
        "spark.sql.adaptive.coalescePartitions.enabled",
        "spark.sql.shuffle.partitions",
        "adaptive query execution",
    ]
    # a trailing full stop must not end up inside the key
    assert required_evidence({"answer": "set spark.serializer."}) == ["spark.serializer"]


def test_parse_slice():
    assert parse_slice("") is None
    assert parse_slice("0:3") == (0, 3)
    assert parse_slice("5:") == (5, None)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as err:
            failed += 1
            print(f"  FAIL  {t.__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
