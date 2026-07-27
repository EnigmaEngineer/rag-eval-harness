"""Tests for the reranker's pure parts.

The cross-encoder itself is a 90 MB model and not worth loading in a unit test. What is
worth testing is everything around it. The ordering. The tie behaviour. The length check.
The rule that a candidate with no passage text is dropped rather than scored on an empty
string. So a tiny stub stands in for the model. It scores a pair by the length of the
passage, which is deterministic and lets each test assert an exact order.

    python -m tests.test_rerank
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.rerank import order_by_scores, rerank  # noqa: E402


class LengthScorer:
    """Stand-in for the cross-encoder. Scores each (query, passage) pair by passage length,
    so tests can predict the exact ranking without a model."""

    def predict(self, pairs):
        return [len(passage) for _query, passage in pairs]


def test_orders_best_first():
    cands = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    ranked = order_by_scores(cands, [0.1, 5.0, 2.0], top_k=3)
    assert [r["id"] for r in ranked] == ["b", "c", "a"]


def test_ties_keep_upstream_order():
    # Two passages the cross-encoder scores identically must fall back to the order they
    # came in, which is the fused rank. Near-duplicate chunks hit this constantly.
    cands = [{"id": "first"}, {"id": "second"}]
    ranked = order_by_scores(cands, [1.0, 1.0], top_k=2)
    assert [r["id"] for r in ranked] == ["first", "second"]


def test_top_k_caps_output():
    cands = [{"id": str(i)} for i in range(10)]
    ranked = order_by_scores(cands, list(range(10)), top_k=3)
    assert len(ranked) == 3


def test_length_mismatch_raises():
    try:
        order_by_scores([{"id": "a"}], [1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched candidate/score counts")


def test_rerank_drops_candidates_without_text():
    # "b" has no entry in texts. Scoring it on "" would give it a real length-0 score and a
    # place in the ranking. It should be dropped instead, so only a and c come back.
    cands = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    texts = {"a": "short", "c": "a much longer passage than a"}
    ranked = rerank("q", cands, texts, LengthScorer(), top_k=5)
    assert [r["id"] for r in ranked] == ["c", "a"]
    assert all("ce_score" in r for r in ranked)


def test_rerank_empty_when_no_text():
    ranked = rerank("q", [{"id": "x"}], {}, LengthScorer(), top_k=5)
    assert ranked == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as err:
            failed += 1
            print(f"  FAIL  {t.__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
