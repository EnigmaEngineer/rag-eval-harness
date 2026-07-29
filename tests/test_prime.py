"""Tests that prime() actually warms every model the run will time.

This suite exists because the same bug landed twice. Day 4 timed a bge model load inside the
first question and read 12049 ms against 19 ms for the rest. Day 6 read 1606 ms against 19 ms
with the model already constructed, because torch does not allocate buffers or select kernels
until the first forward pass. Both times a single outlier dragged the reported mean to a
number no query took.

Day 5's audit accepted "no test covers this" as a risk on the grounds that any such test
would just restate prime(). That was wrong, and the recurrence proved it. The contract worth
testing is not how long a query takes. It is that every model prime() touches has had a real
query pushed through it before the harness starts a timer. Stubs record the calls, so this
runs in milliseconds and needs no model download.

    python -m tests.test_prime
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import run_eval  # noqa: E402


class FakeDenseModule:
    def __init__(self):
        self.searches = []

    def search(self, query, k=5, model=None, index=None, meta=None):
        self.searches.append(query)
        return []


class FakeBm25:
    def __init__(self):
        self.searches = []

    def search(self, query, k=5):
        self.searches.append(query)
        return []


class FakeCrossEncoder:
    def __init__(self):
        self.predictions = []

    def predict(self, pairs):
        self.predictions.append(pairs)
        return [0.0] * len(pairs)


class FakeRetrievers:
    def __init__(self):
        self.dense_mod = FakeDenseModule()
        self.bm25_obj = FakeBm25()
        self.ce = FakeCrossEncoder()

    def dense(self):
        return (self.dense_mod, "index", "meta", "model")

    def bm25(self):
        return self.bm25_obj

    def cross_encoder(self):
        return self.ce


def test_dense_gets_a_real_query_not_just_a_constructor():
    rv = FakeRetrievers()
    run_eval.prime(["dense"], rv)
    assert rv.dense_mod.searches, "prime built the dense model but never ran a query through it"


def test_bm25_is_warmed():
    rv = FakeRetrievers()
    run_eval.prime(["bm25"], rv)
    assert rv.bm25_obj.searches


def test_rerank_warms_all_three_stages():
    rv = FakeRetrievers()
    run_eval.prime(["rerank"], rv)
    assert rv.dense_mod.searches, "rerank runs dense to build its pool"
    assert rv.bm25_obj.searches, "rerank runs bm25 to build its pool"
    assert rv.ce.predictions, "the cross-encoder is the expensive stage and must be warmed"


def test_bm25_only_does_not_touch_the_dense_side():
    # running --systems bm25 should not pay for torch at all
    rv = FakeRetrievers()
    run_eval.prime(["bm25"], rv)
    assert not rv.dense_mod.searches
    assert not rv.ce.predictions


def test_the_warmup_query_is_never_scored():
    # if the warmup text ever reached a results row it would look like an eleventh question
    rv = FakeRetrievers()
    run_eval.prime(["fused"], rv)
    assert run_eval.WARMUP_QUERY in rv.dense_mod.searches
    assert run_eval.WARMUP_QUERY in rv.bm25_obj.searches
    qids = {q["id"] for q in run_eval.load_questions()}
    assert run_eval.WARMUP_QUERY not in qids


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
