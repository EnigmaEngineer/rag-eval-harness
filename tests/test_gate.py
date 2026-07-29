"""Tests for the regression gate.

The gate is the one piece of this repo that is supposed to fail. A gate that cannot fail is
a green tick with nothing behind it, so these check that a real drop trips it, that a drop
inside tolerance does not, and that a partial run is refused rather than compared against a
full baseline.

    python -m tests.test_gate
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import gate  # noqa: E402


def baseline_stats(**over):
    s = {"n": 10, "recall@1": 0.5, "recall@5": 0.7, "recall@10": 0.8,
         "hit@1": 0.5, "hit@5": 0.8, "mrr@10": 0.6, "support@5": 0.9, "ms": 10.0}
    s.update(over)
    return s


def with_baseline(base, fn):
    """Point the gate at a throwaway baseline file, run fn, put the real path back."""
    real = gate.BASELINE
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        path.write_text(json.dumps(base))
        gate.BASELINE = path
        try:
            return fn()
        finally:
            gate.BASELINE = real


def run_check(base, now):
    saved = gate.current
    gate.current = lambda systems=None, results=None: now
    try:
        return with_baseline(base, lambda: gate.check())
    finally:
        gate.current = saved


def test_a_real_drop_fails():
    base = {"systems": {"bm25": baseline_stats()}, "questions": 10}
    now = {"bm25": baseline_stats(**{"recall@5": 0.55})}   # down 0.15, tolerance 0.05
    assert run_check(base, now) == 1


def test_a_drop_inside_tolerance_passes():
    base = {"systems": {"bm25": baseline_stats()}, "questions": 10}
    now = {"bm25": baseline_stats(**{"recall@5": 0.67})}   # down 0.03
    assert run_check(base, now) == 0


def test_an_improvement_passes():
    base = {"systems": {"bm25": baseline_stats()}, "questions": 10}
    now = {"bm25": baseline_stats(**{"recall@5": 0.9, "mrr@10": 0.75})}
    assert run_check(base, now) == 0


def test_a_partial_run_is_not_compared():
    # 6 questions against a baseline of 10 is a different question set, not a better score
    base = {"systems": {"bm25": baseline_stats()}, "questions": 10}
    now = {"bm25": baseline_stats(n=6, **{"recall@5": 1.0})}
    assert run_check(base, now) == 1


def test_latency_blowup_fails():
    base = {"systems": {"bm25": baseline_stats(ms=10.0)}, "questions": 10}
    now = {"bm25": baseline_stats(ms=40.0)}
    assert run_check(base, now) == 1


def test_tiny_latency_moves_do_not_fail():
    # 1ms to 3ms is 3x and means nothing. the absolute floor stops the gate crying about it
    base = {"systems": {"bm25": baseline_stats(ms=1.0)}, "questions": 10}
    now = {"bm25": baseline_stats(ms=3.0)}
    assert run_check(base, now) == 0


def test_a_system_missing_from_the_baseline_is_skipped():
    base = {"systems": {"bm25": baseline_stats()}, "questions": 10}
    now = {"bm25": baseline_stats(), "dense": baseline_stats(**{"recall@5": 0.0})}
    assert run_check(base, now) == 0, "dense has no baseline, so there is nothing to regress"


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
