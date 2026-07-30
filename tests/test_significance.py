"""Tests for harness/significance.py.

The permutation test is the part worth testing hard. It is the thing that decides whether a
component stays in the pipeline, and a subtly wrong p-value is worse than none because it
looks authoritative. Two of the cases below are there because they fail against the obvious
wrong implementations rather than because they describe happy paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import significance as sig  # noqa: E402


class approx:
    """Tiny stand-in for pytest.approx.

    The suites in this repo are plain scripts with their own runner, which is what the CI
    job loops over. Importing pytest here made `python -m tests.test_significance` die on a
    missing module, so CI failed on the import rather than on a test.
    """

    def __init__(self, expected, tol=1e-9):
        self.expected = expected
        self.tol = tol

    def __eq__(self, other):
        return abs(other - self.expected) <= self.tol

    def __repr__(self):
        return f"approx({self.expected})"


def test_all_same_sign_gives_the_smallest_possible_p():
    # Every difference points the same way, which is the most extreme arrangement available.
    # Only two of the 1024 sign assignments reach it: the observed one and its mirror.
    diffs = [0.1] * 10
    assert sig.permutation_p(diffs) == approx(2 / 1024)


def test_a_gap_driven_by_one_question_cannot_be_distinguished_from_noise():
    # This is the recall@1 case in the real table, and the reason the module exists.
    # Nine questions tie and one moves. Flipping the sign of a zero changes nothing, so all
    # 1024 assignments are as extreme as the observed one and p is exactly 1.
    diffs = [0.0] * 9 + [0.5]
    assert sig.permutation_p(diffs) == 1.0


def test_p_is_unchanged_when_the_comparison_is_reversed():
    # a minus b and b minus a are the same evidence. A one-sided implementation fails here.
    diffs = [0.3, -0.1, 0.2, 0.4, 0.0, 0.1, 0.2, -0.2, 0.3, 0.1]
    assert sig.permutation_p(diffs) == sig.permutation_p([-d for d in diffs])


def test_ties_are_counted_as_extreme():
    # The observed assignment is one of the 1024 and has to be counted, so no exact p on ten
    # questions can be below 2/1024. An implementation using strict > instead of >= returns
    # 0.0 here and reports a significant result for a two-question set.
    assert sig.permutation_p([0.2, 0.2]) == approx(2 / 4)


def test_too_many_questions_returns_none_rather_than_a_different_method():
    assert sig.permutation_p([0.1] * (sig.EXACT_LIMIT + 1)) is None


def test_empty_input_has_no_p_value():
    assert sig.permutation_p([]) is None


def test_bootstrap_is_reproducible_for_a_seed():
    diffs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.1, 0.0, -0.3]
    assert sig.bootstrap_ci(diffs, seed=7) == sig.bootstrap_ci(diffs, seed=7)


def test_the_interval_does_not_depend_on_the_seed_at_reported_precision():
    # The first version of this test asserted that two seeds give different intervals. They
    # do not, and finding that out was worth more than the test. Resampling ten discrete
    # values lands the 2.5th and 97.5th percentiles on a handful of possible endpoints, so
    # the interval is quantized. Measured: 40 seeds produce 8 distinct intervals spanning
    # about 0.01, which is well under any effect size being argued about. The seed does
    # change the draws, it just cannot move the endpoints far.
    diffs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.1, 0.0, -0.3]
    intervals = [sig.bootstrap_ci(diffs, seed=s) for s in range(20)]
    lows = [lo for lo, _ in intervals]
    highs = [hi for _, hi in intervals]
    assert max(lows) - min(lows) < 0.02
    assert max(highs) - min(highs) < 0.02


def test_bootstrap_on_a_constant_difference_has_no_width():
    lo, hi = sig.bootstrap_ci([0.25] * 10, reps=500, seed=1)
    assert lo == approx(0.25)
    assert hi == approx(0.25)


def _rows():
    return [
        {"system": "a", "qid": "q1", "ranked": ["c1", "c2"], "gold": ["c1"],
         "seconds": 0.010, "support": 1.0},
        {"system": "a", "qid": "q2", "ranked": ["c9", "c3"], "gold": ["c3"],
         "seconds": 0.010, "support": 0.0},
        {"system": "b", "qid": "q1", "ranked": ["c2", "c1"], "gold": ["c1"],
         "seconds": 0.020, "support": 1.0},
        {"system": "b", "qid": "q2", "ranked": ["c3", "c9"], "gold": ["c3"],
         "seconds": 0.020, "support": 0.0},
    ]


def test_per_question_recomputes_metrics_from_the_ranking():
    scores = sig.per_question(_rows())
    # a has the gold chunk first on q1 and second on q2. b is the other way round.
    assert scores["a"]["q1"]["mrr@10"] == 1.0
    assert scores["a"]["q2"]["mrr@10"] == 0.5
    assert scores["b"]["q1"]["mrr@10"] == 0.5
    assert scores["b"]["q2"]["mrr@10"] == 1.0


def test_pairing_is_by_qid_not_by_position():
    # Same question set, and the two systems are exact mirrors of each other. A paired
    # comparison sees +0.5 and -0.5 and returns a mean of zero. An implementation that
    # zipped the two lists in file order would get the same answer here by luck, so the
    # ordering assertion below is the one that matters.
    scores = sig.per_question(_rows())
    qids, diffs = sig.paired(scores, "a", "b", "mrr@10")
    assert qids == ["q1", "q2"]
    assert diffs == [0.5, -0.5]


def test_a_question_only_one_system_answered_is_dropped():
    rows = _rows()
    rows.append({"system": "a", "qid": "q3", "ranked": ["c1"], "gold": ["c1"],
                 "seconds": 0.01, "support": 1.0})
    scores = sig.per_question(rows)
    qids, diffs = sig.paired(scores, "a", "b", "mrr@10")
    assert qids == ["q1", "q2"]
    assert len(diffs) == 2


def test_compare_reports_how_many_questions_actually_moved():
    scores = sig.per_question(_rows())
    result = sig.compare(scores, "a", "b", "mrr@10")
    assert result["questions_moved"] == 2
    assert result["n"] == 2
    assert result["diff"] == approx(0.0)


def test_latency_separates_where_the_quality_metrics_do_not():
    # The positive control. If this ever stops firing, the module is broken rather than the
    # systems being identical.
    scores = sig.per_question(_rows())
    result = sig.compare(scores, "a", "b", "ms")
    assert result["diff"] == approx(-10.0)
    assert not result["crosses_zero"]


def test_the_p_value_floor_is_set_by_how_many_questions_differ():
    # Ties are invariant under a sign flip, so they drop out of the permutation space.
    # One question moving leaves 2 assignments and a floor of 1.0. Four leaves 16 and a
    # floor of 0.125, which is above 0.05, so that comparison is unwinnable before any data
    # is collected.
    assert sig.p_floor([0.0] * 9 + [0.5]) == 1.0
    assert sig.p_floor([0.0] * 6 + [0.2] * 4) == approx(0.125)
    assert sig.p_floor([0.1] * 10) == approx(2 / 1024)


def test_the_floor_is_actually_attained_when_every_difference_agrees():
    # The floor is not a loose bound. When the nonzero differences all point the same way
    # the observed p equals it exactly.
    for k in (1, 2, 4, 8):
        diffs = [0.0] * (10 - k) + [0.3] * k
        assert sig.permutation_p(diffs) == approx(sig.p_floor(diffs))


def test_no_observed_p_can_fall_below_its_floor():
    diffs = [0.4, -0.1, 0.0, 0.2, 0.3, 0.0, 0.1, 0.5, -0.2, 0.0]
    assert sig.permutation_p(diffs) >= sig.p_floor(diffs) - 1e-12


def _table_for(a_scores, b_scores, metric="m"):
    import contextlib
    import io
    scores = {"a": a_scores, "b": b_scores}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sig.print_table([sig.compare(scores, "a", "b", metric)], "a", "b")
    return buf.getvalue()


def test_an_interval_is_not_allowed_to_decide_an_underpowered_metric():
    # Four questions move, all the same way, so the bootstrap interval happily excludes zero
    # while the permutation floor is 0.125. Reporting that as "excludes zero" next to
    # "cannot reach p<.05" lets a reader pick whichever half they prefer.
    a = {f"q{i:03d}": {"m": 1.0 if i <= 4 else 0.0} for i in range(1, 11)}
    b = {f"q{i:03d}": {"m": 0.0} for i in range(1, 11)}
    out = _table_for(a, b)
    assert "too few questions moved to believe it" in out
    assert "interval excludes zero on: m" not in out


def test_a_fully_powered_metric_still_gets_to_decide():
    # All ten move, floor is 2/1024, so this one is allowed to conclude something.
    a = {f"q{i:03d}": {"m": 20.0} for i in range(1, 11)}
    b = {f"q{i:03d}": {"m": 1.0} for i in range(1, 11)}
    out = _table_for(a, b)
    assert "interval excludes zero on: m" in out
    assert "too few questions moved" not in out


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
