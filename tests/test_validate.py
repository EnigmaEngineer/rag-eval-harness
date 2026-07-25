"""Negative tests for the golden-set validator.

A validator that has never rejected anything is not a validator. These exist so the
grounding check keeps working when it gets extended.

    python -m tests.test_validate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evalset.validate import check_answers_grounded, validate, load  # noqa: E402


def q(**over):
    base = {
        "id": "qTEST", "question": "x", "answer": "y",
        "source_docs": ["tuning.md"], "difficulty": "easy",
    }
    base.update(over)
    return [(1, base)]


def test_config_key_in_wrong_doc():
    rows = q(answer="Set spark.sql.adaptive.skewJoin.enabled to true.", source_docs=["monitoring.md"])
    problems = check_answers_grounded(rows)
    assert problems, "should reject a config key absent from the cited doc"
    assert "skewJoin" in problems[0]


def test_config_key_in_right_doc():
    rows = q(answer="Set spark.sql.adaptive.skewJoin.enabled to true.",
             source_docs=["sql-performance-tuning.md"])
    assert not check_answers_grounded(rows)


def test_missing_answer_span():
    rows = q(answer_spans=["quantum tuning mode"])
    problems = check_answers_grounded(rows)
    assert problems and "quantum tuning mode" in problems[0]


def test_duplicate_ids_rejected():
    rows = [(1, q()[0][1]), (2, q()[0][1])]
    problems, _ = validate(rows)
    assert any("duplicate id" in p for p in problems)


def test_bad_difficulty_rejected():
    problems, _ = validate(q(difficulty="trivial"))
    assert any("difficulty" in p for p in problems)


def test_real_golden_set_is_clean():
    rows = load()
    schema_problems, _ = validate(rows)
    assert not schema_problems, schema_problems
    assert not check_answers_grounded(rows)


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
