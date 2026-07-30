"""Tests for the prose-claim report in evalset/validate.py.

Two of these exist to fail against the implementations I nearly wrote. Splitting on
sentences only, and scoring against the whole document rather than one paragraph. Both
would have produced a report that says everything is fine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evalset import validate  # noqa: E402


def test_a_claim_joined_to_a_config_key_by_a_semicolon_is_still_seen():
    # This is q005's exact shape. Split on sentence enders alone and the claim rides along
    # with the config key, the clause looks grounded and the defect stays invisible.
    answer = ("No. Broadcast requires the smaller relation to fit in driver and executor "
              "memory; spark.sql.autoBroadcastJoinThreshold controls the cutoff.")
    claims = validate.claim_clauses(answer)
    assert len(claims) == 1
    assert claims[0].startswith("Broadcast requires the smaller relation")


def test_clauses_naming_a_config_key_are_left_to_the_grounding_check():
    answer = "Set spark.serializer to org.apache.spark.serializer.KryoSerializer."
    assert validate.claim_clauses(answer) == []


def test_a_bare_yes_or_no_is_not_a_claim():
    assert validate.claim_clauses("No. Yes. Maybe.") == []


def test_coverage_is_against_one_paragraph_not_the_whole_document():
    # The failure being hunted is a claim assembled from words scattered across a long
    # reference page. Every word of the claim appears somewhere in `scattered`, so an
    # implementation that scores against the concatenated document returns 1.0 and reports
    # the claim as grounded. The first version of this test used paragraphs that did not
    # cover the claim between them, so it passed against that broken implementation and
    # tested nothing.
    claim = "broadcast requires driver memory"
    scattered = ["a broadcast join requires size limits", "the driver holds memory for tasks"]
    together = ["a broadcast join requires driver memory to hold the relation"]
    cover_scattered, _ = validate.best_paragraph(claim, scattered)
    cover_together, _ = validate.best_paragraph(claim, together)
    assert cover_together == 1.0
    assert cover_scattered == 0.5


def test_the_matching_paragraph_comes_back_with_the_score():
    claim = "watermark bounds late data"
    paras = ["unrelated text about joins", "the watermark bounds how late data may arrive"]
    cover, para = validate.best_paragraph(claim, paras)
    assert cover == 1.0
    assert para.startswith("the watermark")


def test_a_claim_with_no_content_words_scores_nothing():
    cover, para = validate.best_paragraph("it is to be", ["anything at all"])
    assert cover == 0.0
    assert para == ""


def test_the_report_is_ordered_weakest_support_first():
    rows = [
        (1, {"id": "qA", "answer": "Executors run tasks inside the cluster workers.",
             "source_docs": ["a.md"]}),
        (2, {"id": "qB", "answer": "Quantum entanglement governs partition placement.",
             "source_docs": ["a.md"]}),
    ]

    class FakeDir:
        def is_dir(self):
            return True

        def __truediv__(self, name):
            return FakePath()

    class FakePath:
        def exists(self):
            return True

        def read_text(self, **kwargs):
            return "Executors run tasks inside the cluster workers.\n\nNothing else here.\n"

    report = validate.claim_report(rows, corpus_dir=FakeDir())
    assert [r[1] for r in report] == ["qB", "qA"]
    assert report[0][0] < report[1][0]


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
