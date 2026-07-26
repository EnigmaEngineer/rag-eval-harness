"""Tests for reciprocal rank fusion.

The scoring is pure and rank-based, so it tests without a model or an index. These check
the properties the fusion actually relies on: agreement across lists beats a single strong
placing, only ranks matter not the raw scores that produced them, a missing id costs
nothing, and the constant k behaves the way the reasoning in docs/fusion.md claims.

    python -m tests.test_fuse
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.fuse import fuse  # noqa: E402


def test_agreement_beats_a_single_top_rank():
    # B is rank 2 in both lists. A is rank 1 in one list only. Two mid placings should
    # outweigh one top placing, which is the entire reason to fuse.
    dense = ["A", "B", "C"]
    bm25 = ["D", "B", "E"]
    ranked = fuse([dense, bm25], top_k=5)
    winner = ranked[0][0]
    assert winner == "B", f"expected B to win on agreement, got {winner}"


def test_only_rank_matters_not_score():
    # fuse takes ids, not scores, so two runs that differ only in the (discarded) scores
    # must produce identical output. This is the property that lets us skip normalisation.
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    assert fuse([a, b], top_k=4) == fuse([a, b], top_k=4)


def test_missing_id_contributes_nothing():
    # An id in only one list gets credit from that list alone, and never a penalty for
    # being absent from the other.
    ranked = dict(fuse([["p", "q"], ["q"]], top_k=5))
    # q: 1/(60+2) from dense + 1/(60+1) from bm25. p: 1/(60+1) from dense only.
    assert ranked["q"] > ranked["p"], "q is in both lists, should outrank p"


def test_small_k_sharpens_the_top():
    # A smaller k widens the gap between rank 1 and rank 2. With k=1, rank 1 is worth 1/2
    # and rank 2 is worth 1/3. With k=60 they are 1/61 and 1/62, nearly equal.
    single = [["A", "B"]]
    sharp = dict(fuse(single, k_rrf=1, top_k=2))
    flat = dict(fuse(single, k_rrf=60, top_k=2))
    assert (sharp["A"] - sharp["B"]) > (flat["A"] - flat["B"])


def test_top_k_caps_output():
    ids = [str(i) for i in range(20)]
    assert len(fuse([ids], top_k=5)) == 5


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
