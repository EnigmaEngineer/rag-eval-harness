"""Tests for the BM25 retriever.

These build a tiny in-memory index from a handful of fake chunks, so they run with no
corpus and no model. They check the parts that would silently degrade retrieval if wrong:
the tokenizer splits dotted config keys, term frequency saturates, and a rarer term
outweighs a common one.

    python -m tests.test_sparse
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.sparse import BM25, tokenize  # noqa: E402


def make_index(texts):
    # Mirror sparse.build() without touching disk or the real corpus.
    postings = defaultdict(list)
    doc_len = [0] * len(texts)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        doc_len[i] = len(toks)
        for term, tf in Counter(toks).items():
            postings[term].append((i, tf))
    avgdl = sum(doc_len) / len(texts)
    meta = [{"id": f"c{i}", "doc": f"c{i}.md", "heading_path": "", "kind": "text",
             "n_tokens": doc_len[i]} for i in range(len(texts))]
    return BM25({"n": len(texts), "avgdl": avgdl, "doc_len": doc_len,
                 "postings": dict(postings), "meta": meta})


def test_tokenizer_splits_dotted_config_keys():
    # the reason BM25 can catch q002 at all: partitions has to be a matchable token even
    # though the doc only ever writes it inside spark.sql.shuffle.partitions.
    toks = tokenize("Set spark.sql.shuffle.partitions to 200")
    assert "partitions" in toks
    assert "shuffle" in toks
    assert "spark.sql.shuffle.partitions" not in toks


def test_tokenizer_drops_stopwords():
    assert tokenize("how do I set the value") == ["set", "value"]


def test_rarer_term_scores_higher():
    # "shuffle" appears in one doc, "spark" in all three. A query hitting both should rank
    # the doc matched on the rare term first.
    idx = make_index([
        "spark shuffle partitions tuning",
        "spark configuration reference",
        "spark memory and storage",
    ])
    hits = idx.search("spark shuffle", k=3)
    assert hits[0]["id"] == "c0", f"doc with the rare term should win, got {hits[0]['id']}"


def test_term_frequency_saturates():
    # BM25 tf is sublinear. Ten occurrences must not score ten times one occurrence, or a
    # keyword-stuffed chunk would beat a genuinely relevant one.
    idx = make_index(["alpha " * 10 + "filler " * 90, "alpha " + "filler " * 99])
    hits = {h["id"]: h["score"] for h in idx.search("alpha", k=2)}
    assert hits["c0"] < 10 * hits["c1"], "tf should saturate, not scale linearly"


def test_unknown_term_returns_nothing():
    idx = make_index(["spark shuffle partitions"])
    assert idx.search("kubernetes helm chart", k=5) == []


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
