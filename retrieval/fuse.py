"""Hybrid retrieval: fuse the dense and BM25 rankings with reciprocal rank fusion.

    python -m retrieval.fuse search "coalesce small partitions after a shuffle" --k 5
    python -m retrieval.fuse smoke        # dense vs bm25 vs fused, side by side

Reciprocal rank fusion, not weighted score fusion. The reasoning is in docs/fusion.md and
matters enough to summarise here: dense cosine and BM25 scores live on different, unbounded
scales, so combining the raw numbers means normalising them first, and every normalisation
(min-max, z-score) is set by the batch it sees and shifts under you. RRF throws the scores
away and keeps only the ranks. A term appearing at rank 3 in either list is worth the same
no matter what its raw score was. That removes the one knob that would otherwise need tuning
against a metric this project does not have until day 5.

The RRF constant k damps the top ranks so a single list cannot dominate. 60 is the value
from the original Cormack et al. 2009 paper and the common default. It is exposed so the
day-5 harness can sweep it if the metrics say to.

`fuse` is pure and takes ranked id lists, so it is unit tested without a model. `search`
wires the two real retrievers into it.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

RRF_K = 60
# Fuse deeper than we return. A doc that is rank 30 in dense and rank 2 in BM25 should get
# its BM25 credit, so both candidate lists run deeper than the final k.
DEFAULT_DEPTH = 50


def fuse(ranked_lists, k_rrf=RRF_K, top_k=5):
    """Combine several ranked lists of ids into one. Each list is ids best-first.

    score(id) = sum over lists of 1 / (k_rrf + rank), rank starting at 1. Returns a list of
    (id, score) pairs best-first and capped at top_k. An id missing from a list contributes
    nothing from that list, which is the whole point. There is no score to normalise.
    """
    scores = defaultdict(float)
    for ids in ranked_lists:
        for rank, _id in enumerate(ids, start=1):
            scores[_id] += 1.0 / (k_rrf + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def search(query, k=5, depth=DEFAULT_DEPTH, dense_index=None, dense_meta=None,
           dense_model=None, bm25=None):
    """Retrieve with both indexes to `depth`, fuse, return the top k fused hits.

    The heavy objects (faiss index, bge model, bm25 index) can be passed in so a batch
    like smoke() loads them once instead of per query.
    """
    from retrieval import dense as dense_mod
    from retrieval import sparse as sparse_mod

    if dense_index is None:
        dense_index, dense_meta = dense_mod._load_index()
    if dense_model is None:
        dense_model = dense_mod.get_model()
    if bm25 is None:
        bm25 = sparse_mod.load()

    dense_hits = dense_mod.search(query, k=depth, model=dense_model,
                                  index=dense_index, meta=dense_meta)
    bm25_hits = bm25.search(query, k=depth)

    by_id = {h["id"]: h for h in bm25_hits}
    by_id.update({h["id"]: h for h in dense_hits})  # dense meta wins on overlap, same fields

    fused = fuse([[h["id"] for h in dense_hits], [h["id"] for h in bm25_hits]],
                 top_k=k)
    out = []
    for _id, score in fused:
        row = dict(by_id[_id])
        row["rrf_score"] = score
        out.append(row)
    return out


def smoke():
    """Run all three retrievers over the golden set and print a hit/miss line each, so the
    fusion result sits next to the two inputs it came from. Doc-level overlap at top-5, the
    same coarse proxy the other two use. Real recall@k is day 5."""
    from retrieval import dense as dense_mod
    from retrieval import sparse as sparse_mod

    golden = ROOT / "evalset" / "golden.jsonl"
    rows = [json.loads(l) for l in golden.read_text(encoding="utf-8").splitlines() if l.strip()]

    dense_index, dense_meta = dense_mod._load_index()
    dense_model = dense_mod.get_model()
    bm25 = sparse_mod.load()

    k = 5
    tallies = {"dense": 0, "bm25": 0, "fused": 0}
    for r in rows:
        want = set(r["source_docs"])
        d = dense_mod.search(r["question"], k=k, model=dense_model,
                             index=dense_index, meta=dense_meta)
        s = bm25.search(r["question"], k=k)
        f = search(r["question"], k=k, dense_index=dense_index, dense_meta=dense_meta,
                   dense_model=dense_model, bm25=bm25)

        d_ok = bool(want & {h["doc"] for h in d})
        s_ok = bool(want & {h["doc"] for h in s})
        f_ok = bool(want & {h["doc"] for h in f})
        tallies["dense"] += d_ok
        tallies["bm25"] += s_ok
        tallies["fused"] += f_ok

        def m(ok):
            return "hit " if ok else "MISS"
        print(f"  {r['id']}  dense {m(d_ok)}  bm25 {m(s_ok)}  fused {m(f_ok)}   "
              f"want {r['source_docs']}")

    n = len(rows)
    print(f"\n  dense {tallies['dense']}/{n}   bm25 {tallies['bm25']}/{n}   "
          f"fused {tallies['fused']}/{n}   (expected source doc in top-{k})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    sub.add_parser("smoke")
    args = ap.parse_args()

    if args.cmd == "search":
        for h in search(args.query, k=args.k, depth=args.depth):
            print(f"  {h['rrf_score']:.4f}  {h['doc']:42} {h['heading_path'][:55]}")
    elif args.cmd == "smoke":
        smoke()


if __name__ == "__main__":
    main()
