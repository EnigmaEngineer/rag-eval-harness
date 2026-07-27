"""Cross-encoder reranking over the fused candidate pool.

    python -m retrieval.rerank search "coalesce small partitions after a shuffle" --k 5
    python -m retrieval.rerank smoke     # fused top-5 vs reranked top-5, side by side

Why a cross-encoder here and nowhere else. The dense retriever is a bi-encoder: query and
passage are embedded separately into two vectors and compared by cosine. That is cheap and
the passage vectors are precomputed once, which is exactly what lets it scan 3228 chunks in
milliseconds. The price is that the query never actually sees the passage. A cross-encoder
does the opposite. It concatenates query and passage and runs one transformer forward over
the pair, so every query term can attend to every passage term. Far more accurate at telling
a real answer from a near-miss, and far too slow to run over the whole corpus. One forward
per pair, nothing cacheable.

So it runs last, over the short fused pool, not the corpus. Retrieve wide and cheap with
dense + BM25, then rerank narrow and expensive. That is the whole retrieve-then-rerank
shape and the reason the day-3 fusion regression might be recoverable: q002's correct chunk
is in the fused pool at rank 7, and the reranker gets to look at the text, not just the
rank.

The pool comes in as candidate hits carrying an `id`. The cross-encoder needs the passage
*text*, which the retriever meta deliberately does not carry (it is just ids and doc names).
So load_texts() reads it back from data/chunks.jsonl keyed by id.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_model():
    import torch
    # same two-core pin as the dense path. left to itself torch grabs one thread and the
    # per-pair forward roughly doubles.
    torch.set_num_threads(2)
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL)


def load_texts():
    """id -> chunk text. The reranker scores query against the passage body, and neither
    index stores the body, so read it back from the chunk file."""
    if not CHUNKS.exists():
        raise SystemExit("no chunks. run: python -m ingest.chunk")
    texts = {}
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        texts[c["id"]] = c["text"]
    return texts


def order_by_scores(candidates, scores, top_k=5):
    """Attach each score to its candidate and sort best-first. Pure, so the ordering is
    tested without loading a 90 MB model.

    candidates and scores are parallel lists. A stable sort on the negated score keeps the
    upstream fused order as the tiebreak when the cross-encoder ties two passages, which it
    does for near-duplicate chunks. Returns copies with a `ce_score` field, capped at top_k.
    """
    if len(candidates) != len(scores):
        raise ValueError(f"{len(candidates)} candidates but {len(scores)} scores")
    paired = []
    for cand, score in zip(candidates, scores):
        row = dict(cand)
        row["ce_score"] = float(score)
        paired.append(row)
    paired.sort(key=lambda r: r["ce_score"], reverse=True)
    return paired[:top_k]


def rerank(query, candidates, texts, model, top_k=5):
    """Score every candidate against the query with the cross-encoder and reorder.

    candidates is the fused pool (hits with an `id`). texts maps id -> passage body. A
    candidate whose id is missing from texts is dropped rather than scored on an empty
    string, because a blank passage scores like noise and would just add a random reorder.
    """
    usable = [c for c in candidates if c["id"] in texts]
    if not usable:
        return []
    pairs = [(query, texts[c["id"]]) for c in usable]
    scores = model.predict(pairs)
    return order_by_scores(usable, scores, top_k=top_k)


def _candidate_pool(query, pool, dense_index, dense_meta, dense_model, bm25):
    """dense + BM25 to `pool` depth, fused with RRF into one pool best-first. Returns
    (pool_hits, timings) where timings holds the dense, bm25 and fuse stage seconds. Split
    out so search() and smoke() share one retrieval, and smoke can score the fused pool and
    the reranked pool against the same candidates instead of retrieving twice."""
    from retrieval import dense as dense_mod
    from retrieval import fuse as fuse_mod

    timings = {}
    t = time.perf_counter()
    dense_hits = dense_mod.search(query, k=pool, model=dense_model,
                                  index=dense_index, meta=dense_meta)
    timings["dense"] = time.perf_counter() - t

    t = time.perf_counter()
    bm25_hits = bm25.search(query, k=pool)
    timings["bm25"] = time.perf_counter() - t

    t = time.perf_counter()
    by_id = {h["id"]: h for h in bm25_hits}
    by_id.update({h["id"]: h for h in dense_hits})
    fused = fuse_mod.fuse([[h["id"] for h in dense_hits], [h["id"] for h in bm25_hits]],
                          top_k=pool)
    pool_hits = [dict(by_id[_id], rrf_score=s) for _id, s in fused]
    timings["fuse"] = time.perf_counter() - t
    return pool_hits, timings


def search(query, k=5, pool=50, texts=None, model=None,
           dense_index=None, dense_meta=None, dense_model=None, bm25=None):
    """Full pipeline for one query: dense + BM25 -> fuse to a pool -> cross-encoder rerank.

    Returns (hits, timings). timings is seconds per stage, measured with perf_counter, so
    the caller can report where the query budget actually goes. The heavy objects load once
    when passed in, which is how smoke() avoids paying the model load per question.
    """
    from retrieval import dense as dense_mod
    from retrieval import sparse as sparse_mod

    if dense_index is None:
        dense_index, dense_meta = dense_mod._load_index()
    if dense_model is None:
        dense_model = dense_mod.get_model()
    if bm25 is None:
        bm25 = sparse_mod.load()
    if texts is None:
        texts = load_texts()
    if model is None:
        model = get_model()

    pool_hits, timings = _candidate_pool(query, pool, dense_index, dense_meta,
                                         dense_model, bm25)
    t = time.perf_counter()
    reranked = rerank(query, pool_hits, texts, model, top_k=k)
    timings["rerank"] = time.perf_counter() - t
    return reranked, timings


def smoke(pool=50):
    """For each golden question, print fused top-5 vs reranked top-5 doc-hit side by side,
    and a mean per-stage latency line. Same doc-overlap proxy the other smokes use. Real
    recall@k and MRR are day 5. What this shows today is whether the reranker moves the
    fused result and whether it recovers q002, the chunk day 3 lost to RRF consensus bias.

    Fused top-5 and reranked top-5 are scored over the *same* fused pool, so any difference
    is the reranker's doing and not a different candidate set. On 2 CPU cores the cross
    encoder runs ~3.5s a question, so the full ten-question pass is ~40s. Prints flush per
    question so progress shows even when it is slow. Syed's laptop has no per-call cap."""
    from retrieval import dense as dense_mod
    from retrieval import sparse as sparse_mod

    golden = ROOT / "evalset" / "golden.jsonl"
    rows = [json.loads(l) for l in golden.read_text(encoding="utf-8").splitlines() if l.strip()]

    dense_index, dense_meta = dense_mod._load_index()
    dense_model = dense_mod.get_model()
    bm25 = sparse_mod.load()
    texts = load_texts()
    model = get_model()

    k = 5
    tallies = {"fused": 0, "rerank": 0}
    stage_totals = {"dense": 0.0, "bm25": 0.0, "fuse": 0.0, "rerank": 0.0}
    for r in rows:
        want = set(r["source_docs"])
        pool_hits, timings = _candidate_pool(r["question"], pool, dense_index,
                                             dense_meta, dense_model, bm25)
        fused_top = pool_hits[:k]
        t = time.perf_counter()
        reranked = rerank(r["question"], pool_hits, texts, model, top_k=k)
        timings["rerank"] = time.perf_counter() - t

        f_ok = bool(want & {h["doc"] for h in fused_top})
        rr_ok = bool(want & {h["doc"] for h in reranked})
        tallies["fused"] += f_ok
        tallies["rerank"] += rr_ok
        for stage, dt in timings.items():
            stage_totals[stage] += dt

        def m(ok):
            return "hit " if ok else "MISS"
        print(f"  {r['id']}  fused {m(f_ok)}  rerank {m(rr_ok)}   want {r['source_docs']}",
              flush=True)

    n = len(rows)
    print(f"\n  fused {tallies['fused']}/{n}   rerank {tallies['rerank']}/{n}   "
          f"(expected source doc in top-{k})")
    print("  mean per-stage ms: " + "  ".join(
        f"{s} {stage_totals[s] / n * 1000:.0f}" for s in ("dense", "bm25", "fuse", "rerank")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--pool", type=int, default=50)
    sub.add_parser("smoke")
    args = ap.parse_args()

    if args.cmd == "search":
        hits, timings = search(args.query, k=args.k, pool=args.pool)
        for h in hits:
            print(f"  {h['ce_score']:8.3f}  {h['doc']:42} {h['heading_path'][:55]}")
        print("  stage ms: " + "  ".join(f"{s} {timings[s] * 1000:.0f}" for s in timings))
    elif args.cmd == "smoke":
        smoke()


if __name__ == "__main__":
    main()
