"""The eval harness. Runs each retrieval system over the golden set and records per-question
results, then prints the ablation table.

    python -m harness.run_eval run --systems dense,bm25,fused
    python -m harness.run_eval run --systems rerank --slice 0:3
    python -m harness.run_eval report

Why it is split into `run` and `report`, and why `run` is resumable.

The cross-encoder costs about 3.5 seconds a question on two CPU cores (measured day 4). Ten
questions of rerank alone is past the 45 second cap this sandbox puts on a single shell call.
Four systems in one process is nowhere near possible. So `run` appends one JSON line per
(system, question) to a results file and skips any pair already recorded. Kill it, rerun it,
run it in slices, the file converges on the same content either way. `report` then reads the
file and does the arithmetic. Same shape as the resumable index build from day 2, and for the
same reason.

The four systems are the ablation the project promised. Dense only. Sparse only. Hybrid. Hybrid
plus rerank. They are scored on identical questions with identical gold labels, so the columns
are comparable to each other. What they are not is a benchmark against anyone else's numbers.
Ten questions over one corpus.
"""

import argparse
import json
import re
import time
from pathlib import Path

from harness import metrics

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "evalset" / "golden.jsonl"
GOLD_CHUNKS = ROOT / "evalset" / "gold_chunks.jsonl"
CHUNKS = ROOT / "data" / "chunks.jsonl"
RESULTS = ROOT / "reports" / "results.jsonl"

SYSTEMS = ("dense", "bm25", "fused", "rerank")
KS = (1, 3, 5, 10)
DEPTH = 10          # how many hits each system returns, so recall@10 is measurable
POOL = 50           # fused candidate depth the reranker scores over, same as day 4

# Config keys are exact strings, so they make a cheap check that the retrieved context really
# carries the evidence. Same regex as evalset/validate.py uses for grounding.
CONFIG_KEY = re.compile(r"spark\.[a-zA-Z0-9._]+[a-zA-Z0-9]")


def load_jsonl(path):
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_questions():
    """Golden questions joined to their gold chunk labels. Fails loudly on a mismatch, because
    a silently unlabelled question would just look like a retrieval miss."""
    rows = {r["id"]: r for r in load_jsonl(GOLDEN)}
    gold = {r["id"]: r for r in load_jsonl(GOLD_CHUNKS)}
    missing = set(rows) - set(gold)
    if missing:
        raise SystemExit(f"no gold chunks for {sorted(missing)}. see docs/labelling.md")
    extra = set(gold) - set(rows)
    if extra:
        raise SystemExit(f"gold chunks for questions not in the golden set: {sorted(extra)}")
    out = []
    for qid, r in rows.items():
        r = dict(r)
        r["gold_chunks"] = gold[qid]["gold_chunks"]
        r["required"] = required_evidence(r)
        out.append(r)
    return out


def required_evidence(row):
    """The exact strings a correct answer has to be able to point at. Config keys named in the
    reference answer, plus any answer_spans the eval set already declares."""
    keys = sorted(set(CONFIG_KEY.findall(row["answer"])))
    return keys + list(row.get("answer_spans", []))


def load_texts():
    return {c["id"]: c["text"] for c in load_jsonl(CHUNKS)}


class Retrievers:
    """Loads the heavy objects once and hands out one callable per system.

    The cross-encoder and bge model are the expensive part, so a system that does not need one
    never triggers the load. Running `--systems bm25` should not pay for torch.
    """

    def __init__(self):
        self._dense = None
        self._bm25 = None
        self._ce = None

    def dense(self):
        if self._dense is None:
            from retrieval import dense as dense_mod
            index, meta = dense_mod._load_index()
            self._dense = (dense_mod, index, meta, dense_mod.get_model())
        return self._dense

    def bm25(self):
        if self._bm25 is None:
            from retrieval import sparse as sparse_mod
            self._bm25 = sparse_mod.load()
        return self._bm25

    def cross_encoder(self):
        if self._ce is None:
            from retrieval import rerank as rerank_mod
            self._ce = rerank_mod.get_model()
        return self._ce


WARMUP_QUERY = "warm up the kernels, this text is never scored"


def prime(systems, rv):
    """Load every heavy object the run will need and push one query through it, before any
    timer starts.

    Constructing the model is not enough, and that cost two separate measurements to learn.
    Day 4 timed a bge model load inside the first question and read 12049 ms against 19 ms
    for the other nine. Loading moved out and the column looked right. Then the day-6 rebuild
    on a newer torch read 1606 ms for the first dense question and 19 ms for the rest, with
    the model already loaded. Torch allocates buffers and picks kernels on the first forward
    pass, not at construction. So the fix is a real forward pass, not a constructor call.

    A mean over ten questions hides this badly. One 1.6 second outlier drags a 19 ms system
    to 178 ms and the table then reports a number no query ever took.
    """
    if {"dense", "fused", "rerank"} & set(systems):
        dense_mod, index, meta, model = rv.dense()
        dense_mod.search(WARMUP_QUERY, k=1, model=model, index=index, meta=meta)
    if {"bm25", "fused", "rerank"} & set(systems):
        rv.bm25().search(WARMUP_QUERY, k=1)
    if "rerank" in systems:
        rv.cross_encoder().predict([(WARMUP_QUERY, "a passage to warm the cross encoder")])


def run_one(system, query, rv, texts):
    """Retrieve for one question with one system. Returns (ranked ids, seconds).

    Timing covers retrieval only, and only because prime() has already loaded the models and
    indexes. Query cost and setup cost are different numbers and mixing them is how a latency
    table ends up lying.
    """
    from retrieval import fuse as fuse_mod
    from retrieval import rerank as rerank_mod

    t = time.perf_counter()
    if system == "dense":
        dense_mod, index, meta, model = rv.dense()
        hits = dense_mod.search(query, k=DEPTH, model=model, index=index, meta=meta)
    elif system == "bm25":
        hits = rv.bm25().search(query, k=DEPTH)
    elif system == "fused":
        dense_mod, index, meta, model = rv.dense()
        hits = fuse_mod.search(query, k=DEPTH, dense_index=index, dense_meta=meta,
                               dense_model=model, bm25=rv.bm25())
    elif system == "rerank":
        dense_mod, index, meta, model = rv.dense()
        pool, _ = rerank_mod._candidate_pool(query, POOL, index, meta, model, rv.bm25())
        hits = rerank_mod.rerank(query, pool, texts, rv.cross_encoder(), top_k=DEPTH)
    else:
        raise ValueError(f"unknown system {system!r}")
    elapsed = time.perf_counter() - t
    return [h["id"] for h in hits], elapsed


def already_done(results):
    if not results.exists():
        return set()
    return {(r["system"], r["qid"]) for r in load_jsonl(results)}


def append_result(row, results):
    results.parent.mkdir(parents=True, exist_ok=True)
    with results.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run(systems, qslice=None, budget=38.0, results=RESULTS):
    """Retrieve and record. Stops cleanly when the time budget is nearly spent so a partial
    run leaves a valid results file instead of a truncated line."""
    questions = load_questions()
    if qslice:
        start, end = qslice
        questions = questions[start:end]

    texts = load_texts()
    rv = Retrievers()
    prime(systems, rv)
    done = already_done(results)
    started = time.perf_counter()
    written = 0

    for system in systems:
        for q in questions:
            if (system, q["id"]) in done:
                continue
            if time.perf_counter() - started > budget:
                print(f"  stopping at the {budget:.0f}s budget, {written} new rows written. "
                      f"rerun to continue.", flush=True)
                return written
            ranked, elapsed = run_one(system, q["question"], rv, texts)
            row = {
                "system": system,
                "qid": q["id"],
                "ranked": ranked,
                "gold": q["gold_chunks"],
                "seconds": elapsed,
                "support": metrics.evidence_support(
                    [texts[i] for i in ranked[:5] if i in texts], q["required"]),
                "required": q["required"],
            }
            append_result(row, results)
            written += 1
            rr = metrics.reciprocal_rank(ranked, q["gold_chunks"])
            print(f"  {system:7} {q['id']}  rr {rr:.3f}  "
                  f"hit@5 {int(metrics.hit_at_k(ranked, q['gold_chunks'], 5))}  "
                  f"{elapsed * 1000:.0f} ms", flush=True)
    print(f"  {written} new rows", flush=True)
    return written


def summarise(rows):
    by_system = {}
    for r in rows:
        by_system.setdefault(r["system"], []).append(r)

    out = {}
    for system, rs in by_system.items():
        stats = {"n": len(rs)}
        for k in KS:
            stats[f"recall@{k}"] = metrics.mean(
                metrics.recall_at_k(r["ranked"], r["gold"], k) for r in rs)
            stats[f"hit@{k}"] = metrics.mean(
                float(metrics.hit_at_k(r["ranked"], r["gold"], k)) for r in rs)
        stats["mrr@10"] = metrics.mean(
            metrics.reciprocal_rank(r["ranked"], r["gold"], k=10) for r in rs)
        stats["support@5"] = metrics.mean(r["support"] for r in rs)
        stats["ms"] = metrics.mean(r["seconds"] for r in rs) * 1000
        out[system] = stats
    return out


def report(results=RESULTS):
    rows = load_jsonl(results)
    stats = summarise(rows)
    n_q = len(load_questions())

    cols = ["recall@1", "recall@5", "recall@10", "hit@1", "hit@5", "mrr@10", "support@5", "ms"]
    print(f"| system | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for system in SYSTEMS:
        if system not in stats:
            continue
        s = stats[system]
        cells = [f"{s[c]:.0f}" if c == "ms" else f"{s[c]:.3f}" for c in cols]
        flag = "" if s["n"] == n_q else "  (partial)"
        print(f"| {system} | {s['n']}{flag} | " + " | ".join(cells) + " |")

    print("\nper-question reciprocal rank at 10:")
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["system"]] = metrics.reciprocal_rank(
            r["ranked"], r["gold"], k=10)
    present = [s for s in SYSTEMS if s in stats]
    print("  qid    " + "  ".join(f"{s:>7}" for s in present))
    for qid in sorted(by_q):
        cells = "  ".join(f"{by_q[qid].get(s, float('nan')):7.3f}" for s in present)
        print(f"  {qid}  {cells}")


def parse_slice(text):
    if not text:
        return None
    start, _, end = text.partition(":")
    return int(start or 0), int(end) if end else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--systems", default=",".join(SYSTEMS))
    rp.add_argument("--slice", default="", help="question range, e.g. 0:3")
    rp.add_argument("--budget", type=float, default=38.0)
    # CI must not resume from the results file committed in the repo. If it did, every pair
    # would already be present, the run would write nothing and the gate would pass without
    # measuring anything. So CI points --out at its own file.
    rp.add_argument("--out", default=str(RESULTS), help="results file to append to")
    pp = sub.add_parser("report")
    pp.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args()

    if args.cmd == "run":
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
        unknown = set(systems) - set(SYSTEMS)
        if unknown:
            raise SystemExit(f"unknown systems {sorted(unknown)}, pick from {list(SYSTEMS)}")
        run(systems, parse_slice(args.slice), budget=args.budget, results=Path(args.out))
    elif args.cmd == "report":
        report(Path(args.results))


if __name__ == "__main__":
    main()
