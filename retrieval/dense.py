"""Dense retrieval over the chunked corpus with FAISS + bge-small-en-v1.5.

    python -m ingest.chunk                 # produce data/chunks.jsonl first
    python -m retrieval.dense build        # embed chunks, write indexes/
    python -m retrieval.dense search "how do I stop tiny tasks after a shuffle" --k 5
    python -m retrieval.dense smoke        # run the golden questions as a sanity check

Model choice (resolves the day-1 open thread): bge-small-en-v1.5. 384-dim, 512-token
context, retrieval-tuned, and small enough to embed the whole corpus on 2 CPUs. The 512
chunk budget follows from its context window rather than being picked first and hoping a
model fits it.

bge wants an instruction prefixed to the *query* only, not the passages. Skipping it on
queries measurably hurts recall, so the prefix lives here, once, where it cannot be
forgotten.
"""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "indexes"
INDEX_PATH = INDEX_DIR / "dense.faiss"
META_PATH = INDEX_DIR / "dense_meta.jsonl"

# The embedding memmap is written batch by batch during a resumable build. On a slow or
# network-mounted working directory that write dominates the runtime, so it can be pointed
# at fast local scratch with RAG_BUILD_SCRATCH. The finished index and meta still land in
# indexes/. Defaults to indexes/ so a normal machine needs no env var.
SCRATCH = Path(os.environ.get("RAG_BUILD_SCRATCH", INDEX_DIR))

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks():
    if not CHUNKS.exists():
        raise SystemExit("no chunks. run: python -m ingest.chunk")
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]


EMB_PATH = SCRATCH / "emb.f32.dat"
PROGRESS_PATH = SCRATCH / "build_progress.json"


def get_model():
    import torch
    # the sandbox gives us two cores; without this torch grabs one and CPU encoding
    # roughly halves. Harmless on a bigger box.
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


def build(max_seconds=35, batch_size=32):
    """Embed the chunks into a FAISS flat-IP index. Resumable on purpose.

    CPU encoding runs at ~9 chunks/s here and the corpus is ~3.2k chunks, which is far
    past the sandbox's per-call time limit. So embeddings are written to a memmap as they
    are produced and progress is checkpointed; each call does a slice and stops. Call it
    until it reports done, then it assembles the index. Returns True once complete.

    normalize + IndexFlatIP == exact cosine. Flat is the right call at a few thousand
    vectors; an approximate index would add a recall/latency knob to tune before there is
    a metric to tune it against.
    """
    import faiss
    import numpy as np

    chunks = load_chunks()
    n = len(chunks)
    INDEX_DIR.mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    model = get_model()
    dim = model.get_sentence_embedding_dimension()

    if PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text())
        if prog["n"] != n or prog["dim"] != dim:
            raise SystemExit("corpus or model changed; delete indexes/ and rebuild")
        done = prog["done"]
        emb = np.memmap(EMB_PATH, dtype="float32", mode="r+", shape=(n, dim))
    else:
        done = 0
        emb = np.memmap(EMB_PATH, dtype="float32", mode="w+", shape=(n, dim))
        prog = {"n": n, "dim": dim, "done": 0}

    t0 = time.time()
    start = done
    while done < n and time.time() - t0 < max_seconds:
        j = min(done + batch_size, n)
        vecs = model.encode([c["text"] for c in chunks[done:j]], batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
        emb[done:j] = np.asarray(vecs, dtype="float32")
        done = j
        prog["done"] = done
        PROGRESS_PATH.write_text(json.dumps(prog))
    emb.flush()

    rate = (done - start) / (time.time() - t0) if done > start else 0
    if done < n:
        print(f"progress {done}/{n} ({done * 100 // n}%) at {rate:.0f} chunks/s. "
              f"run 'build' again to continue.")
        return False

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(emb[:]))
    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({k: c[k] for k in ("id", "doc", "heading_path", "kind", "n_tokens")},
                                ensure_ascii=False) + "\n")
    print(f"index complete: {n} vectors, dim {dim} -> {INDEX_PATH.relative_to(ROOT)}")
    return True


def _load_index():
    import faiss
    if not INDEX_PATH.exists():
        raise SystemExit("no index. run: python -m retrieval.dense build")
    index = faiss.read_index(str(INDEX_PATH))
    meta = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return index, meta


def search(query, k=5, model=None, index=None, meta=None):
    import numpy as np
    if index is None:
        index, meta = _load_index()
    if model is None:
        model = get_model()
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype="float32"), k)
    hits = []
    for score, i in zip(scores[0], ids[0]):
        row = dict(meta[i])
        row["score"] = float(score)
        hits.append(row)
    return hits


def smoke():
    """Not the day-5 metric suite. A sanity check: for each golden question, does an
    expected source_doc show up in the top-k? Prints a hit/miss per question and a count.
    Anything below full marks here is a real signal worth chasing before building metrics
    on top of it."""
    golden = ROOT / "evalset" / "golden.jsonl"
    rows = [json.loads(l) for l in golden.read_text(encoding="utf-8").splitlines() if l.strip()]
    model = get_model()
    index, meta = _load_index()

    k = 5
    hit = 0
    for r in rows:
        got = search(r["question"], k=k, model=model, index=index, meta=meta)
        docs = [h["doc"] for h in got]
        ok = any(d in docs for d in r["source_docs"])
        hit += ok
        mark = "hit " if ok else "MISS"
        print(f"  {mark} {r['id']}  expected {r['source_docs']}  top{k}={docs}")
    print(f"\n{hit}/{len(rows)} questions had an expected source doc in top-{k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sp = sub.add_parser("search")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sub.add_parser("smoke")
    args = ap.parse_args()

    if args.cmd == "build":
        build()
    elif args.cmd == "search":
        for h in search(args.query, k=args.k):
            print(f"  {h['score']:.3f}  {h['doc']:42} {h['heading_path'][:60]}")
    elif args.cmd == "smoke":
        smoke()


if __name__ == "__main__":
    main()
