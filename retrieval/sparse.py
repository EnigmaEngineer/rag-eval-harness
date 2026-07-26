"""BM25 sparse retrieval over the same chunks the dense index uses.

    python -m ingest.chunk                 # produce data/chunks.jsonl first
    python -m retrieval.sparse build       # build the inverted index -> indexes/bm25.pkl
    python -m retrieval.sparse search "coalesce small partitions after a shuffle" --k 5
    python -m retrieval.sparse smoke       # the golden questions as a sanity check

Why hand-rolled instead of rank-bm25. BM25 is an inverted index and one scoring formula.
Writing it here keeps the tokenizer, the idf variant and the k1/b choice all visible and
under test, which matters more for a portfolio piece than saving eighty lines. The one real
design decision is the tokenizer. That is the thing worth being able to defend.

The tokenizer lowercases and splits on non-alphanumerics, so a config key like
`spark.sql.shuffle.partitions` splits into spark / sql / shuffle / partitions. That is deliberate.
The dense index misses q002 because the question says "tiny tasks after a shuffle" and never
says "partition", while the doc keeps the term inside the dotted key. Splitting the key
exposes `partitions` as a matchable term. BM25 is here precisely to close that lexical gap.

No stemming. "partition" and "partitions" stay distinct. It did not cost a golden hit here,
and a real stemmer is a dependency plus a source of surprising matches. Noted as a limitation
rather than papered over.
"""

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHUNKS = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "indexes"
INDEX_PATH = INDEX_DIR / "bm25.pkl"

K1 = 1.5
B = 0.75

# A short, boring stopword list. Long lists start dropping terms that carry meaning in a
# technical corpus. Words like "no" and "not" and "off" and "on" all matter in config docs.
# So this stays small.
STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "with", "as", "by", "at", "be", "this", "that", "it", "from", "how", "do",
    "i", "my", "you", "your", "can", "does", "what", "when", "which",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def load_chunks():
    if not CHUNKS.exists():
        raise SystemExit("no chunks. run: python -m ingest.chunk")
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]


META_KEYS = ("id", "doc", "heading_path", "kind", "n_tokens")


def build():
    """Build the inverted index once and pickle it.

    Fast enough (pure Python over ~3k chunks) that this could run at query time, but a
    persisted index keeps parity with the dense path and makes fusion load two ready
    indexes instead of rebuilding one. Postings are {term: [(chunk_idx, tf), ...]}.
    """
    chunks = load_chunks()
    n = len(chunks)
    postings = defaultdict(list)
    doc_len = [0] * n
    for i, c in enumerate(chunks):
        toks = tokenize(c["text"])
        doc_len[i] = len(toks)
        for term, tf in Counter(toks).items():
            postings[term].append((i, tf))

    avgdl = sum(doc_len) / n if n else 0.0
    meta = [{k: c[k] for k in META_KEYS} for c in chunks]

    INDEX_DIR.mkdir(exist_ok=True)
    with INDEX_PATH.open("wb") as fh:
        pickle.dump(
            {"n": n, "avgdl": avgdl, "doc_len": doc_len,
             "postings": dict(postings), "meta": meta},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"bm25 index built: {n} chunks, {len(postings)} terms, avgdl {avgdl:.1f} "
          f"-> {INDEX_PATH.relative_to(ROOT)}")
    return True


class BM25:
    def __init__(self, data):
        self.n = data["n"]
        self.avgdl = data["avgdl"]
        self.doc_len = data["doc_len"]
        self.postings = data["postings"]
        self.meta = data["meta"]

    def _idf(self, df):
        # BM25+ non-negative idf. The textbook Robertson idf goes negative for a term in
        # more than half the docs, which lets a very common word pull a score down. This
        # variant floors it so a match never actively hurts.
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query, k=5):
        scores = defaultdict(float)
        for term in tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self._idf(len(plist))
            for idx, tf in plist:
                dl = self.doc_len[idx]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[idx] += idf * (tf * (K1 + 1)) / denom
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits = []
        for idx, score in top:
            row = dict(self.meta[idx])
            row["score"] = float(score)
            hits.append(row)
        return hits


def load():
    if not INDEX_PATH.exists():
        raise SystemExit("no bm25 index. run: python -m retrieval.sparse build")
    # The pickle is this project's own build() output, gitignored, never fetched from
    # anywhere. So the usual do-not-unpickle-untrusted-data caution does not apply here.
    with INDEX_PATH.open("rb") as fh:
        return BM25(pickle.load(fh))


def smoke():
    """Same coarse proxy the dense path uses: for each golden question, does an expected
    source doc land in the top-k? Not recall@k on gold chunks (that is day 5). The point
    today is to see where BM25 and dense disagree, q002 above all."""
    golden = ROOT / "evalset" / "golden.jsonl"
    rows = [json.loads(l) for l in golden.read_text(encoding="utf-8").splitlines() if l.strip()]
    bm25 = load()
    k = 5
    hit = 0
    for r in rows:
        got = bm25.search(r["question"], k=k)
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
        for h in load().search(args.query, k=args.k):
            print(f"  {h['score']:.3f}  {h['doc']:42} {h['heading_path'][:60]}")
    elif args.cmd == "smoke":
        smoke()


if __name__ == "__main__":
    main()
