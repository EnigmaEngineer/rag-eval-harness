# RAG evaluation harness

A retrieval system that reports its own accuracy. Hybrid BM25 and dense retrieval over the
Apache Spark documentation, with an eval suite that catches regressions before they ship.

```bash
pip install -r requirements.txt
python -m ingest.fetch_corpus     # pulls Spark docs at v3.5.1 into data/
python -m evalset.validate        # checks the golden set against the corpus
python -m ingest.chunk            # structure-aware chunks -> data/chunks.jsonl
python -m retrieval.dense build   # embed chunks, write the FAISS index
python -m retrieval.dense smoke   # sanity-check retrieval on the golden questions
```

## Why

Most RAG projects show a chat box and a demo question. None of them tell you whether
retrieval actually works, so there is no way to know if a change made things better or
worse. This one measures recall@k, MRR and faithfulness on a golden set, and fails CI when
they regress.

## Corpus

Spark docs, pinned to `v3.5.1`. 244 markdown files. I picked it because I can write the
answers myself, so a wrong retrieval is obviously wrong without checking a label. Reasoning
in `docs/corpus.md`.

Nothing from the corpus is committed here. `fetch_corpus.py` pulls it at build time and
writes a manifest with a SHA-256 per file so runs stay comparable.

## Status

Day 2 of 7.

- [x] Day 1: corpus selection and fetch, chunking strategy, eval-set format
- [x] Day 2: structure-aware chunker, dense index
- [ ] Day 3: BM25 index, reciprocal rank fusion
- [ ] Day 4: cross-encoder reranking, per-stage latency
- [ ] Day 5: recall@k, MRR, faithfulness scoring
- [ ] Day 6: ablation runs, CI regression gate
- [ ] Day 7: results and failure analysis

## Retrieval

Chunks are embedded with `BAAI/bge-small-en-v1.5` (384-dim, 512-token window) into a FAISS
flat inner-product index over L2-normalized vectors, which makes the search exact cosine
similarity. Flat is deliberate: at a few thousand chunks a linear scan is fast, and an
approximate index would only add a recall/latency knob to tune before there is a metric to
tune it against. The query gets bge's instruction prefix. Passages do not. See
`docs/chunking.md` for the chunking decisions.

## Known limitations

The golden set has 10 questions. That is enough to catch gross breakage and not enough to
trust a two-point recall difference. Target is 60 by day 5, added when a real retrieval
failure turns up that no existing question covers.

The chunker special-cases HTML tables but not markdown pipe tables. A few markdown rows are
written as one very long physical line and only survive because the oversized-block
windower falls back to splitting on spaces. That is a safety net, not real markdown-table
handling, and it is on the list for a later day.
