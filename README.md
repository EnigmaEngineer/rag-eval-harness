# RAG evaluation harness

A retrieval system that reports its own accuracy. Hybrid BM25 and dense retrieval over the
Apache Spark documentation, with an eval suite that catches regressions before they ship.

```bash
pip install -r requirements.txt
python -m ingest.fetch_corpus     # pulls Spark docs at v3.5.1 into data/
python -m evalset.validate        # checks the golden set against the corpus
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

Day 1 of 7.

- [x] Day 1: corpus selection and fetch, chunking strategy, eval-set format
- [ ] Day 2: structure-aware chunker, dense index
- [ ] Day 3: BM25 index, reciprocal rank fusion
- [ ] Day 4: cross-encoder reranking, per-stage latency
- [ ] Day 5: recall@k, MRR, faithfulness scoring
- [ ] Day 6: ablation runs, CI regression gate
- [ ] Day 7: results and failure analysis

## Known limitations

The golden set has 10 questions. That is enough to catch gross breakage and not enough to
trust a two-point recall difference. Target is 60 by day 5, added when a real retrieval
failure turns up that no existing question covers.

`configuration.md` is one enormous table and will almost certainly break the chunker's
"keep tables whole" rule. Noted in `docs/chunking.md`, unsolved on purpose until the
failure is visible.
