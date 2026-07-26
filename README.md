# RAG evaluation harness

A retrieval system that reports its own accuracy. Hybrid BM25 and dense retrieval over the
Apache Spark documentation, with an eval suite that catches regressions before they ship.

```bash
pip install -r requirements.txt
python -m ingest.fetch_corpus     # pulls Spark docs at v3.5.1 into data/
python -m evalset.validate        # checks the golden set against the corpus
python -m ingest.chunk            # structure-aware chunks -> data/chunks.jsonl
python -m retrieval.dense build   # embed chunks, write the FAISS index
python -m retrieval.sparse build  # build the BM25 inverted index
python -m retrieval.fuse smoke    # dense vs bm25 vs fused, side by side
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

Day 3 of 7.

- [x] Day 1: corpus selection and fetch, chunking strategy, eval-set format
- [x] Day 2: structure-aware chunker, dense index
- [x] Day 3: BM25 index, reciprocal rank fusion
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

BM25 runs alongside it as a sparse index over the same chunks. It is hand-rolled, not
`rank-bm25`, so the tokenizer and the k1/b/idf choices stay visible and under test. The two
rankings are combined with reciprocal rank fusion. RRF keeps only the ranks and throws the
raw scores away, which sidesteps having to normalize cosine and BM25 onto a shared scale.
The full argument is in `docs/fusion.md`.

## What day 3 measured, including a result that went the wrong way

The plan said BM25 plus fusion would close the lexical gap that dense retrieval leaves. Half
of that held. The other half did not, and the eval caught it.

The proxy is coarse: for each of the 10 golden questions, does an expected source doc land
in the top 5. Real recall@k and MRR are day 5. Numbers measured on this machine today.

| retriever | source doc in top-5 |
|---|---|
| dense only | 9 / 10 |
| BM25 only | 10 / 10 |
| fused (RRF) | 9 / 10 |

The one question dense misses is q002: "stop Spark splitting my job into too many tiny tasks
after a shuffle." The doc keeps the answer inside the config key
`spark.sql.shuffle.partitions` and never says "tiny tasks," so the paraphrase defeats the
embedding. BM25 catches it at rank 3, exactly the lexical hit it exists for.

Then fusion loses it again. Measured on q002: the correct doc is BM25 rank 3 and dense rank
23, and after RRF it sits at fused rank 7, one place outside the top 5. RRF rewards agreement
across both rankers, so a cluster of docs that both retrievers rank moderately well outscores
a doc only BM25 is sure about. That is RRF's consensus bias, the cost `docs/fusion.md` flags
in the abstract, showing up on a real query.

This is not fixed by tuning the fusion, and tuning it to pass one question on a 10-question
proxy would be gaming the number. The right read is that fusion widened the candidate pool.
The correct chunk is now in the fused top-50, which it was not in dense alone. Pulling it into
the top 5 is the cross-encoder reranker's job on day 4. Whether the reranker actually recovers
q002 is the open question day 4 has to answer, not assume.

## Known limitations

The golden set has 10 questions. That is enough to catch gross breakage and not enough to
trust a two-point recall difference. Target is 60 by day 5, added when a real retrieval
failure turns up that no existing question covers.

The chunker special-cases HTML tables but not markdown pipe tables. A few markdown rows are
written as one very long physical line and only survive because the oversized-block
windower falls back to splitting on spaces. That is a safety net, not real markdown-table
handling, and it is on the list for a later day.

BM25 does no stemming. "partition" and "partitions" are distinct terms. It did not cost a
golden hit on this corpus, but a query using the singular against a doc that only uses the
plural would miss. A real stemmer is a dependency and a source of surprising matches, so it
is deferred rather than added blind.

The three smoke checks report source-doc overlap at top-5, not recall@k on gold chunks. They
are a sanity proxy for spotting where the retrievers disagree. The real metrics are day 5.
