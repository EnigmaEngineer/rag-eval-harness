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
python -m retrieval.rerank smoke  # fused vs reranked, with per-stage latency
python -m harness.run_eval run    # score every system on the golden set
python -m harness.run_eval report # the ablation table
python -m harness.gate check      # fail if a metric regressed past its tolerance
python -m harness.significance sweep  # is any gap in the table bigger than one question
```

`run` appends one row per system and question and skips pairs it has already recorded, so it
is safe to interrupt and rerun. Rerank costs about 3.5 seconds a question, so a full four
system pass takes a few minutes. Use `--systems` and `--slice` to work through it in pieces.

## Why

Most RAG projects show a chat box and a demo question. None of them tell you whether
retrieval actually works, so there is no way to know if a change made things better or
worse. This one measures recall@k, MRR and faithfulness on a golden set. It fails CI when
they regress.

## Corpus

Spark docs, pinned to `v3.5.1`. 241 markdown files. I picked it because I can write the
answers myself, so a wrong retrieval is obviously wrong without checking a label. Reasoning
in `docs/corpus.md`.

`README.md`, `index.md` and `404.md` are excluded. They live in the docs folder but they are
the docs site's build instructions, a link menu and an error page. A file that mentions every
topic and explains none is the worst thing a lexical index can hold.

Nothing from the corpus is committed here. `fetch_corpus.py` pulls it at build time and
writes a manifest with a SHA-256 per file so runs stay comparable.

## Retrieval

Chunks are embedded with `BAAI/bge-small-en-v1.5` (384-dim, 512-token window) into a FAISS
flat inner-product index over L2-normalized vectors, which makes the search exact cosine
similarity. Flat is deliberate. At a few thousand chunks a linear scan is fast. An
approximate index would only add a recall/latency knob to tune before there is a metric to
tune it against. The query gets bge's instruction prefix. Passages do not. See
`docs/chunking.md` for the chunking decisions.

BM25 runs alongside it as a sparse index over the same chunks. It is hand-rolled, not
`rank-bm25`, so the tokenizer and the k1/b/idf choices stay visible and under test. The two
rankings are combined with reciprocal rank fusion. RRF keeps only the ranks and throws the
raw scores away, which sidesteps having to normalize cosine and BM25 onto a shared scale.
The full argument is in `docs/fusion.md`.

The fused pool then goes through a cross-encoder reranker,
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The dense retriever is a bi-encoder. Query and
passage become two separate vectors, compared by cosine, and the passage vectors are
precomputed. That is what lets it scan every chunk in milliseconds. The query never sees
the passage. A cross-encoder concatenates query and passage and runs one transformer
forward over the pair, so every query term attends to every passage term. Much better at
telling a real answer from a near-miss, and far too slow to run over the corpus. One
forward per pair, nothing cacheable. So it runs last, over the short fused pool, not the
3228 chunks.

## Results

Every number below was measured on this machine. Ten golden questions, chunk-level gold labels
from `evalset/gold_chunks.jsonl`, the same questions and labels for all four systems.

| system | recall@1 | recall@5 | recall@10 | hit@1 | hit@5 | MRR@10 | support@5 | ms |
|---|---|---|---|---|---|---|---|---|
| dense only | 0.167 | 0.600 | 0.633 | 0.400 | 0.700 | 0.483 | 0.700 | 23 |
| BM25 only | 0.200 | **0.700** | **0.750** | 0.500 | **0.800** | 0.607 | **0.900** | **1** |
| fused (RRF) | **0.250** | 0.633 | 0.633 | **0.600** | 0.700 | **0.620** | 0.750 | 23 |
| fused + rerank | 0.217 | 0.450 | 0.683 | 0.500 | 0.700 | 0.592 | 0.700 | 4412 |

**BM25 alone wins on four of the eight columns and costs one millisecond.** It has the best
recall at 5 and at 10, the best hit@5, and the best evidence support. The dense retriever,
the expensive part, is last on almost everything.

**The reranker is worse than the fusion it reranks.** recall@5 drops from 0.633 to 0.450 and
MRR from 0.620 to 0.592, for 4412 ms a query. A doc-overlap proxy cannot measure ordering,
which is the only thing a reranker changes. That was the reason for building these metrics
and it is the first thing they caught.

So the reranker came off the default path. It still lives in `retrieval/rerank.py`, because
the measurement is the point. Whether hybrid holds up against plain BM25 after the chunk
rebuild is a longer story and it is in the design notes.

Ten questions is a small eval set. Seven of the eight metric comparisons above move three
questions or fewer, and below six no gap can reach p = 0.05 at any effect size. Treat the
table as a regression check for this project rather than a benchmark.

## Known limitations

- The golden set has 10 questions. That catches gross breakage and cannot back a two point
  recall difference.
- The reranker is an off the shelf cross-encoder trained on MS MARCO, not on Spark docs.
- At about 3.5 seconds a query on 2 CPU cores, reranking every query is not something to ship
  as is.
- BM25 does no stemming, so "partition" and "partitions" are different terms.
- Faithfulness is not measured, because there is no generator here to be faithful.
- Gold labels are minimal sufficient rather than exhaustive, so every recall number is a floor.
- API docs full of exact config keys are unusually kind to a lexical retriever.

The full list with the reasoning is in [docs/limitations.md](docs/limitations.md).

## More detail

**[Design notes](docs/design-notes.md)**

- [A hybrid result that went the wrong way](docs/design-notes.md#a-hybrid-result-that-went-the-wrong-way)
- [The reranker did not recover q002](docs/design-notes.md#the-reranker-did-not-recover-q002)
- [The ablation table, and what it decided](docs/design-notes.md#the-ablation-table-and-what-it-decided)
  - [What changed as a result](docs/design-notes.md#what-changed-as-a-result)
  - [The rebuild that changed almost nothing](docs/design-notes.md#the-rebuild-that-changed-almost-nothing)
  - [Where the retrievers actually fail](docs/design-notes.md#where-the-retrievers-actually-fail)
  - [Is any gap in this table bigger than one question](docs/design-notes.md#is-any-gap-in-this-table-bigger-than-one-question)
  - [support@5 reports full marks on a question every system fails](docs/design-notes.md#support5-reports-full-marks-on-a-question-every-system-fails)
- [The regression gate](docs/design-notes.md#the-regression-gate)
- [Latency per stage](docs/design-notes.md#latency-per-stage)

**Other docs**

- [chunking](docs/chunking.md)
- [corpus](docs/corpus.md)
- [fusion](docs/fusion.md)
- [labelling](docs/labelling.md)
