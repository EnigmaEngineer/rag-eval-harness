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

Spark docs, pinned to `v3.5.1`. 244 markdown files. I picked it because I can write the
answers myself, so a wrong retrieval is obviously wrong without checking a label. Reasoning
in `docs/corpus.md`.

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

The fused pool then goes through a cross-encoder reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`.
The dense retriever is a bi-encoder. Query and passage become two separate vectors, compared
by cosine, and the passage vectors are precomputed. That is what lets it scan every chunk in
milliseconds. The query never sees the passage. A cross-encoder concatenates query and
passage and runs one transformer forward over the pair, so every query term attends to every
passage term. Much better at telling a real answer from a near-miss, and far too slow to run
over the corpus. One forward per pair, nothing cacheable. So it runs last, over the short
fused pool, not the 3228 chunks.

## A hybrid result that went the wrong way

The expectation was that BM25 plus fusion would close the lexical gap that dense retrieval
leaves. Half of that held. The other half did not. The eval caught it.

The proxy is coarse: for each of the 10 golden questions, does an expected source doc land
in the top 5. Real recall@k and MRR come next. Numbers measured on this machine.

| retriever | source doc in top-5 |
|---|---|
| dense only | 9 / 10 |
| BM25 only | 10 / 10 |
| fused (RRF) | 9 / 10 |
| fused + rerank | 9 / 10 |

The one question dense misses is q002: "stop Spark splitting my job into too many tiny tasks
after a shuffle." The doc keeps the answer inside the config key
`spark.sql.shuffle.partitions` and never says "tiny tasks," so the paraphrase defeats the
embedding. BM25 catches it at rank 3, exactly the lexical hit it exists for.

Then fusion loses it again. Measured on q002: the correct doc is BM25 rank 3 and dense rank
23. After RRF it sits at fused rank 7. One place outside the top 5. RRF rewards agreement
across both rankers, so a cluster of docs that both retrievers rank moderately well outscores
a doc only BM25 is sure about. That is RRF's consensus bias, the cost `docs/fusion.md` flags
in the abstract, showing up on a real query.

This is not fixed by tuning the fusion. Tuning it to pass one question on a 10-question
proxy would be gaming the number. The right read is that fusion widened the candidate pool.
The correct chunk is now in the fused top-50, which it was not in dense alone. Pulling it into
the top 5 was meant to be the cross-encoder reranker's job.

## The reranker did not recover q002

Day 3 predicted the reranker would rescue q002. It did not. The correct chunk moved from
fused rank 7 to reranked rank 6. Still outside the top 5. Measured on this machine with the
rank trace in the day-4 audit.

The reason is worth more than the fix. The question asks how to stop Spark making too many
tiny tasks after a shuffle. The right answer coalesces post-shuffle partitions with adaptive
query execution, and it lives in `sql-performance-tuning.md`. The cross-encoder instead put
`tuning.md` on top at score 2.453, a passage titled "Memory Usage of Reduce Tasks" that tells
you to *increase* parallelism so each task is smaller. That is the opposite fix. It shares
the vocabulary of the question almost word for word. Shuffle. Tasks. Level of parallelism.
The reranker scored the correct chunk at minus 0.733 and ranked it sixth.

So the cross-encoder rewarded surface topicality over the actual answer. It cannot tell
"coalesce partitions" from "increase parallelism" when both sit in dense shuffle-tuning
prose. This is the honest limit of a relevance model that was never trained on Spark. At chunk
level the reranker leaves q002's correct chunk at rank 6, the same place day 4 found it. The
exit condition was written in `docs/fusion.md`: if it loses to BM25 alone on real metrics,
hybrid gets cut. The table below is that measurement.

## The ablation table, and what it decided

Every number below was measured on this machine. Ten golden questions, chunk-level gold labels
from `evalset/gold_chunks.jsonl`, the same questions and labels for all four systems.

| system | recall@1 | recall@5 | recall@10 | hit@1 | hit@5 | MRR@10 | support@5 | ms |
|---|---|---|---|---|---|---|---|---|
| dense only | 0.167 | 0.600 | 0.633 | 0.400 | 0.700 | 0.478 | 0.700 | 22 |
| BM25 only | 0.200 | **0.700** | **0.783** | 0.500 | **0.800** | 0.607 | **0.900** | **1** |
| fused (RRF) | **0.250** | 0.633 | 0.683 | **0.600** | 0.700 | **0.634** | 0.750 | 23 |
| fused + rerank | 0.217 | 0.500 | 0.683 | 0.500 | 0.700 | 0.592 | 0.700 | 3755 |

`recall@k` is the fraction of a question's gold chunks inside the top k. `hit@k` asks only
whether one of them made it, so it does not punish a question for having three gold chunks
instead of one. Read them together. `support@5` is the share of the config keys named in the
reference answer that appear anywhere in the top-5 text. Definitions and the reasoning behind
each are in `harness/metrics.py`.

**BM25 alone wins on four of the eight columns and costs one millisecond.** It has the best
recall at 5 and at 10, the best hit@5, and the best evidence support. The dense retriever,
the expensive part, is last on almost everything.

**The reranker is worse than the fusion it reranks.** recall@5 drops from 0.633 to 0.500 and
MRR from 0.634 to 0.592, for 3755 ms a query. Day 4 could not see this. A doc-overlap proxy
cannot measure ordering, which is the only thing a reranker changes. That was the reason for
building these metrics and it is the first thing they caught.

Hybrid keeps a narrow lead where it was always supposed to: the top of the list. It has the
best recall@1, hit@1 and MRR@10. On ten questions a 0.1 gap in hit@1 is one question, which
this eval set is too small to call a real difference.

### What changed as a result

The reranker comes off the default path. Nothing in these numbers justifies 3.7 seconds a
query for a measurably worse ranking. `retrieval/rerank.py` stays, because the measurement is
the point and a domain-tuned cross-encoder is still the obvious next thing to try.

Whether hybrid survives against plain BM25 is deliberately not decided here. Day 6 rebuilds
the chunks to fix the token-budget overflow, which re-embeds the corpus and moves every number
in this table. Cutting a component on figures that are about to change would be guessing.
The comparison gets rerun after the rebuild and the exit condition stays as written.

### Where the retrievers actually fail

q005 scores zero on all four systems. It asks whether a broadcast join works when one side
does not fit in driver memory. Every system returns join-hint prose while the labelled
evidence, the `spark.sql.autoBroadcastJoinThreshold` row, sits inside a large HTML config
table. The question is also the one with the grounding defect described below. It is a bad
question before it is a retrieval failure.

q009 exposes a corpus problem. BM25 ranks `README.md` first for "why was a completed
application slow." That file is Spark's instructions for building its own documentation. The
corpus fetch takes every markdown file in the docs folder, including `README.md`, `index.md`
and `404.md`, none of which are documentation. Filtering them is folded into the day-6
rebuild rather than done here, because it changes every number in the table above.

## Latency per stage

The reranker buys precision with time. Mean per-query latency over the 10 golden questions,
measured with `perf_counter` inside the pipeline.

| stage | mean ms |
|---|---|
| dense | 21.0 |
| BM25 | 1.4 |
| fuse | 0.1 |
| rerank | 3567.7 |

Rerank is 99.4 percent of the query budget and roughly 160 times the three retrieval stages
combined. That is 50 query-passage pairs through a cross-encoder on 2 CPU cores. It is the
number that decides whether reranking every query is worth it, and the reason production
reranking runs on a GPU or over a much shorter pool.

## Known limitations

The golden set has 10 questions. That is enough to catch gross breakage and not enough to
trust a two-point recall difference. Target is 60, added when a real retrieval failure turns
up that no existing question covers.

The chunker special-cases HTML tables but not markdown pipe tables. A few markdown rows are
written as one very long physical line and only survive because the oversized-block
windower falls back to splitting on spaces. That is a safety net, not real markdown-table
handling. It is on the list.

BM25 does no stemming. "partition" and "partitions" are distinct terms. It did not cost a
golden hit on this corpus, but a query using the singular against a doc that only uses the
plural would miss. A real stemmer is a dependency and a source of surprising matches, so it
is deferred rather than added blind.

The reranker is an off-the-shelf cross-encoder trained on MS MARCO web search, not on Spark
docs. q002 shows the cost. It reads shuffle-tuning prose and cannot separate the right fix
from the wrong one when both use the same words. A domain-tuned reranker would likely close
that gap. Fine-tuning one is out of scope for this project and noted rather than attempted.

At roughly 3.5 seconds a query on 2 CPU cores, reranking every query is not something you
would ship as is. The pool size is a knob (`--pool`) and the honest production answer is a
GPU or a shorter pool. The point here is to measure the tradeoff, not to hide it.

The smoke checks report source-doc overlap at top-5, not recall@k on gold chunks. They are kept
as a fast sanity proxy for spotting where the retrievers disagree. `harness/run_eval.py` is the
metric that counts.

The gold chunk labels are minimal-sufficient rather than exhaustive. A chunk is gold only if it
would let a reader answer the question on its own. Chunks that are relevant but not sufficient
are scored as misses, so every recall number here is a floor. q009 is the clearest case. Three
systems return the history server configuration table, which is the same section as the labelled
chunk and arguably useful, and score zero for it. The labels were fixed before any system was
run and have not been touched since. Moving a label after seeing a score is how an eval set
stops meaning anything. Full method in `docs/labelling.md`.

q005's reference answer claims a broadcast join needs the smaller relation to fit in driver and
executor memory. No chunk in either cited source doc says that. `evalset/validate.py` missed it
because its grounding check verifies config keys and declared spans, and q005 has no spans while
its one config key is present. The answer has been left as written rather than quietly edited to
match the corpus.

Faithfulness is not measured, because there is no generator in this repo to be faithful or
unfaithful. What `support@5` measures is whether the retrieved context even contains the
evidence a correct answer would need. That is a ceiling on faithfulness rather than a
measurement of it, and it is named accordingly.

Ten questions, one corpus, one machine. These numbers are for catching a regression in this
project. They are not a benchmark of BM25 against dense retrieval in general, and a corpus of
API documentation dense with exact config keys is unusually kind to a lexical retriever. 11 of
the 20 gold chunks literally contain a config key from their own reference answer, so some of
BM25's advantage is the corpus and some of it is real.
