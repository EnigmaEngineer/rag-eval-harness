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
| dense only | 0.167 | 0.600 | 0.633 | 0.400 | 0.700 | 0.483 | 0.700 | 23 |
| BM25 only | 0.200 | **0.700** | **0.750** | 0.500 | **0.800** | 0.607 | **0.900** | **1** |
| fused (RRF) | **0.250** | 0.633 | 0.633 | **0.600** | 0.700 | **0.620** | 0.750 | 23 |
| fused + rerank | 0.217 | 0.450 | 0.683 | 0.500 | 0.700 | 0.592 | 0.700 | 4412 |

`recall@k` is the fraction of a question's gold chunks inside the top k. `hit@k` asks only
whether one of them made it, so it does not punish a question for having three gold chunks
instead of one. Read them together. `support@5` is the share of the config keys named in the
reference answer that appear anywhere in the top-5 text. Definitions and the reasoning behind
each are in `harness/metrics.py`.

**BM25 alone wins on four of the eight columns and costs one millisecond.** It has the best
recall at 5 and at 10, the best hit@5, and the best evidence support. The dense retriever,
the expensive part, is last on almost everything.

**The reranker is worse than the fusion it reranks.** recall@5 drops from 0.633 to 0.450 and
MRR from 0.620 to 0.592, for 4412 ms a query. A doc-overlap proxy cannot measure ordering,
which is the only thing a reranker changes. That was the reason for building these metrics
and it is the first thing they caught.

Hybrid keeps a narrow lead where it was always supposed to: the top of the list. It has the
best recall@1, hit@1 and MRR@10. On ten questions a 0.1 gap in hit@1 is one question, which
this eval set is too small to call a real difference.

### What changed as a result

The reranker comes off the default path. Nothing in these numbers justifies 3.7 seconds a
query for a measurably worse ranking. `retrieval/rerank.py` stays, because the measurement is
the point and a domain-tuned cross-encoder is still the obvious next thing to try.

Whether hybrid survives against plain BM25 took two more days to settle. The plan was to
decide it after the chunk rebuild, on the grounds that re-embedding the corpus would move
every number here. It did not. Both sections below are that story.

### The rebuild that changed almost nothing

Two defects were fixed together so their effect could be measured in one pass. 37 chunks ran
up to 62 tokens past the 512 budget, so bge silently truncated their tails. And the corpus
included `README.md`, `index.md` and `404.md`, which are the docs site's own build
instructions, a link menu and an error page.

Both are real bugs. The corpus went from 244 files and 3,228 chunks to 241 files and 3,212
chunks, no chunk now exceeds 512 tokens, and the whole corpus was re-embedded against the
rebuilt text.

**23 of the 28 quality numbers came back byte-identical.** recall@1, hit@1, hit@5 and
support@5 did not move on any of the four systems. The five that moved got slightly worse.

| moved | before | after |
|---|---|---|
| bm25 recall@10 | 0.783 | 0.750 |
| fused recall@10 | 0.683 | 0.633 |
| fused MRR@10 | 0.634 | 0.620 |
| rerank recall@5 | 0.500 | 0.450 |
| dense MRR@10 | 0.478 | 0.483 |

`README.md` did leave rank 1 on q009. It was replaced by `job-scheduling.md`, and the gold
chunk stayed at rank 5 either way. Deleting the junk document changed the ranking without
changing the answer.

The useful lesson is about where effort goes. Both defects looked serious enough to hold a
decision open for a day. Neither was load-bearing. The real reason q005 and q009 fail has
nothing to do with chunk boundaries, and a full re-embed was the expensive way to find that
out. It was still worth running, because "we assumed it mattered" and "we measured it and it
did not" are different claims and only one of them is evidence.

### Where the retrievers actually fail

q005 scores zero on all four systems. It asks whether a broadcast join works when one side
does not fit in driver memory. Every system returns join-hint prose while the labelled
evidence, the `spark.sql.autoBroadcastJoinThreshold` row, sits inside a large HTML config
table. The question is also the one with the grounding defect described below. It is a bad
question before it is a retrieval failure.

q009 still scores 0.200 on BM25 and zero on the other three. It asks why a completed
application was slow, and the gold chunk is the history server section of `monitoring.md`.
Removing the meta files moved `README.md` off rank 1 and moved nothing else. The gold chunk
sat at rank 5 before and after.

### Is any gap in this table bigger than one question

Two days were spent refusing to call the hybrid-versus-BM25 result on the grounds that the
numbers were close. That was the right instinct and a bad way to decide anything, so it got
replaced with a test.

```bash
python -m harness.significance compare --a fused --b bm25
```

Paired on question id, because both systems answer the same ten questions against the same
labels. The p-value is an exact permutation test. Under the null the two systems are
interchangeable, so flipping the sign of any paired difference is a valid relabelling. Ten
questions give 2^10 = 1024 sign assignments and all of them are enumerated. No sampling and
no seed. The interval is a paired bootstrap over questions, which is the thing a wider eval
set would change.

| metric | diff | 95% CI | p | questions moved |
|---|---|---|---|---|
| recall@1 | +0.050 | +0.000 to +0.150 | 1.000 | 1/10 |
| recall@5 | -0.067 | -0.300 to +0.100 | 1.000 | 2/10 |
| recall@10 | -0.117 | -0.350 to +0.067 | 0.500 | 3/10 |
| hit@1 | +0.100 | +0.000 to +0.300 | 1.000 | 1/10 |
| hit@5 | -0.100 | -0.300 to +0.000 | 1.000 | 1/10 |
| MRR@10 | +0.013 | -0.077 to +0.133 | 1.000 | 3/10 |
| support@5 | -0.150 | -0.350 to +0.000 | 0.500 | 2/10 |
| ms | +21 | +20 to +24 | 0.002 | 10/10 |

Seven of the eight intervals include zero. The one that does not is latency, where fusion
costs 21 ms more than BM25. **The cost of hybrid retrieval is measurable on this eval set and
the benefit is not.**

The `ms` row is there as a positive control and it earns its place. Its p-value of 0.002 is
2/1024, the smallest value ten questions can produce. Without a row that fires, eight
non-significant results are indistinguishable from a broken test.

The `p = 1.000` on recall@1 is not a rounding artefact. Exactly one question separates the two
systems there, and flipping the sign of nine zeros changes nothing, so every one of the 1024
assignments is as extreme as the observed one.

That generalises, and `p_floor()` reports it directly. Ties drop out of the permutation space,
so only the k questions that actually differ matter, leaving a smallest possible p-value of
2/2^k.

| questions that differ | best p-value available |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 4 | 0.125 |
| 6 | 0.031 |
| 10 | 0.002 |

Below six moved questions the comparison cannot reach 0.05 no matter how large the gap is.
Seven of the eight rows above move three questions or fewer, so every quality comparison in
the table was unwinnable before any data was collected. Only latency moved all ten.
**The answer to a gap this size is more questions, not more tuning.**

Hybrid stays on the default path, and the reason is now stated honestly rather than implied by
a table. It is not that fusion was shown to be better. It is that fusion is not shown to be
worse, it costs 21 ms, and it wins the rank-1 metrics that matter most for a downstream
generator that only sees the top few chunks. That is a design preference standing in for
evidence this eval set is too small to supply. If the golden set reaches 60 questions and the
gap still moves three questions, fusion should come off.

### support@5 reports full marks on a question every system fails

The worst finding of the project, and it is about a metric built here rather than a retriever.

```bash
python -m harness.diagnose
```

```
3 disagreement(s):

| qid | kind | systems |
|---|---|---|
| q002 | evidence without support | bm25, rerank |
| q005 | support without evidence | all |
| q007 | evidence without support | dense, rerank |

7 of 10 questions rest on one required string or none
```

q005 scores `support@5` of 1.0 on all four systems while `hit@10` is 0 on all four. Every
system missed the labelled answer and the metric described above as a ceiling on faithfulness
reported a perfect score.

The cause is that `spark.sql.autoBroadcastJoinThreshold` appears in three chunks of
`sql-performance-tuning.md`. The retrievers returned `#0005`, which names the key while
explaining `BROADCAST` query hints and carries neither the 10 MB default nor the `-1` disable.
It does not answer the question. Substring presence satisfied `support@5` anyway.

This was visible in the day-5 table and went unnoticed for two days, because a column of means
cannot show that two metrics disagree about the same question. Across all 40 rows `support@5`
returns 1.0 on 30 of them, and 7 of the 10 questions have one required string or none, which
makes it close to a one-bit metric on most of the set. The ceiling is real and it does not
bind.

`harness/diagnose.py` exists to catch that shape. A disagreement on one system out of four
usually means that system retrieved badly. A disagreement on all four means the eval set is
wrong, and `--strict` exits nonzero on those.

## The regression gate

`harness/gate.py` compares the current report against `reports/baseline.json` and exits
nonzero when a tracked metric has fallen further than its tolerance.

```bash
python -m harness.gate check
python -m harness.gate update --note "why the numbers moved"
```

Tolerances are per metric and deliberately loose. The golden set has ten questions, so one
question flipping is 0.100 of recall@1. A gate tuned tighter than the noise floor fails on
nothing real and gets switched off within a week, which is worse than no gate. Latency gets a
2x band rather than a millisecond budget, because a CI runner is not this machine.

Re-baselining requires a `--note`. That is the whole design. Numbers are allowed to move, but
somebody has to type a reason, and it lands in the diff where a reviewer sees it.

`.github/workflows/eval.yml` runs the unit tests on every push, then fetches the corpus,
chunks it, builds BM25 and gates on it. **Dense, fused and rerank are not gated on every
push.** Embedding 3,212 chunks with bge-small takes about 17 minutes on two cores, which is
what a hosted runner gives you, and paying that per pull request to gate a ten-question eval
is not a trade worth making. They run on `workflow_dispatch` instead. BM25 is also the system
hardest to beat in the table above, so it is the right thing to protect by default.

CI writes to `reports/ci-results.jsonl` rather than the committed `reports/results.jsonl`.
`run_eval` is resumable and skips pairs it already has, so pointing CI at the committed file
would skip every question, write nothing and pass a gate that measured nothing.

## Latency per stage

The reranker buys precision with time. Mean per-query latency over the 10 golden questions,
measured with `perf_counter` inside the pipeline.

| stage | mean ms |
|---|---|
| dense | 21.0 |
| BM25 | 1.4 |
| fuse | 0.1 |
| rerank | 3567.7 |

Those per-stage figures are from the day-4 pipeline run. The end-to-end column in the
ablation table above was re-measured on the rebuilt corpus and reads 23 ms for dense and 4412
ms for rerank on the same hardware. Rerank latency moves several hundred milliseconds between
runs on a 2-core box, so treat it as a magnitude and not a benchmark.

Getting a trustworthy number here took two separate fixes. The first was loading the model
outside the timer. The second is that torch does not allocate buffers or pick kernels until
the first forward pass, so a freshly constructed model still charges its warmup to whichever
query runs first. That read 1606 ms against 19 ms for the other nine and dragged the dense
mean to 178 ms, a number no query in the run actually took. `prime()` now pushes a throwaway
query through every model before any timer starts.

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

`python -m evalset.validate --claims` now scores every prose clause against the best-matching
paragraph in its source docs and lists the weakest first. **It is advisory and it does not
cleanly find the defect.** q005's fabricated claim scores 0.50 and lands second on a list of
ten, below a correct q006 claim at 0.40. So it narrows ten claims to a shortlist worth reading
by hand and it is not a gate. Two sharper approaches were measured and thrown away first.
Anchoring each sentence to a config key flagged 8 clauses of which 1 was the real defect.
Content bigram coverage scored the fabricated clause at 0.00 and scored three correct clauses at
0.00 as well. Neither discriminates, and shipping either would have looked rigorous while being
noise.

Faithfulness is not measured, because there is no generator in this repo to be faithful or
unfaithful. What `support@5` measures is whether the retrieved context even contains the
evidence a correct answer would need. That is a ceiling on faithfulness rather than a
measurement of it, and it is named accordingly. As of day 7 that ceiling is known not to bind.
It returns 1.0 on 30 of 40 rows, 7 of the 10 questions rest on a single required string, and on
q005 it reports 1.0 while every system misses the answer. Treat it as a smoke alarm for missing
evidence and not as a quality score. `harness/diagnose.py` is the check that catches it.

Every quality comparison in the ablation table is underpowered, and `p_floor()` quantifies it
rather than leaving it as a hedge. Below six questions moving, no gap can reach p = 0.05 at any
effect size. Seven of eight metric comparisons move three questions or fewer. This is the
single strongest argument for growing the golden set and it is why "target is 60" above is a
plan rather than a nice-to-have.

Ten questions, one corpus, one machine. These numbers are for catching a regression in this
project. They are not a benchmark of BM25 against dense retrieval in general, and a corpus of
API documentation dense with exact config keys is unusually kind to a lexical retriever. 11 of
the 20 gold chunks literally contain a config key from their own reference answer, so some of
BM25's advantage is the corpus and some of it is real.
