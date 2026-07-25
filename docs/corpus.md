# Corpus

Apache Spark documentation, pinned at tag `v3.5.1`.

244 markdown files, roughly 2.75 MB.

## Why this corpus

Three things mattered.

**I can write the answers myself.** An eval harness is only as good as its golden set.
Benchmarks like HotpotQA come with labels but I cannot tell whether a wrong answer is the
retriever's fault or the label's. I have worked with Spark for three years, so when
retrieval returns the wrong chunk I know it is wrong without checking anything.

**It has real structure.** Nested headings, code blocks, config tables. Chunking that
respects structure should beat fixed-size chunking here, and the difference should be
measurable rather than asserted. A corpus of blog posts would not test that.

**It has genuinely hard retrieval cases.** `configuration.md` is one enormous table of
property names. `sql-performance-tuning.md` and `tuning.md` overlap heavily and a
retriever has to pick the right one. Both are the kind of thing that breaks naive chunking.

## Why pinned to a tag

Spark docs change between releases. If the corpus moves, eval scores move, and I would
not know whether I improved retrieval or the source text shifted underneath me. The tag
makes results comparable across weeks.

## Licence

Apache License 2.0. The fetch script pulls from the upstream repository at build time and
writes into `data/`, which is gitignored. Nothing from the corpus is committed here, so
this repo carries no third-party content.

## Reproducibility

`ingest/fetch_corpus.py` writes `data/manifest.json` with a SHA-256 for every file plus
the ref it fetched. Two people running the script get byte-identical corpora, or the
manifest tells them why not.
