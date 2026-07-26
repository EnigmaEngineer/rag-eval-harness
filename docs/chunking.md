# Chunking strategy

Decided on day 1, implemented on day 2. Written down first so the eval has something to
disprove.

All numbers below come from `python -m ingest.corpus_stats` against the pinned corpus.

## What the corpus actually looks like

244 files, 2687 KB.

| | count |
|---|---|
| headings | 2,519 |
| fenced code blocks | 337 |
| HTML `<table>` tags | 192 |
| HTML `<tr>` rows | 1,714 |
| markdown pipe-table rows | 2,292 |

The important discovery: **Spark docs use HTML tables, not markdown tables.**
`configuration.md` has 21 `<table>` elements and 355 `<tr>` rows across 155 KB. I had
assumed it was one giant markdown table. It is neither one table nor markdown.

Both markup styles appear across the corpus, so the chunker has to handle each.

## The claim

Structure-aware chunking beats fixed-size chunking on this corpus. Day 6 either shows that
in the ablation table or it does not. If it does not I say so.

## Approach

Split on heading boundaries, then pack sibling sections up to a token budget.

- **Budget:** 512 tokens, 64 token overlap. The number follows the embedding model:
  bge-small-en-v1.5 has a 512-token context window and truncates anything past it. The
  day-1 plan set 512 before choosing a model, which was backwards. It happens to be right,
  but for a reason now instead of a hope. Token counts come from the model's own WordPiece
  tokenizer, not a word-count estimate.
- **Reserve the heading path in the budget.** The path is repeated at the top of every
  chunk, so it spends tokens. Content is packed into `512 - len(path)` rather than 512, so
  the *measured* chunk stays under the model's window instead of overflowing once the
  prefix is added.
- **Carry the heading path.** Prefix each chunk with its ancestors, e.g.
  `Tuning Spark > Memory Tuning > Garbage Collection Tuning`. Without it, a chunk saying
  "increase this value" has no idea what "this" is.
- **Split HTML tables on `<tr>` boundaries, repeating the header row.** With 21 tables and
  355 rows in `configuration.md` alone, keeping tables whole is not an option. Repeating
  the header keeps each fragment self-describing.
- **Keep code blocks whole *until* they exceed the budget.** The day-1 rule was "never
  split a fenced code block, let it be its own chunk". Profiling on day 2 broke that: 40
  code blocks and 11 tables run past 512 tokens, the worst example log at 15,757. An
  unsplit block does not stay whole. The model truncates it at 512 and silently drops the
  rest, which is worse than a clean window split. So blocks that fit stay whole. Oversized
  ones get windowed on the coarsest boundary available. Newline first, then space. Character
  only for a handful of markdown table rows written as one 1,000-token line.

## What I expect to go wrong

**HTML inside markdown.** A naive markdown parser treats `<table>` as an opaque text blob,
so heading-based splitting will not see inside it. The chunker needs a pass that
understands both, which is more work than a pure markdown splitter and is the main risk
for day 2.

**Heading-path dilution.** Deeply nested sections carry long prefixes. If the prefix starts
dominating the embedding, retrieval gets worse for exactly the specific subsections where
precision matters most. Worth measuring, not assuming.

**Not every file has structure to exploit.** `tuning.md` has 14 headings, no code blocks
and no tables. On files like that, structure-aware and fixed-size chunking should produce
nearly identical output, which will dampen the overall ablation delta. If the gain shows up
only on table-heavy files, the honest reporting is per-file, not one averaged number.

## Baselines for day 6

1. Fixed 512-token chunks, 64 overlap, no structure awareness. The naive default.
2. Whole documents as chunks. Bad precision, useful as a floor.
3. Structure-aware, as above.

Same eval, same golden set, all three.
