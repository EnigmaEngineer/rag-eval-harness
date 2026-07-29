"""Tests for the structure-aware chunker.

Uses the approximate counter so these run without downloading a model. They check the
invariants that actually matter for retrieval: chunks stay near budget, oversized blocks
get split instead of silently truncated, table fragments keep their header and the
heading path rides along on every chunk.

    python -m tests.test_chunk
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.chunk import (  # noqa: E402
    approx_counter, segment, table_units, hard_split, chunk_doc,
)

count = approx_counter()


def test_segment_separates_code_and_headings():
    doc = "# Title\n\nsome prose here.\n\n```python\nx = 1\n```\n\nmore prose."
    _, blocks = segment(doc)
    kinds = [b["kind"] for b in blocks]
    assert "heading" in kinds
    assert "code" in kinds
    code = [b for b in blocks if b["kind"] == "code"][0]
    assert "x = 1" in code["text"] and code["text"].count("```") == 2


def test_front_matter_title_becomes_root_path():
    doc = "---\ntitle: My Page\n---\n# Section\n\nbody text."
    chunks = chunk_doc("x.md", doc, count, budget=512, overlap=64)
    assert chunks and chunks[0]["heading_path"].startswith("My Page")
    assert "Section" in chunks[0]["heading_path"]


def test_table_split_repeats_header():
    tbl = ("<table>\n<thead><tr><th>Name</th><th>Default</th></tr></thead>\n"
           + "\n".join(f"<tr><td>prop{i}</td><td>{i}</td></tr>" for i in range(40))
           + "\n</table>")
    block = {"kind": "table", "text": tbl}
    units = table_units(block, count, budget=80)
    assert len(units) > 1, "a 40-row table should not fit in one 80-token chunk"
    for u in units:
        assert "Name" in u["text"] and "Default" in u["text"], "every fragment keeps the header"


def test_hard_split_bounds_a_giant_code_block():
    big = {"kind": "code", "text": "\n".join(f"line number {i} of a long block" for i in range(400))}
    parts = hard_split(big, count, budget=100, overlap=10)
    assert len(parts) > 1
    assert all(count(p["text"]) <= 100 for p in parts)


def test_hard_split_handles_one_giant_line():
    # a markdown table row written as a single physical line, the real case from
    # sql-ref-ansi-compliance.md that produced a 1,000-token chunk before windowing
    one_line = {"kind": "text", "text": "word " * 500}
    parts = hard_split(one_line, count, budget=60, overlap=8)
    assert all(count(p["text"]) <= 60 for p in parts)


def test_overlap_tail_never_pushes_a_chunk_over_budget():
    """The day-2 build shipped 37 chunks over the 512 budget and the recorded cause was
    wrong. It was not wordpiece non-additivity. The packer reset a full chunk to its overlap
    tail and then appended the unit that had just forced the flush, without re-checking the
    two together. Worst case is room plus the whole tail budget.

    Sizes are picked to hit that window rather than to look tidy. Budget 200 and a one-token
    heading path give room 199 and a tail budget of 40. Alpha then Bravo fills 155. Charlie
    at 180 forces the flush and fits on its own. Bravo at 35 is small enough to be carried as
    overlap, so the second chunk starts at 35 and used to land on 216.

    Budget stays well above 200 on purpose. `pack` floors room at 96 so a long heading path
    cannot starve the content, and a budget near that floor would exercise the floor rather
    than the tail.
    """
    body = " ".join([
        "Alpha " * 119 + ".",
        "Bravo " * 34 + ".",
        "Charlie " * 179 + ".",
    ])
    chunks = chunk_doc("t.md", f"# H\n\n{body}\n", count, budget=200, overlap=40)
    assert len(chunks) > 1, "expected the packer to flush at least once"
    over = [c for c in chunks if c["n_tokens"] > 200]
    assert not over, f"{len(over)} chunk(s) over budget: {[c['n_tokens'] for c in over]}"
    assert all(c["oversized"] is False for c in chunks)


def test_every_chunk_carries_its_heading_path():
    doc = "# A\n\n## B\n\nsome text about tuning.\n\n## C\n\nmore text about memory."
    chunks = chunk_doc("d.md", doc, count, budget=512, overlap=64)
    for c in chunks:
        assert c["heading_path"], "a chunk with no heading path is context-free"
        assert c["text"].startswith(c["heading_path"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as err:
            failed += 1
            print(f"  FAIL  {t.__name__}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
