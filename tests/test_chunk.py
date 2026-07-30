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
    approx_counter, segment, table_units, hard_split, pack, chunk_doc, is_documentation,
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


def test_room_floor_produces_a_chunk_that_is_flagged_oversized():
    """`pack` floors room at 96 so a very long heading path cannot starve the content down
    to nothing. The cost is that the chunk it produces can exceed the budget. This has been
    an open question for three days: does the floor let an over-budget chunk through
    unreported?

    It does not. `emit` computes `oversized` as `n > budget`, and it never looks at `room`.
    So a floor-produced chunk is counted honestly by the same rule as every other chunk.
    That is worth a test rather than another note, because the cheap way to write `emit`
    is against `room`, and then this whole class of chunk goes silently unflagged.

    Sizes are chosen to land inside the floor. A 20-word heading path costs 39 tokens, and
    a 120 budget leaves 81, which is below 96. So room floors up to 96 and a single 96-token
    unit lands at 135 against a budget of 120.
    """
    path = " > ".join(f"section{i}" for i in range(20))
    budget, overlap = 120, 16
    prefix_cost = count(path + "\n")
    assert budget - prefix_cost < 96, "sizes drifted, this no longer exercises the floor"

    units = [{"kind": "text", "text": "alpha " * 96}]
    chunks = pack(units, path, count, budget, overlap)

    assert len(chunks) == 1
    c = chunks[0]
    assert c["n_tokens"] > budget, "the floor is supposed to overrun here, that is the point"
    assert c["oversized"] is True, "a floor-produced chunk went unflagged"


def test_meta_files_are_not_documentation():
    assert is_documentation("configuration.md")
    assert not is_documentation("README.md"), "docs-site build instructions, not documentation"
    assert not is_documentation("index.md")
    assert not is_documentation("404.md")
    assert not is_documentation("Makefile")


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
