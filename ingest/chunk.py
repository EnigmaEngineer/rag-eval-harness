"""Structure-aware chunker for the Spark docs corpus.

The plan in docs/chunking.md, made real. Three things drive the design and they all
come from profiling the corpus on day 1, not from a blog post:

  - Spark docs mix markdown headings, fenced code and HTML tables in the same file.
  - configuration.md alone is 21 HTML tables / 355 rows. Keeping tables whole is not an
    option, so tables are split on <tr> boundaries with the header row repeated.
  - A chunk that says "increase this value" is useless without its heading path, so every
    chunk carries "Tuning > Memory > GC" style ancestry.

    python -m ingest.chunk                 # writes data/chunks.jsonl, prints stats
    python -m ingest.chunk --tokenizer approx   # no model download, approximate counts
    python -m ingest.chunk --budget 512 --overlap 64
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from ingest.fetch_corpus import is_documentation

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "chunks.jsonl"

# bge-small-en-v1.5 takes 512 tokens. The budget follows the model, not the other way
# round (that ordering was the mistake flagged on day 1). 64-token overlap so a fact that
# straddles a boundary still lands whole in one of the two neighbours.
DEFAULT_BUDGET = 512
DEFAULT_OVERLAP = 64
MODEL = "BAAI/bge-small-en-v1.5"

FRONT_MATTER = re.compile(r"^---\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*```")
TITLE = re.compile(r"^(?:displayTitle|title):\s*(.+?)\s*$")
# body rows carry <td>, the header row carries <th>. thead/tbody wrappers are ignored;
# we only care about the <tr> spans.
TR = re.compile(r"<tr\b.*?</tr>", re.I | re.S)
TH = re.compile(r"<th\b", re.I)


# ---- token counting -------------------------------------------------------------------

def bge_counter():
    """Real WordPiece counts from the model's own tokenizer. Downloads once, cached."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    def count(text):
        # no special tokens: we are budgeting content. The two [CLS]/[SEP] tokens are
        # a fixed overhead the model adds regardless of how we chunk.
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


def approx_counter():
    """Dependency-free fallback. Roughly words + punctuation. Labelled approximate
    everywhere it is used, because an estimate dressed as a measurement is the exact
    failure this project guards against."""
    token = re.compile(r"\w+|[^\w\s]")

    def count(text):
        return len(token.findall(text))

    return count


# ---- segmentation ---------------------------------------------------------------------

def read_front_matter(lines):
    """Return (title, body_start_index). Spark docs open with a --- yaml block."""
    if not lines or not FRONT_MATTER.match(lines[0]):
        return None, 0
    title = None
    for i in range(1, len(lines)):
        if FRONT_MATTER.match(lines[i]):
            return title, i + 1
        m = TITLE.match(lines[i])
        if m and title is None:
            title = m.group(1)
    return title, len(lines)  # unterminated front matter, treat whole file as consumed


def segment(text):
    """Split a document into ordered blocks: heading, code, table or text.

    Kept deliberately linear. A real markdown+HTML parser would be more correct but the
    corpus is regular enough that a state machine over lines is enough. It is also far
    easier to reason about when a chunk comes out wrong.
    """
    lines = text.splitlines()
    title, start = read_front_matter(lines)
    blocks = []
    buf = []          # pending text lines
    i = start

    def flush_text():
        if buf:
            joined = "\n".join(buf).strip()
            if joined:
                blocks.append({"kind": "text", "text": joined})
            buf.clear()

    while i < len(lines):
        line = lines[i]

        if FENCE.match(line):
            flush_text()
            code = [line]
            i += 1
            while i < len(lines) and not FENCE.match(lines[i]):
                code.append(lines[i])
                i += 1
            if i < len(lines):
                code.append(lines[i])  # closing fence
            blocks.append({"kind": "code", "text": "\n".join(code)})
            i += 1
            continue

        if "<table" in line.lower():
            flush_text()
            tbl = [line]
            i += 1
            while i < len(lines) and "</table>" not in lines[i].lower():
                tbl.append(lines[i])
                i += 1
            if i < len(lines):
                tbl.append(lines[i])  # closing tag
            blocks.append({"kind": "table", "text": "\n".join(tbl)})
            i += 1
            continue

        m = HEADING.match(line)
        if m:
            flush_text()
            blocks.append({"kind": "heading", "level": len(m.group(1)), "text": m.group(2)})
            i += 1
            continue

        if not line.strip():
            flush_text()
        else:
            buf.append(line)
        i += 1

    flush_text()
    return title, blocks


# ---- turning blocks into packable units -----------------------------------------------

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


def text_units(block):
    """Sentence-ish units so packing and overlap land on real boundaries, not mid-word."""
    parts = []
    for para in block["text"].split("\n"):
        para = para.strip()
        if not para:
            continue
        # list items and table-ish lines stay whole. prose gets sentence-split
        if para.startswith(("-", "*", "|", ">")) or para[0].isdigit() and para[1:2] == ".":
            parts.append(para)
        else:
            parts.extend(s for s in SENTENCE.split(para) if s)
    return [{"kind": "text", "text": p} for p in parts]


def table_units(block, count, budget):
    """One or more chunks per HTML table, each = header row + as many body rows as fit.

    Repeating the header is the whole point: a fragment holding three property rows is
    still self-describing, where a bare <tr> with a value and no column name is noise.
    """
    rows = TR.findall(block["text"])
    if not rows:
        return [{"kind": "table", "text": block["text"]}]  # malformed, keep as-is

    header = next((r for r in rows if TH.search(r)), None)
    body = [r for r in rows if r is not header]
    header_txt = header.strip() if header else ""
    header_cost = count(header_txt)

    units, cur = [], []
    cur_cost = header_cost
    for row in body:
        rc = count(row)
        if cur and cur_cost + rc > budget:
            units.append(_wrap_rows(header_txt, cur))
            cur, cur_cost = [], header_cost
        cur.append(row.strip())
        cur_cost += rc
    if cur:
        units.append(_wrap_rows(header_txt, cur))
    return units


def _wrap_rows(header_txt, rows):
    inner = "\n".join(([header_txt] if header_txt else []) + rows)
    return {"kind": "table", "text": f"<table>\n{inner}\n</table>"}


# ---- packing --------------------------------------------------------------------------

def hard_split(unit, count, budget, overlap):
    """Last resort for a single unit that exceeds the budget on its own.

    The day-1 plan was "never split a code block, let it be its own chunk". Profiling on
    day 2 killed that: 40 code blocks and 11 tables run past 512 tokens, the worst at
    15,757. bge-small truncates at 512, so an unsplit block would embed its first few
    hundred tokens and silently drop the rest. Worse than a clean window split. So blocks
    that fit stay whole. only the ones that would be truncated get windowed, on the
    coarsest boundary that works: newline, then space, then character. The character floor
    exists because a few markdown table rows are written as one 1,000-token physical line
    with no other break to grab.
    """
    if count(unit["text"]) <= budget:
        return [unit]
    return [{"kind": unit["kind"], "text": t}
            for t in _window(unit["text"], count, budget, overlap)]


def _window(text, count, budget, overlap, seps=("\n", " ", "")):
    for depth, sep in enumerate(seps):
        pieces = list(text) if sep == "" else text.split(sep)
        if len(pieces) <= 1:
            continue
        join = "" if sep == "" else sep
        out, cur, cost = [], [], 0
        for p in pieces:
            c = count(p) or 1
            if c > budget and sep != "":
                # a single piece is still too big. flush and recurse on a finer separator
                if cur:
                    out.append(join.join(cur))
                    cur, cost = [], 0
                out.extend(_window(p, count, budget, overlap, seps[depth + 1:]))
                continue
            if cur and cost + c > budget:
                out.append(join.join(cur))
                keep, kc = [], 0
                for x in reversed(cur):
                    xc = count(x) or 1
                    if kc + xc > overlap:
                        break
                    keep.insert(0, x)
                    kc += xc
                cur, cost = list(keep), kc
            cur.append(p)
            cost += c
        if cur:
            out.append(join.join(cur))
        return out
    return [text]  # single character, genuinely unsplittable


def pack(units, path, count, budget, overlap):
    """Greedy pack a run of units under one heading path into budget-sized chunks.

    The heading path is repeated at the top of every chunk, so it spends part of the
    budget. We reserve that up front and pack content into the room that is left, which is
    why the measured chunk stays under 512 instead of blowing past it once the prefix is
    added. Overlap is applied only across text units. Carrying half a code block or a
    stray table row into the next chunk hurts more than the boundary it protects.
    """
    prefix = path + "\n" if path else ""
    prefix_cost = count(prefix)
    # if a heading path is so long it eats the budget, keep a floor so content still fits.
    # those chunks trip the oversized flag and get counted honestly.
    room = max(budget - prefix_cost, 96)
    tail_budget = min(overlap, room // 4)

    expanded = []
    for u in units:
        expanded.extend(hard_split(u, count, room, tail_budget))

    chunks = []
    cur, cur_cost = [], 0

    def emit():
        if cur:
            body = "\n".join(u["text"] for u in cur)
            kinds = {u["kind"] for u in cur}
            kind = "mixed" if len(kinds) > 1 else kinds.pop()
            n = count(prefix + body)
            chunks.append({
                "heading_path": path,
                "kind": kind,
                "text": prefix + body,
                "n_tokens": n,
                "oversized": n > budget,
            })

    for u in expanded:
        u["n_tokens"] = count(u["text"])
        if cur and cur_cost + u["n_tokens"] > room:
            emit()
            tail = _overlap_tail(cur, count, tail_budget)
            # The overlap tail was being carried into the new chunk without being charged
            # against the unit that forced the flush. A unit is allowed to be as large as
            # room on its own, so tail + unit could reach room + tail_budget. That is the
            # whole story behind the 37 chunks measured 3 to 62 tokens over 512 on day 2,
            # and every one of those overruns is under the 64-token tail budget. Overlap is
            # a nicety. Fitting inside the window the model actually reads is not. When the
            # two conflict the tail loses.
            if sum(t["n_tokens"] for t in tail) + u["n_tokens"] > room:
                tail = []
            cur = list(tail)
            cur_cost = sum(t["n_tokens"] for t in tail)
        cur.append(u)
        cur_cost += u["n_tokens"]
    emit()
    return chunks


def _overlap_tail(units, count, overlap):
    if overlap <= 0:
        return []
    tail, cost = [], 0
    for u in reversed(units):
        if u["kind"] != "text":
            break
        c = u.get("n_tokens") or count(u["text"])
        if cost + c > overlap:
            break
        tail.insert(0, u)
        cost += c
    return tail


def chunk_doc(name, text, count, budget, overlap):
    title, blocks = segment(text)
    stack = []            # (level, heading_text)
    root = [title] if title else []
    chunks = []
    run = []              # content units awaiting a flush at the next heading

    def path():
        return " > ".join(root + [h for _, h in stack])

    def flush_run(p):
        nonlocal run
        if run:
            chunks.extend(pack(run, p, count, budget, overlap))
            run = []

    for b in blocks:
        if b["kind"] == "heading":
            flush_run(path())
            lvl = b["level"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, b["text"]))
        elif b["kind"] == "table":
            run.extend(table_units(b, count, budget))
        elif b["kind"] == "code":
            run.append({"kind": "code", "text": b["text"]})
        else:
            run.extend(text_units(b))
    flush_run(path())

    out = []
    for i, c in enumerate(chunks):
        c["id"] = f"{name}#{i:04d}"
        c["doc"] = name
        out.append(c)
    return out


def run(budget, overlap, which):
    if not CORPUS.is_dir():
        raise SystemExit("no corpus. run: python -m ingest.fetch_corpus")

    count = bge_counter() if which == "bge" else approx_counter()
    docs = [p for p in sorted(CORPUS.glob("*.md")) if is_documentation(p.name)]
    all_chunks = []
    for path in docs:
        all_chunks.extend(chunk_doc(path.name, path.read_text(encoding="utf-8", errors="replace"),
                                    count, budget, overlap))

    OUT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks) + "\n",
                   encoding="utf-8")

    toks = [c["n_tokens"] for c in all_chunks]
    kinds = {}
    for c in all_chunks:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    oversized = sum(1 for c in all_chunks if c["oversized"])

    label = "wordpiece" if which == "bge" else "approx (word+punct heuristic)"
    print(f"{len(all_chunks)} chunks from {len(docs)} docs -> {OUT.name}")
    print(f"token counts: {label}")
    print(f"  min {min(toks)}  median {int(statistics.median(toks))}  "
          f"mean {statistics.mean(toks):.0f}  max {max(toks)}")
    print(f"  over budget ({budget}): {oversized}")
    print("  by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--tokenizer", choices=["bge", "approx"], default="bge")
    args = ap.parse_args()
    run(args.budget, args.overlap, args.tokenizer)


if __name__ == "__main__":
    main()
