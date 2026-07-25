"""Profiles the fetched corpus so chunking decisions rest on numbers.

    python -m ingest.corpus_stats
    python -m ingest.corpus_stats --top 15
"""

import argparse
import re
from pathlib import Path

CORPUS = Path(__file__).parent.parent / "data" / "corpus"

FENCE = re.compile(r"^```", re.M)
HEADING = re.compile(r"^#{1,6} ", re.M)
MD_TABLE_ROW = re.compile(r"^\|", re.M)


def profile(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": path.name,
        "bytes": len(text),
        "headings": len(HEADING.findall(text)),
        "fenced_blocks": len(FENCE.findall(text)) // 2,
        "html_tables": text.count("<table"),
        "html_rows": text.count("<tr>"),
        "md_table_rows": len(MD_TABLE_ROW.findall(text)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if not CORPUS.is_dir():
        raise SystemExit("no corpus. run: python -m ingest.fetch_corpus")

    rows = [profile(p) for p in sorted(CORPUS.glob("*.md"))]

    total = {k: sum(r[k] for r in rows) for k in
             ("bytes", "headings", "fenced_blocks", "html_tables", "html_rows", "md_table_rows")}

    print(f"{len(rows)} files, {total['bytes'] / 1024:.0f} KB")
    print(f"  headings          {total['headings']}")
    print(f"  fenced code       {total['fenced_blocks']}")
    print(f"  html <table>      {total['html_tables']}")
    print(f"  html <tr>         {total['html_rows']}")
    print(f"  markdown | rows   {total['md_table_rows']}")

    # The chunker has to survive these, so they are worth naming individually.
    print(f"\ntop {args.top} by html table rows:")
    for r in sorted(rows, key=lambda r: -r["html_rows"])[:args.top]:
        if r["html_rows"]:
            print(f"  {r['name']:46} rows={r['html_rows']:4} tables={r['html_tables']:3} {r['bytes'] / 1024:6.0f} KB")

    print(f"\ntop {args.top} by size:")
    for r in sorted(rows, key=lambda r: -r["bytes"])[:args.top]:
        print(f"  {r['name']:46} {r['bytes'] / 1024:6.0f} KB  headings={r['headings']:4}")


if __name__ == "__main__":
    main()
