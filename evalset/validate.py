"""Checks golden.jsonl before anything downstream trusts it.

Run after fetching the corpus:
    python -m evalset.validate
"""

import json
import re
import sys
from pathlib import Path

REQUIRED = {"id", "question", "source_docs", "answer", "difficulty"}
OPTIONAL = {"answer_spans", "failure_mode"}
DIFFICULTIES = {"easy", "medium", "hard"}

GOLDEN = Path(__file__).parent / "golden.jsonl"
CORPUS = Path(__file__).parent.parent / "data" / "corpus"


def load(path=GOLDEN):
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append((lineno, json.loads(line)))
        except json.JSONDecodeError as err:
            raise SystemExit(f"{path.name}:{lineno} is not valid JSON: {err}")
    return rows


def validate(rows, corpus_dir=CORPUS):
    problems = []
    seen = set()
    # Only check filenames against the corpus if it has actually been fetched.
    # Failing on a missing corpus would make this unusable before ingest runs.
    known = {p.name for p in corpus_dir.glob("*.md")} if corpus_dir.is_dir() else None

    for lineno, row in rows:
        where = f"line {lineno}"
        missing = REQUIRED - row.keys()
        if missing:
            problems.append(f"{where}: missing {sorted(missing)}")
            continue

        unknown = row.keys() - REQUIRED - OPTIONAL
        if unknown:
            problems.append(f"{where}: unexpected fields {sorted(unknown)}")

        if row["id"] in seen:
            problems.append(f"{where}: duplicate id {row['id']}")
        seen.add(row["id"])

        if row["difficulty"] not in DIFFICULTIES:
            problems.append(f"{where}: difficulty {row['difficulty']!r} not in {sorted(DIFFICULTIES)}")

        if not row["source_docs"]:
            problems.append(f"{where}: source_docs is empty")
        elif known is not None:
            for doc in row["source_docs"]:
                if doc not in known:
                    problems.append(f"{where}: source_doc {doc!r} is not in the corpus")

    return problems, known


# Spark config properties are exact strings, which makes them a cheap way to check that
# an answer really lives in the docs it claims to. Catches a mislabelled source_doc, which
# would otherwise look like a retrieval failure on day 5.
CONFIG_KEY = re.compile(r"spark\.[a-zA-Z0-9._]+[a-zA-Z0-9]")


def check_answers_grounded(rows, corpus_dir=CORPUS):
    """Verify config keys and answer_spans appear in the cited source docs."""
    if not corpus_dir.is_dir():
        return ["corpus not fetched, cannot check grounding"]

    cache = {}
    problems = []

    for lineno, row in rows:
        text = ""
        for doc in row["source_docs"]:
            if doc not in cache:
                path = corpus_dir / doc
                cache[doc] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            text += cache[doc]

        for key in set(CONFIG_KEY.findall(row["answer"])):
            if key not in text:
                problems.append(
                    f"line {lineno} ({row['id']}): answer cites {key} but it is not in "
                    f"{row['source_docs']}"
                )

        for span in row.get("answer_spans", []):
            if span.lower() not in text.lower():
                problems.append(f"line {lineno} ({row['id']}): span {span!r} not found in source docs")

    return problems


def main():
    rows = load()
    problems, known = validate(rows)

    if known is None:
        print("corpus not fetched yet, skipping filename checks "
              "(run ingest/fetch_corpus.py first)")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(" ", p)
        return 1

    grounding = check_answers_grounded(rows)
    if grounding:
        print(f"{len(grounding)} grounding problem(s):")
        for g in grounding:
            print(" ", g)
        return 1

    by_difficulty = {}
    for _, row in rows:
        by_difficulty[row["difficulty"]] = by_difficulty.get(row["difficulty"], 0) + 1
    print(f"{len(rows)} questions, all valid")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(by_difficulty.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
