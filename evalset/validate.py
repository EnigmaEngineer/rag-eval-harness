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


# Prose claims, and why this is a report rather than a check.
#
# q005's reference answer says a broadcast join "requires the smaller relation to fit in
# driver and executor memory". Neither cited doc says that. Measured: "driver memory" and
# "fit in driver" both return 0 hits across sql-performance-tuning.md and tuning.md. The
# grounding check above missed it because it only verifies config keys and declared spans,
# and q005 has no spans while its one config key is present.
#
# The obvious fix is to check prose claims the same mechanical way. It was built and
# measured and it does not work. Scoring each claim by the fraction of its content words
# found in the best single paragraph of a cited doc puts q005 at 0.50 and puts "Client mode
# suits interactive work, cluster mode suits production jobs" at 0.40. The second one is
# correct and well supported. It is a paraphrase, and paraphrase is what the metric cannot
# see. No threshold isolates the one real defect because the real defect does not score
# lowest.
#
# So this exits 0 and prints. It is for the person writing question 11 through 60, who
# should look at whatever sits at the bottom of the list before freezing the labels. Making
# it a gate would train everyone to pass it, which on this evidence means deleting correct
# sentences.

CLAUSE_SPLIT = re.compile(r"(?<=[.!?])\s+|;\s*")
WORD = re.compile(r"[a-z0-9]+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# Deliberately small. A real stopword list would start deleting Spark vocabulary.
STOPWORDS = frozenset("""
a an and are as at be by can does do for from has have if in into is it its of on or that
the to use used using with when what which while you your not no than then them they this
these those be been being will would should could may might there here also
""".split())


def claim_clauses(answer):
    """Split an answer into clauses that assert something checkable.

    Splits on semicolons as well as sentence enders. q005's claim and its config key sit in
    one sentence joined by a semicolon, so a sentence-level split hides the claim behind the
    key and reports the whole thing as grounded.
    """
    out = []
    for part in CLAUSE_SPLIT.split(answer):
        part = part.strip()
        if not part or CONFIG_KEY.search(part):
            continue
        if len(content_words(part)) < 3:
            continue
        out.append(part)
    return out


def content_words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2]


def best_paragraph(claim, paragraphs):
    """Highest content-word coverage over any single paragraph, and that paragraph.

    Single paragraph, not the whole document. A claim assembled from words scattered across
    a 4000 line reference page is exactly the failure being looked for, and scoring against
    the union of the doc would score it 1.0.
    """
    words = content_words(claim)
    if not words:
        return 0.0, ""
    best, best_para = 0.0, ""
    for para in paragraphs:
        tokens = set(WORD.findall(para.lower()))
        cover = sum(1 for w in words if w in tokens) / len(words)
        if cover > best:
            best, best_para = cover, para
    return best, best_para


def claim_report(rows, corpus_dir=CORPUS):
    """One tuple per prose claim. Fields are coverage, qid, claim and best paragraph.

    Sorted ascending so the weakest support comes first.
    """
    if not corpus_dir.is_dir():
        return []

    cache = {}
    found = []
    for _lineno, row in rows:
        paragraphs = []
        for doc in row["source_docs"]:
            if doc not in cache:
                path = corpus_dir / doc
                text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
                cache[doc] = [p for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
            paragraphs.extend(cache[doc])

        for claim in claim_clauses(row["answer"]):
            cover, para = best_paragraph(claim, paragraphs)
            found.append((cover, row["id"], claim, para))
    return sorted(found)


def print_claim_report(rows):
    report = claim_report(rows)
    if not report:
        print("no prose claims to report (corpus not fetched, or every clause names a config key)")
        return
    print(f"\n{len(report)} prose claims, weakest support first.")
    print("Advisory only. Read the top of this list before freezing labels.\n")
    for cover, qid, claim, para in report:
        first_line = " ".join(para.split())[:90] if para else "(nothing matched)"
        print(f"  {cover:.2f}  {qid}  {claim[:78]}")
        print(f"        best paragraph: {first_line}")


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

    if "--claims" in sys.argv[1:]:
        print_claim_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
