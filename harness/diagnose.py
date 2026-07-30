"""Where two metrics disagree about the same question, one of them is wrong.

    python -m harness.diagnose

`support@5` and `hit@10` are supposed to be loosely coupled. `hit@10` asks whether the
labelled gold chunk was retrieved. `support@5` asks whether the strings a correct answer
has to point at appear anywhere in the top-5 text. A question can reasonably fail one and
pass the other. What is not reasonable is for that to keep happening quietly.

Each direction means something different and both are worth catching.

**support 1.0 with hit 0** says the required strings turned up but the labelled evidence did
not. Either the gold set is under-complete, or the required strings are so weak that an
unrelated chunk satisfies them. On this eval set it is the second one, which makes
`support@5` the metric that lied rather than the label.

**support 0 with hit 1** says the labelled chunk was retrieved and the required strings
still were not found in the top 5. Usually the required string is spelled differently in the
docs than in the reference answer.

This file exists because I read the day-6 ablation table for two days without noticing that
q005 scored `support@5` 1.0 on all four systems while every one of them missed the answer.
A table of means hides that. A per-question disagreement check does not.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "reports" / "results.jsonl"


def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def disagreements(rows, k=10):
    """Rows where support@5 and hit@k point in opposite directions.

    Returns one record per (qid, system) rather than per question. A disagreement on one
    system out of four is a different problem from a disagreement on all four. The first
    usually means that system retrieved badly. The second means the eval set is wrong.
    """
    out = []
    for r in rows:
        hit = bool(set(r["gold"]) & set(r["ranked"][:k]))
        support = r["support"]
        if support >= 1.0 and not hit:
            kind = "support without evidence"
        elif support <= 0.0 and hit:
            kind = "evidence without support"
        else:
            continue
        out.append({
            "qid": r["qid"],
            "system": r["system"],
            "kind": kind,
            "support": support,
            f"hit@{k}": hit,
            "required": r.get("required", []),
        })
    return out


def by_question(records, n_systems):
    """Group the records so a whole-set defect is visible as one line, not four."""
    grouped = {}
    for rec in records:
        key = (rec["qid"], rec["kind"])
        grouped.setdefault(key, []).append(rec["system"])
    rows = []
    for (qid, kind), systems in sorted(grouped.items()):
        rows.append({
            "qid": qid,
            "kind": kind,
            "systems": sorted(systems),
            "all_systems": len(systems) == n_systems,
        })
    return rows


def thin_anchors(rows, threshold=1):
    """Questions whose support@5 rests on one string or none.

    Not a disagreement, a power problem. With a single required string `support@5` is a
    one-bit metric for that question, so it cannot distinguish a chunk that answers the
    question from a chunk that mentions the key in passing.
    """
    seen = {}
    for r in rows:
        seen.setdefault(r["qid"], r.get("required", []))
    return sorted(q for q, req in seen.items() if len(req) <= threshold)


def main():
    ap = argparse.ArgumentParser(description="find questions where the metrics disagree")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any disagreement holds across every system")
    args = ap.parse_args()

    rows = load_rows(args.results)
    if not rows:
        print(f"no rows in {args.results}")
        return 1

    systems = {r["system"] for r in rows}
    questions = {r["qid"] for r in rows}
    records = disagreements(rows, k=args.k)
    grouped = by_question(records, len(systems))

    print(f"{len(questions)} questions, {len(systems)} systems, {len(rows)} rows")

    if not grouped:
        print(f"no support@5 / hit@{args.k} disagreements")
    else:
        print(f"\n{len(grouped)} disagreement(s):\n")
        print("| qid | kind | systems |")
        print("|---|---|---|")
        for g in grouped:
            scope = "all" if g["all_systems"] else ", ".join(g["systems"])
            print(f"| {g['qid']} | {g['kind']} | {scope} |")

    thin = thin_anchors(rows)
    if thin:
        print(f"\n{len(thin)} of {len(questions)} questions rest on one required string "
              f"or none: {', '.join(thin)}")
        print("support@5 is close to a one-bit metric on those.")

    whole_set = [g for g in grouped if g["all_systems"]]
    if whole_set:
        print(f"\n{len(whole_set)} disagreement(s) hold on every system, so the eval set "
              f"is the suspect and not the retriever: "
              f"{', '.join(g['qid'] for g in whole_set)}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
