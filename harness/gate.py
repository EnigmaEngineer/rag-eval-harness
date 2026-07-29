"""Regression gate. Compares the current eval report against a committed baseline and
exits nonzero if a tracked metric has dropped by more than its tolerance.

    python -m harness.gate check
    python -m harness.gate check --systems bm25
    python -m harness.gate update --note "rebuilt corpus, meta files filtered"

The point is not to make CI green. It is to make a silent retrieval regression impossible
to merge without someone typing a reason. A chunker tweak that looks harmless can move
recall by a tenth and nobody notices until the answers get worse in production.

Tolerances are not uniform, because the metrics are not equally noisy. The golden set is ten
questions, so one question moving is 0.1 of recall@1. Anything tighter than that would fail
on noise and get switched off within a week, which is worse than no gate. Latency gets a
proportional band instead of an absolute one because it varies with the machine CI happens
to schedule.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from harness import run_eval

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / "reports" / "baseline.json"

# metric -> how far it is allowed to fall before this fails. Quality metrics are absolute
# drops. One question out of ten is 0.100, so 0.050 catches a real change without firing on
# a single borderline question flipping.
TOLERANCE = {
    "recall@1": 0.05,
    "recall@5": 0.05,
    "recall@10": 0.05,
    "hit@1": 0.10,
    "hit@5": 0.10,
    "mrr@10": 0.05,
    "support@5": 0.10,
}

# Latency is allowed to grow by this factor before it counts as a regression. A CI runner is
# not this laptop and never will be, so an absolute millisecond budget here would be fiction.
LATENCY_FACTOR = 2.0


def current(systems=None, results=None):
    rows = run_eval.load_jsonl(results or run_eval.RESULTS)
    if systems:
        rows = [r for r in rows if r["system"] in systems]
    if not rows:
        raise SystemExit("no results to check. run: python -m harness.run_eval run")
    return run_eval.summarise(rows)


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def update(note, results=None):
    stats = current(results=results)
    n_q = len(run_eval.load_questions())
    partial = {s: v["n"] for s, v in stats.items() if v["n"] != n_q}
    if partial:
        raise SystemExit(f"refusing to baseline a partial run: {partial}, expected {n_q} each")
    payload = {
        "note": note,
        "commit": git_sha(),
        "questions": n_q,
        "systems": stats,
    }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"baseline written for {len(stats)} systems over {n_q} questions -> "
          f"{BASELINE.relative_to(ROOT)}")
    print(f"  note: {note}")


def check(systems=None, results=None):
    if not BASELINE.exists():
        raise SystemExit("no baseline. run: python -m harness.gate update --note '...'")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    now = current(systems, results)

    failures = []
    skipped = []
    lines = []

    for system in run_eval.SYSTEMS:
        if system not in now:
            continue
        if system not in base["systems"]:
            skipped.append(system)
            continue
        b, c = base["systems"][system], now[system]
        if c["n"] != b["n"]:
            # comparing a partial run to a full baseline compares different question sets
            failures.append(f"{system}: {c['n']} questions against a baseline of {b['n']}")
            continue
        for metric, tol in TOLERANCE.items():
            drop = b[metric] - c[metric]
            mark = "ok"
            if drop > tol:
                mark = "FAIL"
                failures.append(
                    f"{system} {metric}: {b[metric]:.3f} -> {c[metric]:.3f} "
                    f"(down {drop:.3f}, tolerance {tol:.3f})")
            elif drop < -tol:
                mark = "up"
            lines.append(f"  {system:7} {metric:10} {b[metric]:.3f} -> {c[metric]:.3f}  {mark}")
        if c["ms"] > b["ms"] * LATENCY_FACTOR and c["ms"] - b["ms"] > 5:
            failures.append(f"{system} ms: {b['ms']:.0f} -> {c['ms']:.0f} "
                            f"(over {LATENCY_FACTOR:.0f}x the baseline)")

    print("\n".join(lines))
    for system in skipped:
        print(f"  {system}: not in the baseline, nothing to compare")

    if failures:
        print("\nREGRESSION")
        for f in failures:
            print(f"  {f}")
        print("\nIf this is intended, re-baseline with a reason:")
        print("  python -m harness.gate update --note 'why the numbers moved'")
        return 1

    print("\nno regression")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("check")
    cp.add_argument("--systems", default="", help="comma separated, default all present")
    cp.add_argument("--results", default=str(run_eval.RESULTS),
                    help="results file to check, matches run_eval --out")
    up = sub.add_parser("update")
    up.add_argument("--note", required=True, help="why the baseline moved")
    args = ap.parse_args()

    if args.cmd == "check":
        systems = [s.strip() for s in args.systems.split(",") if s.strip()] or None
        sys.exit(check(systems, Path(args.results)))
    else:
        update(args.note)


if __name__ == "__main__":
    main()
