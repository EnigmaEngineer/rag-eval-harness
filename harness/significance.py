"""Is the gap between two systems real, or is it one question?

The ablation table has fused beating bm25 on recall@1 by 0.050. On ten questions that is
half a question. Day 5 and day 6 both looked at that number and both refused to decide,
which is the correct instinct and a bad way to run a project. This module replaces the
instinct with an interval.

    python -m harness.significance compare --a fused --b bm25
    python -m harness.significance sweep

Two tests, and they answer different questions.

**Exact paired permutation** gives the p-value. Under the null the two systems are
interchangeable, so for any question you could swap which system produced which score and
the world would look the same. That means flipping the sign of a paired difference is a
valid relabelling. With ten questions there are 2**10 = 1024 sign assignments, so we
enumerate every one of them instead of sampling. No seed, no approximation, no "10000
iterations" that somebody has to trust. A ten-question eval set is bad for almost
everything and this is the one thing it makes easier.

**Paired bootstrap** gives the confidence interval. The p-value says whether to believe the
sign of the gap. The interval says how big the gap could be, which is the number you
actually need when deciding whether to keep a component. Resampling questions is sampled
rather than exhaustive, so this one takes a seed.

Both are paired on qid. The systems answer the same ten questions against the same labels,
so pairing removes question difficulty from the comparison. Treating the two score lists as
independent samples would throw that away and widen everything for no reason.

Latency is included deliberately. If every comparison comes back non-significant you cannot
tell an underpowered test from a broken one. bm25 at 1 ms against fused at 23 ms is a real
difference on any sane test, so it works as a positive control on this file.
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from harness import metrics

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "reports" / "results.jsonl"

SYSTEMS = ["dense", "bm25", "fused", "rerank"]
METRICS = ["recall@1", "recall@5", "recall@10", "hit@1", "hit@5", "mrr@10", "support@5", "ms"]

# 1024 permutations is the whole space for n=10. Above this we would have to sample, and
# the honest thing is to say so rather than silently switch method.
EXACT_LIMIT = 20


def per_question(rows):
    """{system: {qid: {metric: value}}} from the raw results rows.

    Deliberately recomputes from `ranked` and `gold` using harness.metrics rather than
    reading a summary. The summary is a mean and a mean cannot be un-averaged.
    """
    out = {}
    for r in rows:
        vals = {}
        for k in (1, 5, 10):
            vals[f"recall@{k}"] = metrics.recall_at_k(r["ranked"], r["gold"], k)
            vals[f"hit@{k}"] = float(metrics.hit_at_k(r["ranked"], r["gold"], k))
        vals["mrr@10"] = metrics.reciprocal_rank(r["ranked"], r["gold"], k=10)
        vals["support@5"] = r["support"]
        vals["ms"] = r["seconds"] * 1000
        out.setdefault(r["system"], {})[r["qid"]] = vals
    return out


def paired(scores, a, b, metric):
    """Differences a - b over the questions both systems answered, ordered by qid.

    Missing pairs are dropped rather than filled. A half-finished `run_eval` should not
    silently become a comparison over a different question set.
    """
    qids = sorted(set(scores.get(a, {})) & set(scores.get(b, {})))
    return qids, [scores[a][q][metric] - scores[b][q][metric] for q in qids]


def permutation_p(diffs):
    """Two-sided exact p-value by enumerating every sign flip.

    Returns None when the space is too large to enumerate, so the caller reports "not
    computed" rather than a number from a different method wearing the same label.
    """
    n = len(diffs)
    if n == 0 or n > EXACT_LIMIT:
        return None
    observed = abs(sum(diffs))
    # Count |sum| >= observed rather than > it. The unflipped assignment is itself one of
    # the 1024 and has to be counted, which is why no exact p-value can be below 1/1024.
    extreme = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12:
            extreme += 1
    return extreme / (2 ** n)


def p_floor(diffs):
    """Smallest p-value this comparison could possibly return, whatever the effect size.

    A tie is invariant under a sign flip, so questions where the two systems scored the same
    contribute nothing to the permutation distribution. Only the k questions that actually
    differ matter, which leaves 2**k distinguishable assignments. The most extreme arrangement
    is reached twice, by the observed one and its mirror, so no p can be below 2 / 2**k.

    This is the whole argument about eval set size in one number. On the real table fusion
    beats bm25 on recall@1 by 0.050, and exactly one question is responsible. k=1 puts the
    floor at 1.0, so that comparison cannot return a significant result at any threshold no
    matter how large the gap gets. The fix is more questions, not a bigger difference.
    """
    k = sum(1 for d in diffs if abs(d) > 1e-12)
    if k == 0:
        return 1.0
    return 2 / (2 ** k)


def bootstrap_ci(diffs, reps=10000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for the mean paired difference.

    Resamples questions, not scores. The unit of uncertainty here is which questions ended
    up in the golden set, and that is the thing a wider set would change.
    """
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * reps)]
    hi = means[min(int((1 - alpha / 2) * reps), reps - 1)]
    return lo, hi


def compare(scores, a, b, metric, seed=0):
    qids, diffs = paired(scores, a, b, metric)
    n = len(diffs)
    mean_diff = sum(diffs) / n if n else float("nan")
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return {
        "metric": metric,
        "n": n,
        "diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "p": permutation_p(diffs),
        "p_floor": p_floor(diffs),
        "crosses_zero": lo <= 0 <= hi,
        "questions_moved": sum(1 for d in diffs if abs(d) > 1e-12),
        "qids": qids,
    }


def _fmt_p(p):
    if p is None:
        return "  n/a"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def print_table(rows, a, b):
    print(f"\n{a} minus {b}, paired over {rows[0]['n']} questions")
    print("exact permutation p, and a 95% percentile bootstrap interval on the difference\n")
    print("| metric | diff | 95% CI | p | best p possible | questions moved |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        prec = 0 if r["metric"] == "ms" else 3
        ci = f"{r['ci_low']:+.{prec}f} to {r['ci_high']:+.{prec}f}"
        print(f"| {r['metric']} | {r['diff']:+.{prec}f} | {ci} | {_fmt_p(r['p'])} | "
              f"{_fmt_p(r['p_floor'])} | {r['questions_moved']}/{r['n']} |")

    capped = [r["metric"] for r in rows if r["p_floor"] > 0.05]
    if capped:
        print(f"\n{len(capped)} metrics cannot reach p<.05 at any effect size, because too "
              f"few questions differ: {', '.join(capped)}")

    undecided = [r["metric"] for r in rows if r["crosses_zero"]]
    if undecided:
        print(f"\ninterval includes zero on {len(undecided)} of {len(rows)} metrics: "
              f"{', '.join(undecided)}")

    # A percentile bootstrap on ten questions is happy to exclude zero on four moved
    # questions, where the permutation test says no p below 0.125 is reachable. Printing
    # "excludes zero" next to "cannot reach p<.05" invites the reader to take the half of
    # the output that agrees with them. So the interval only gets to decide something on a
    # metric that could have been significant in the first place.
    decided = [r["metric"] for r in rows
               if not r["crosses_zero"] and r["p_floor"] <= 0.05]
    if decided:
        print(f"interval excludes zero on: {', '.join(decided)}")
    overruled = [r["metric"] for r in rows
                 if not r["crosses_zero"] and r["p_floor"] > 0.05]
    if overruled:
        print(f"interval excludes zero but too few questions moved to believe it: "
              f"{', '.join(overruled)}")


def load_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="significance testing for the ablation table")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("compare", help="one pair of systems across every metric")
    cp.add_argument("--a", default="fused")
    cp.add_argument("--b", default="bm25")
    cp.add_argument("--results", default=str(RESULTS))
    cp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("sweep", help="every system against bm25 on the headline metrics")
    sp.add_argument("--baseline", default="bm25")
    sp.add_argument("--results", default=str(RESULTS))
    sp.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    scores = per_question(load_rows(args.results))

    if args.cmd == "compare":
        missing = [s for s in (args.a, args.b) if s not in scores]
        if missing:
            print(f"no rows for {missing} in {args.results}")
            return 1
        rows = [compare(scores, args.a, args.b, m, seed=args.seed) for m in METRICS]
        print_table(rows, args.a, args.b)
        return 0

    any_decided = False
    for system in SYSTEMS:
        if system == args.baseline or system not in scores:
            continue
        rows = [compare(scores, system, args.baseline, m, seed=args.seed) for m in METRICS]
        print_table(rows, system, args.baseline)
        any_decided |= any(not r["crosses_zero"] for r in rows)
    if not any_decided:
        print("\nnothing separated from the baseline on any metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
