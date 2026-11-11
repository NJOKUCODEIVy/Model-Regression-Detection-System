"""Compare two eval runs case by case, measure the change on every dimension, and rate how serious it is.

A regression is a case that passed in the baseline run and fails in the candidate run.
Comparing case by case matters because a better overall score can hide a case that got worse.

Severity comes from the drop in pass rate (percentage points), so a flip or two is treated as noise:
    critical  drop > CRITICAL_DELTA (default 8)  -> FAIL, blocks the merge
    warning   drop > WARNING_DELTA  (default 3)  -> WARN
    ok        smaller drops; any regression or API error still shows as WARN so it gets a look
Optional hard limit: MAX_REGRESSIONS (unset by default) -> FAIL when more cases regress than that.
Each setting can also be passed on the command line (--warning-delta, --critical-delta, --max-regressions).

Usage:
    python -m src.evals.compare --baseline v3 --candidate v4
    python -m src.evals.compare --baseline results/v3_20261110-210512.json --candidate results/v4_20261111-093000.json

The exit code is 1 when the status is FAIL, so CI can block the change.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
WARNING_DELTA = 3.0
CRITICAL_DELTA = 8.0


def env_number(name: str, default, cast=float):
    value = os.getenv(name, "").strip()
    return cast(value) if value else default


def add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """The same threshold options for every command, with environment variables as defaults."""
    parser.add_argument("--warning-delta", type=float, default=env_number("WARNING_DELTA", WARNING_DELTA),
                        help=f"Warn when the pass rate drops more than this many points (default {WARNING_DELTA}, env WARNING_DELTA)")
    parser.add_argument("--critical-delta", type=float, default=env_number("CRITICAL_DELTA", CRITICAL_DELTA),
                        help=f"Fail when the pass rate drops more than this many points (default {CRITICAL_DELTA}, env CRITICAL_DELTA)")
    parser.add_argument("--max-regressions", type=int, default=env_number("MAX_REGRESSIONS", None, int),
                        help="Optional hard limit: fail when more cases regress than this (default: no limit, env MAX_REGRESSIONS)")


def thresholds(args: argparse.Namespace) -> dict:
    return {"warning_delta": args.warning_delta, "critical_delta": args.critical_delta,
            "max_regressions": args.max_regressions}


def find_results(ref: str) -> Path:
    """Accept a results file path, or a prompt version like 'v3' (uses its newest results file)."""
    path = Path(ref)
    if path.suffix == ".json" or "/" in ref:  # a file path, not a version name
        if not path.exists():
            sys.exit(f"Results file not found: {ref}")
        return path
    matches = sorted(RESULTS_DIR.glob(f"{ref}_2*.json"))
    if not matches:
        sys.exit(f"No results found for '{ref}'. Run: python -m src.evals.run_eval --prompt prompts/{ref}.yaml")
    return matches[-1]  # file names end in a timestamp, so the last one is the newest


def pct(passed: int, total: int) -> float:
    return round(100 * passed / total, 1) if total else 0.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def dimensions(rows: list[dict]) -> dict:
    """The non-category dimensions of a run. None where the run didn't record them (older runs)."""
    scores = [r["judge_score"] for r in rows if r.get("judge_score") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    token_keys = ("prompt_tokens", "completion_tokens", "judge_prompt_tokens", "judge_completion_tokens")
    has_tokens = any(r.get("prompt_tokens") is not None for r in rows)
    return {
        "summary_relevance": round(statistics.mean(scores), 2) if scores else None,
        "summaries_scored": len(scores),
        "latency_mean_ms": round(statistics.mean(latencies)) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95),
        "tokens_total": sum((r.get(k) or 0) for r in rows for k in token_keys) if has_tokens else None,
    }


def compare(baseline: dict, candidate: dict, warning_delta: float = WARNING_DELTA,
            critical_delta: float = CRITICAL_DELTA, max_regressions: int | None = None) -> dict:
    base = {c["id"]: c for c in baseline["cases"]}
    cand = {c["id"]: c for c in candidate["cases"]}
    shared = [cid for cid in base if cid in cand]  # only compare cases both runs have

    def entry(cid: str) -> dict:
        b, c = base[cid], cand[cid]
        return {"id": cid, "difficulty": c["difficulty"], "expected": c["expected"],
                "was": b["predicted"], "now": c["predicted"], "error": c["error"],
                "summary_score_was": b.get("judge_score"), "summary_score_now": c.get("judge_score")}

    regressions = [entry(cid) for cid in shared if base[cid]["passed"] and not cand[cid]["passed"]]
    fixes = [entry(cid) for cid in shared if not base[cid]["passed"] and cand[cid]["passed"]]
    still_failing = [entry(cid) for cid in shared if not base[cid]["passed"] and not cand[cid]["passed"]]

    by_category = defaultdict(lambda: {"total": 0, "baseline": 0, "candidate": 0})
    for cid in shared:
        group = by_category[cand[cid]["expected"]]
        group["total"] += 1
        group["baseline"] += base[cid]["passed"]
        group["candidate"] += cand[cid]["passed"]
    for group in by_category.values():
        group["change_pts"] = round(pct(group["candidate"], group["total"]) - pct(group["baseline"], group["total"]), 1)

    n = len(shared)
    base_passed = sum(base[cid]["passed"] for cid in shared)
    cand_passed = sum(cand[cid]["passed"] for cid in shared)
    change = round(pct(cand_passed, n) - pct(base_passed, n), 1)
    drop = -change
    errors = sum(1 for cid in shared if cand[cid].get("error"))

    # Severity from the size of the drop, plus the optional hard limit on regressions
    reasons = []
    if drop > critical_delta:
        severity = "critical"
        reasons.append(f"Pass rate dropped {drop:.1f} pts, more than the {critical_delta:g}-point critical threshold")
    elif drop > warning_delta:
        severity = "warning"
        reasons.append(f"Pass rate dropped {drop:.1f} pts, more than the {warning_delta:g}-point warning threshold")
    else:
        severity = "ok"
    if max_regressions is not None and len(regressions) > max_regressions:
        severity = "critical"
        reasons.append(f"{len(regressions)} regression(s), more than the hard limit of {max_regressions}")
    if severity == "ok" and regressions:
        reasons.append(f"{len(regressions)} case(s) regressed, but the pass-rate change ({change:+.1f} pts) "
                       f"is within the {warning_delta:g}-point noise threshold")
    if errors:
        reasons.append(f"{errors} case(s) failed with API errors in the candidate run")

    if severity == "critical":
        status = "FAIL"
    elif severity == "warning" or regressions or errors:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "compared_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {"prompt_version": baseline["prompt_version"], "run_at": baseline["run_at"],
                     "dataset_version": baseline["dataset_version"], "model": baseline.get("model"),
                     "judge_model": baseline.get("judge_model")},
        "candidate": {"prompt_version": candidate["prompt_version"], "run_at": candidate["run_at"],
                      "dataset_version": candidate["dataset_version"], "model": candidate.get("model"),
                      "judge_model": candidate.get("judge_model")},
        "same_dataset": baseline["dataset_version"] == candidate["dataset_version"],
        "shared_cases": n,
        "only_in_baseline": [cid for cid in base if cid not in cand],
        "only_in_candidate": [cid for cid in cand if cid not in base],
        "baseline_passed": base_passed,
        "candidate_passed": cand_passed,
        "accuracy_change_pts": change,
        "regressions": regressions,
        "fixes": fixes,
        "still_failing": still_failing,
        "candidate_errors": errors,
        "by_category": dict(by_category),
        "dimensions": {"baseline": dimensions([base[cid] for cid in shared]),
                       "candidate": dimensions([cand[cid] for cid in shared])},
        "thresholds": {"warning_delta": warning_delta, "critical_delta": critical_delta,
                       "max_regressions": max_regressions},
        "severity": severity,
        "status": status,
        "reasons": reasons,
    }


def fmt(value, unit: str = "", digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}{unit}"


def dimension_rows(result: dict) -> list[tuple[str, str, str, str]]:
    """(label, baseline, candidate, change) for each extra dimension, for printing or reports."""
    b, c = result["dimensions"]["baseline"], result["dimensions"]["candidate"]
    rows = []
    for label, key, unit, digits in [("Summary relevance", "summary_relevance", "/5", 2),
                                     ("Latency, mean", "latency_mean_ms", " ms", 0),
                                     ("Latency, p95", "latency_p95_ms", " ms", 0),
                                     ("Tokens, total", "tokens_total", "", 0)]:
        change = "" if b[key] is None or c[key] is None else f"{c[key] - b[key]:+,.{digits}f}"
        rows.append((label, fmt(b[key], unit, digits), fmt(c[key], unit, digits), change))
    return rows


def describe(case: dict) -> str:
    now = f"error: {case['error'][:60]}" if case["error"] else case["now"]
    return f"  {case['id']} ({case['difficulty']}): expected {case['expected']}, was {case['was']}, now {now}"


def print_comparison(result: dict) -> None:
    b, c = result["baseline"], result["candidate"]
    n = result["shared_cases"]
    print(f"\nComparing {b['prompt_version']} -> {c['prompt_version']} on {n} shared cases")

    if not result["same_dataset"]:
        print(f"\n  WARNING: different dataset versions (v{b['dataset_version']} vs v{c['dataset_version']}).")
        print("  Scores are only fully comparable on the same dataset version.")
    if result["only_in_baseline"] or result["only_in_candidate"]:
        print(f"  Skipped {len(result['only_in_baseline'])} case(s) only in the baseline and "
              f"{len(result['only_in_candidate'])} only in the candidate.")

    bp, cp = result["baseline_passed"], result["candidate_passed"]
    print(f"\nPass rate: {bp}/{n} ({pct(bp, n)}%) -> {cp}/{n} ({pct(cp, n)}%)  "
          f"({result['accuracy_change_pts']:+.1f} pts)")

    sections = [("REGRESSIONS (passed before, fail now)", result["regressions"]),
                ("FIXES (failed before, pass now)", result["fixes"]),
                ("STILL FAILING", result["still_failing"])]
    for title, cases in sections:
        print(f"\n{title}: {len(cases)}")
        for case in cases:
            print(describe(case))

    print(f"\nBy category:      {b['prompt_version']:>7}  {c['prompt_version']:>7}   change")
    for name, g in sorted(result["by_category"].items()):
        change = f"{g['change_pts']:+.1f} pts" if g["change_pts"] else ""
        print(f"  {name:<14} {g['baseline']:>3}/{g['total']:<3}  {g['candidate']:>3}/{g['total']:<3}  {change}")

    print(f"\nOther dimensions:   {b['prompt_version']:>10}  {c['prompt_version']:>10}   change")
    for label, before, after, change in dimension_rows(result):
        print(f"  {label:<17} {before:>10}  {after:>10}   {change}")

    t = result["thresholds"]
    limit = f", hard limit {t['max_regressions']} regressions" if t["max_regressions"] is not None else ""
    print(f"\nResult: {result['status']} (severity: {result['severity']}; warning above {t['warning_delta']:g} pts, "
          f"critical above {t['critical_delta']:g} pts{limit})")
    for reason in result["reasons"]:
        print(f"  - {reason}")


def save_comparison(result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"compare_{result['baseline']['prompt_version']}_vs_{result['candidate']['prompt_version']}_{stamp}.json"
    path = RESULTS_DIR / name
    path.write_text(json.dumps(result, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two eval runs and rate the change.")
    parser.add_argument("--baseline", required=True, help="Prompt version (e.g. v3) or path to a results file")
    parser.add_argument("--candidate", required=True, help="Prompt version (e.g. v4) or path to a results file")
    add_threshold_args(parser)
    args = parser.parse_args()

    base_path, cand_path = find_results(args.baseline), find_results(args.candidate)
    print(f"Baseline:  {base_path}\nCandidate: {cand_path}")
    result = compare(json.loads(base_path.read_text()), json.loads(cand_path.read_text()), **thresholds(args))
    print_comparison(result)
    print(f"\nSaved comparison to {save_comparison(result)}")
    sys.exit(1 if result["status"] == "FAIL" else 0)