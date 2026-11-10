"""Compare two eval runs case by case and flag regressions.

A regression is a case that passed in the baseline run and fails in the candidate run.
Comparing case by case matters because a better overall score can hide a case that got worse.

Usage:
    python -m src.evals.compare --baseline v3 --candidate v4
    python -m src.evals.compare --baseline results/v3_20261110-210512.json --candidate results/v4_20261111-093000.json

The exit code is 1 when regressions are found or accuracy drops, so CI can block a bad prompt.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def find_results(ref: str) -> Path:
    """Accept a results file path, or a prompt version like 'v3' (uses its newest results file)."""
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return path
    matches = sorted(RESULTS_DIR.glob(f"{ref}_2*.json"))
    if not matches:
        sys.exit(f"No results found for '{ref}'. Run: python -m src.evals.run_eval --prompt prompts/{ref}.yaml")
    return matches[-1]  # file names end in a timestamp, so the last one is the newest


def pct(passed: int, total: int) -> float:
    return round(100 * passed / total, 1) if total else 0.0


def compare(baseline: dict, candidate: dict, max_regressions: int = 0) -> dict:
    base = {c["id"]: c for c in baseline["cases"]}
    cand = {c["id"]: c for c in candidate["cases"]}
    shared = [cid for cid in base if cid in cand]  # only compare cases both runs have

    def entry(cid: str) -> dict:
        b, c = base[cid], cand[cid]
        return {"id": cid, "difficulty": c["difficulty"], "expected": c["expected"],
                "was": b["predicted"], "now": c["predicted"], "error": c["error"]}

    regressions = [entry(cid) for cid in shared if base[cid]["passed"] and not cand[cid]["passed"]]
    fixes = [entry(cid) for cid in shared if not base[cid]["passed"] and cand[cid]["passed"]]
    still_failing = [entry(cid) for cid in shared if not base[cid]["passed"] and not cand[cid]["passed"]]

    by_category = defaultdict(lambda: {"total": 0, "baseline": 0, "candidate": 0})
    for cid in shared:
        group = by_category[cand[cid]["expected"]]
        group["total"] += 1
        group["baseline"] += base[cid]["passed"]
        group["candidate"] += cand[cid]["passed"]

    base_passed = sum(base[cid]["passed"] for cid in shared)
    cand_passed = sum(cand[cid]["passed"] for cid in shared)
    regressed = len(regressions) > max_regressions or cand_passed < base_passed

    return {
        "compared_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {"prompt_version": baseline["prompt_version"], "run_at": baseline["run_at"],
                     "dataset_version": baseline["dataset_version"]},
        "candidate": {"prompt_version": candidate["prompt_version"], "run_at": candidate["run_at"],
                      "dataset_version": candidate["dataset_version"]},
        "same_dataset": baseline["dataset_version"] == candidate["dataset_version"],
        "shared_cases": len(shared),
        "only_in_baseline": [cid for cid in base if cid not in cand],
        "only_in_candidate": [cid for cid in cand if cid not in base],
        "baseline_passed": base_passed,
        "candidate_passed": cand_passed,
        "accuracy_change_pts": round(pct(cand_passed, len(shared)) - pct(base_passed, len(shared)), 1),
        "regressions": regressions,
        "fixes": fixes,
        "still_failing": still_failing,
        "by_category": dict(by_category),
        "max_regressions_allowed": max_regressions,
        "verdict": "REGRESSION" if regressed else "PASS",
    }


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
    print(f"\nAccuracy: {bp}/{n} ({pct(bp, n)}%) -> {cp}/{n} ({pct(cp, n)}%)  "
          f"({result['accuracy_change_pts']:+.1f} pts)")

    sections = [("REGRESSIONS (passed before, fail now)", result["regressions"]),
                ("FIXES (failed before, pass now)", result["fixes"]),
                ("STILL FAILING", result["still_failing"])]
    for title, cases in sections:
        print(f"\n{title}: {len(cases)}")
        for case in cases:
            print(describe(case))

    print(f"\nBy category:      {b['prompt_version']:>7}  {c['prompt_version']:>7}")
    for name, g in sorted(result["by_category"].items()):
        change = g["candidate"] - g["baseline"]
        marker = f"{change:+d}" if change else ""
        print(f"  {name:<14} {g['baseline']:>3}/{g['total']:<3}  {g['candidate']:>3}/{g['total']:<3}  {marker}")

    if result["verdict"] == "REGRESSION":
        print(f"\nResult: REGRESSION DETECTED ({len(result['regressions'])} case(s) got worse)")
    elif result["regressions"]:
        print(f"\nResult: PASS ({len(result['regressions'])} regression(s), within the allowed "
              f"limit of {result['max_regressions_allowed']})")
    else:
        print("\nResult: PASS (no regressions)")


def save_comparison(result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"compare_{result['baseline']['prompt_version']}_vs_{result['candidate']['prompt_version']}_{stamp}.json"
    path = RESULTS_DIR / name
    path.write_text(json.dumps(result, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two eval runs and flag regressions.")
    parser.add_argument("--baseline", required=True, help="Prompt version (e.g. v3) or path to a results file")
    parser.add_argument("--candidate", required=True, help="Prompt version (e.g. v4) or path to a results file")
    parser.add_argument("--max-regressions", type=int, default=0,
                        help="How many regressions to tolerate before failing (default 0)")
    args = parser.parse_args()

    base_path, cand_path = find_results(args.baseline), find_results(args.candidate)
    print(f"Baseline:  {base_path}\nCandidate: {cand_path}")
    result = compare(json.loads(base_path.read_text()), json.loads(cand_path.read_text()),
                     args.max_regressions)
    print_comparison(result)
    print(f"\nSaved comparison to {save_comparison(result)}")
    sys.exit(1 if result["verdict"] == "REGRESSION" else 0)