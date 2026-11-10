"""Run one prompt version against the golden dataset and report how it did.

Usage:
    python -m src.evals.run_eval --prompt prompts/v1.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.classifier import classify_email
from src.config import PromptConfig

GOLDEN_PATH = Path("data/golden.json")
RESULTS_DIR = Path("results")
PAUSE_SECONDS = 1  # small gap between calls to stay friendly with free-tier rate limits


def accuracy_by(rows: list[dict], key: str) -> dict:
    """Passed/total counts grouped by a field, e.g. category or difficulty."""
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in rows:
        groups[row[key]]["total"] += 1
        groups[row[key]]["passed"] += row["passed"]
    return dict(groups)


def run_eval(prompt_path: str) -> dict:
    config = PromptConfig.from_yaml(prompt_path)
    dataset = json.loads(GOLDEN_PATH.read_text())
    cases = dataset["cases"]

    print(f"Running prompt {config.version} ({config.model}) on dataset "
          f"v{dataset['dataset_version']} ({len(cases)} cases)...\n")
    rows = []
    for i, case in enumerate(cases, 1):
        expected = case["expected"]["category"]
        try:
            result = classify_email(case["input"], config)
            predicted, summary, error = result.category, result.summary, None
        except Exception as exc:  # bad JSON, invalid category, API error, etc.
            predicted, summary, error = None, None, str(exc)

        passed = predicted == expected
        rows.append({
            "id": case["id"],
            "difficulty": case["expected_difficulty"],
            "expected": expected,
            "predicted": predicted,
            "expected_summary": case["expected"]["summary"],
            "summary": summary,
            "passed": passed,
            "error": error,
        })
        print(f"  {i:>2}/{len(cases)}  {'PASS' if passed else 'FAIL'}  {case['id']}  ({case['expected_difficulty']})")
        time.sleep(PAUSE_SECONDS)

    passed_total = sum(row["passed"] for row in rows)
    return {
        "prompt_version": config.version,
        "model": config.model,
        "dataset_version": dataset["dataset_version"],
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(passed_total / len(rows), 4),
        "passed": passed_total,
        "total": len(rows),
        "by_category": accuracy_by(rows, "expected"),
        "by_difficulty": accuracy_by(rows, "difficulty"),
        "cases": rows,
    }


def print_report(report: dict) -> None:
    pct = report["accuracy"] * 100
    print(f"\nPrompt {report['prompt_version']} on dataset v{report['dataset_version']}: "
          f"{report['passed']}/{report['total']} correct ({pct:.1f}%)")

    print("\nBy category:")
    for name, counts in sorted(report["by_category"].items()):
        print(f"  {name:<10} {counts['passed']}/{counts['total']}")

    print("\nBy difficulty:")
    for name in ("easy", "medium", "hard"):
        if name in report["by_difficulty"]:
            counts = report["by_difficulty"][name]
            print(f"  {name:<10} {counts['passed']}/{counts['total']}")

    failures = [row for row in report["cases"] if not row["passed"]]
    if failures:
        print("\nFailures:")
        for row in failures:
            detail = f"error: {row['error']}" if row["error"] else f"got {row['predicted']}"
            print(f"  {row['id']} ({row['difficulty']}): expected {row['expected']}, {detail}")


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{report['prompt_version']}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a prompt version against the golden dataset.")
    parser.add_argument("--prompt", default="prompts/v1.yaml", help="Path to the prompt YAML file")
    args = parser.parse_args()

    report = run_eval(args.prompt)
    print_report(report)
    print(f"\nSaved results to {save_report(report)}")