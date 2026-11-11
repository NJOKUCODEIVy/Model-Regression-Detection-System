"""Run one prompt version against the golden dataset and score every case on several dimensions.

Per case: category match (pass/fail), summary relevance (LLM judge, 1-5), latency and token usage.
Cases run concurrently, within the limits set in src/llm.py.

Usage:
    python -m src.evals.run_eval --prompt prompts/v4.yaml
    python -m src.evals.run_eval --prompt prompts/v4.yaml --no-judge   # skip summary scoring (half the API calls)
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.classifier import build_messages, parse_response
from src.config import PromptConfig
from src.evals.judge import JUDGE_MODEL, judge_summary
from src.llm import LLM

GOLDEN_PATH = Path("data/golden.json")
RESULTS_DIR = Path("results")


def accuracy_by(rows: list[dict], key: str) -> dict:
    """Passed/total counts grouped by a field, e.g. category or difficulty."""
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in rows:
        groups[row[key]]["total"] += 1
        groups[row[key]]["passed"] += row["passed"]
    return dict(groups)


async def evaluate_case(llm: LLM, config: PromptConfig, case: dict, judge: bool) -> dict:
    expected = case["expected"]["category"]
    row = {
        "id": case["id"], "difficulty": case["expected_difficulty"], "expected": expected,
        "predicted": None, "expected_summary": case["expected"]["summary"], "summary": None,
        "passed": False, "error": None,
        "latency_ms": None, "prompt_tokens": None, "completion_tokens": None,
        "judge_score": None, "judge_reason": None, "judge_latency_ms": None,
        "judge_prompt_tokens": None, "judge_completion_tokens": None, "judge_error": None,
    }
    try:
        call = await llm.chat(config.model, build_messages(case["input"], config), config.temperature)
        row.update(latency_ms=call.latency_ms, prompt_tokens=call.prompt_tokens,
                   completion_tokens=call.completion_tokens)
        result = parse_response(call.text)
        row.update(predicted=result.category, summary=result.summary, passed=result.category == expected)
    except Exception as exc:  # bad JSON, invalid category, API error, etc.
        row["error"] = str(exc)

    if judge and row["summary"]:
        try:
            row.update(await judge_summary(llm, case["input"], case["expected"]["summary"], row["summary"]))
        except Exception as exc:  # a judge failure never changes the category result
            row["judge_error"] = str(exc)
    return row


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def total(rows: list[dict], key: str) -> int:
    return sum(row[key] or 0 for row in rows)


async def run_eval_async(prompt_path: str, judge: bool = True) -> dict:
    config = PromptConfig.from_yaml(prompt_path)
    dataset = json.loads(GOLDEN_PATH.read_text())
    cases = dataset["cases"]
    print(f"Running prompt {config.version} ({config.model}) on dataset v{dataset['dataset_version']} "
          f"({len(cases)} cases, judge: {JUDGE_MODEL if judge else 'off'})...\n")

    llm = LLM()
    done = 0

    async def run_one(case: dict) -> dict:
        nonlocal done
        row = await evaluate_case(llm, config, case, judge)
        done += 1
        score = f"  summary {row['judge_score']}/5" if row["judge_score"] else ""
        print(f"  {done:>2}/{len(cases)}  {'PASS' if row['passed'] else 'FAIL'}  {row['id']}  "
              f"({row['difficulty']}){score}")
        return row

    started = time.perf_counter()
    try:
        rows = await asyncio.gather(*(run_one(case) for case in cases))  # results stay in dataset order
    finally:
        await llm.close()
    wall_time = round(time.perf_counter() - started, 1)

    passed_total = sum(row["passed"] for row in rows)
    scores = [row["judge_score"] for row in rows if row["judge_score"]]
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    return {
        "prompt_version": config.version,
        "model": config.model,
        "judge_model": JUDGE_MODEL if judge else None,
        "dataset_version": dataset["dataset_version"],
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "accuracy": round(passed_total / len(rows), 4),
        "passed": passed_total,
        "total": len(rows),
        "by_category": accuracy_by(rows, "expected"),
        "by_difficulty": accuracy_by(rows, "difficulty"),
        "summary_relevance": {
            "mean": round(statistics.mean(scores), 2) if scores else None,
            "scored": len(scores),
            "counts": {str(s): scores.count(s) for s in range(1, 6)},
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies)) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "tokens": {
            "classifier_prompt": total(rows, "prompt_tokens"),
            "classifier_completion": total(rows, "completion_tokens"),
            "judge_prompt": total(rows, "judge_prompt_tokens"),
            "judge_completion": total(rows, "judge_completion_tokens"),
        },
        "wall_time_s": wall_time,
        "cases": rows,
    }


def run_eval(prompt_path: str, judge: bool | None = None) -> dict:
    """Synchronous entry point. The judge is on unless judge=False or the JUDGE env var is 'off'."""
    if judge is None:
        judge = os.getenv("JUDGE", "on").lower() != "off"
    return asyncio.run(run_eval_async(prompt_path, judge))


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

    rel, lat, tok = report.get("summary_relevance"), report.get("latency_ms"), report.get("tokens")
    if rel and rel["mean"] is not None:
        spread = "  ".join(f"{s}:{n}" for s, n in rel["counts"].items())
        print(f"\nSummary relevance: {rel['mean']}/5 average over {rel['scored']} summaries  ({spread})")
    if lat and lat["mean"] is not None:
        print(f"Latency per request: {lat['mean']} ms average, {lat['p50']} ms median, {lat['p95']} ms p95")
    if tok:
        print(f"Tokens: classifier {tok['classifier_prompt']:,} in / {tok['classifier_completion']:,} out, "
              f"judge {tok['judge_prompt']:,} in / {tok['judge_completion']:,} out")
    if "wall_time_s" in report:
        print(f"Wall time: {report['wall_time_s']} s")

    failures = [row for row in report["cases"] if not row["passed"]]
    if failures:
        print("\nFailures:")
        for row in failures:
            detail = f"error: {row['error']}" if row["error"] else f"got {row['predicted']}"
            print(f"  {row['id']} ({row['difficulty']}): expected {row['expected']}, {detail}")
    judge_errors = [row for row in report["cases"] if row.get("judge_error")]
    if judge_errors:  # e.g. the judge model isn't available: the category results are still valid
        print(f"\nJudge errors: {len(judge_errors)} summaries not scored. First error: {judge_errors[0]['judge_error'][:200]}")
    low = [row for row in report["cases"] if row.get("judge_score") is not None and row["judge_score"] <= 2]
    if low:
        print("\nWeak summaries (scored 1-2):")
        for row in low:
            print(f"  {row['id']} ({row['judge_score']}/5): {row['judge_reason']}")


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{report['prompt_version']}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a prompt version against the golden dataset.")
    parser.add_argument("--prompt", default="prompts/v4.yaml", help="Path to the prompt YAML file")
    parser.add_argument("--no-judge", action="store_true", help="Skip summary scoring")
    args = parser.parse_args()
    if not os.getenv("GROQ_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to your .env file.")

    report = run_eval(args.prompt, judge=not args.no_judge)
    print_report(report)
    print(f"\nSaved results to {save_report(report)}")