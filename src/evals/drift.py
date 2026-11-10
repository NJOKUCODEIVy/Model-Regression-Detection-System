"""Detect slow drift: a gradual decline that per-run regression checks can miss.

Each regression check compares one run with its baseline. If quality slips by one case at a time
and each small regression is accepted, every single check passes while the system keeps getting
worse. This looks at the rolling average of the last N runs instead, and warns when it falls
below a threshold.

Usage:
    python -m src.evals.drift                      # check your real run history
    python -m src.evals.drift --prompt v4          # only count runs of one prompt version
    python -m src.evals.drift --simulate           # try it on made-up history (no API calls)
    python -m src.evals.drift --simulate --alert   # also post the warning to Slack

The exit code is 1 when drift is detected, so CI can flag it.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

from src.evals.alerts import send_to_slack
from src.evals.compare import RESULTS_DIR

load_dotenv()

WINDOW = 7          # how many recent runs the rolling average covers
THRESHOLD = 95.0    # warn when the rolling average falls below this accuracy (%)


def load_runs(prompt: str | None = None) -> list[dict]:
    """Eval runs from results/, oldest first, on the same dataset version as the newest run."""
    runs = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("v*_2*.json")]
    if prompt:
        runs = [r for r in runs if r["prompt_version"] == prompt]
    runs.sort(key=lambda r: r["run_at"])
    if runs:  # scores are only comparable on the same dataset version
        runs = [r for r in runs if r["dataset_version"] == runs[-1]["dataset_version"]]
    return [{"prompt_version": r["prompt_version"], "run_at": r["run_at"],
             "accuracy": round(r["accuracy"] * 100, 1), "dataset_version": r["dataset_version"]} for r in runs]


def simulated_runs() -> list[dict]:
    """Ten made-up nightly runs of one prompt that lose about one case every other night."""
    scores = [98.3, 98.3, 98.3, 96.7, 96.7, 95.0, 95.0, 93.3, 93.3, 93.3]
    start = datetime(2026, 11, 1, 2, 0)
    return [{"prompt_version": "v4", "run_at": (start + timedelta(days=i)).isoformat(timespec="seconds"),
             "accuracy": score, "dataset_version": "1.0"} for i, score in enumerate(scores)]


def rolling_averages(runs: list[dict], window: int) -> list[float | None]:
    """The rolling average at each run (None until there are enough runs)."""
    return [round(sum(r["accuracy"] for r in runs[i + 1 - window:i + 1]) / window, 1) if i + 1 >= window else None
            for i in range(len(runs))]


def check_drift(runs: list[dict], window: int = WINDOW, threshold: float = THRESHOLD) -> dict:
    base = {"window": window, "threshold": threshold, "runs_available": len(runs),
            "checked_at": datetime.now().isoformat(timespec="seconds")}
    if len(runs) < window:
        return {**base, "status": "INSUFFICIENT_HISTORY",
                "message": f"Need {window} runs for a {window}-run average; only {len(runs)} so far."}

    averages = rolling_averages(runs, window)
    recent = runs[-window:]
    average, previous = averages[-1], averages[-2] if len(runs) > window else None
    steps = [round(b["accuracy"] - a["accuracy"], 1) for a, b in zip(recent, recent[1:])]
    largest_drop = max([0.0] + [-s for s in steps])
    window_change = round(recent[-1]["accuracy"] - recent[0]["accuracy"], 1)

    if average < threshold:
        status = "DRIFT"
        trend = (f" Accuracy changed {window_change:+.1f} pts across these runs, but no single run dropped "
                 f"more than {largest_drop} pts, so individual checks could miss it." if window_change < 0
                 else " Scores have stayed below the threshold across these runs.")
        message = (f"The {window}-run average fell to {average}%, below the {threshold}% threshold"
                   + (f" (it was {previous}% one run earlier)" if previous is not None else "") + "." + trend)
    else:
        status = "OK"
        message = f"The {window}-run average is {average}%, above the {threshold}% threshold."

    return {**base, "status": status, "message": message, "rolling_average": average,
            "previous_average": previous, "largest_single_drop": largest_drop,
            "window_change": window_change, "latest_accuracy": recent[-1]["accuracy"],
            "series": [{**r, "rolling_average": a} for r, a in zip(runs, averages)]}


def print_drift(drift: dict, simulated: bool) -> None:
    label = "SIMULATED history" if simulated else "your run history"
    print(f"Drift check on {label}: {drift['runs_available']} run(s), "
          f"{drift['window']}-run window, threshold {drift['threshold']}%\n")
    if drift["status"] == "INSUFFICIENT_HISTORY":
        print(f"Result: NOT ENOUGH HISTORY. {drift['message']}")
        print("Try it with made-up history:  python -m src.evals.drift --simulate")
        return

    print(f"  {'Run at':<18} {'Prompt':<8} {'Accuracy':>8}  {drift['window']}-run avg")
    for run in drift["series"]:
        avg = f"{run['rolling_average']}%" if run["rolling_average"] is not None else ""
        print(f"  {run['run_at'][:16].replace('T', ' '):<18} {run['prompt_version']:<8} {run['accuracy']:>7}%  {avg:>9}")

    print(f"\nLargest drop between two runs: {drift['largest_single_drop']} pts  |  "
          f"change across the window: {drift['window_change']:+.1f} pts")
    verdict = "SLOW DRIFT DETECTED" if drift["status"] == "DRIFT" else "OK"
    print(f"\nResult: {verdict}. {drift['message']}")


def build_drift_message(drift: dict, simulated: bool) -> dict:
    prefix = "[SIMULATION] " if simulated else ""
    title = f"{prefix}WARN: slow drift detected"
    return {
        "text": f"{title}: {drift['window']}-run average {drift['rolling_average']}% (threshold {drift['threshold']}%)",
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "emoji": True,
                                        "text": f":chart_with_downwards_trend: {title}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": drift["message"]}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*{drift['window']}-run average*\n{drift['rolling_average']}%"},
                {"type": "mrkdwn", "text": f"*Threshold*\n{drift['threshold']}%"},
                {"type": "mrkdwn", "text": f"*Latest run*\n{drift['latest_accuracy']}%"},
                {"type": "mrkdwn", "text": f"*Largest single drop*\n{drift['largest_single_drop']} pts"},
            ]},
            {"type": "context", "elements": [
                {"type": "mrkdwn", "text": f"Checked {drift['checked_at'][:16].replace('T', ' ')}"}]},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warn when the rolling average of eval scores drifts down.")
    parser.add_argument("--window", type=int, default=WINDOW, help=f"Runs in the rolling average (default {WINDOW})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Warn below this accuracy %% (default {THRESHOLD})")
    parser.add_argument("--prompt", help="Only count runs of this prompt version, e.g. v4")
    parser.add_argument("--simulate", action="store_true", help="Use made-up history instead of results/")
    parser.add_argument("--alert", action="store_true", help="Post to Slack if drift is detected")
    args = parser.parse_args()

    runs = simulated_runs() if args.simulate else load_runs(args.prompt)
    drift = check_drift(runs, args.window, args.threshold)
    print_drift(drift, args.simulate)

    if args.alert and drift["status"] == "DRIFT":
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            sys.exit("SLACK_WEBHOOK_URL is not set. Add it to your .env file.")
        try:
            send_to_slack(build_drift_message(drift, args.simulate), webhook_url)
        except RuntimeError as err:
            sys.exit(str(err))
        print("Drift warning sent to Slack.")
    sys.exit(1 if drift["status"] == "DRIFT" else 0)