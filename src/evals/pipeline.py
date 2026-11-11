"""Run the whole regression check with one command.

Steps: evaluate the candidate prompt, compare it with the baseline, build the HTML report,
check for slow drift, and post the result to Slack. This is the command CI runs.

Usage:
    python -m src.evals.pipeline --candidate prompts/v4.yaml --baseline v3
    python -m src.evals.pipeline --candidate prompts/v4.yaml --baseline v3 --skip-eval   # reuse the latest results (no API calls)
    python -m src.evals.pipeline --candidate prompts/v4.yaml --baseline v4               # a fresh v4 run vs the previous v4 run

Settings can also come from environment variables (handy for CI and Docker):
    BASELINE_PROMPT, WARNING_DELTA, CRITICAL_DELTA, MAX_REGRESSIONS (optional), DRIFT_THRESHOLD, DRIFT_WINDOW,
    JUDGE (on/off), JUDGE_MODEL, EVAL_CONCURRENCY, REQUESTS_PER_MINUTE

The exit code is 1 when the regression check FAILS, so CI can block the change.
A drift warning is reported in Slack but doesn't fail the run, because it's about a trend.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config import PromptConfig
from src.evals.alerts import build_message, headline, send_to_slack
from src.evals.compare import RESULTS_DIR, add_threshold_args, dimension_rows, pct, thresholds
from src.evals.drift import THRESHOLD, WINDOW, check_drift, load_runs
from src.evals.report import build_report, save_report
from src.evals.run_eval import print_report, run_eval
from src.evals.run_eval import save_report as save_results

load_dotenv()


def latest_results(version: str, exclude: Path | None = None) -> Path | None:
    """Newest results file for a prompt version, skipping `exclude` (so v4 can be compared with an earlier v4 run)."""
    matches = [p for p in sorted(RESULTS_DIR.glob(f"{version}_2*.json"))
               if exclude is None or p.resolve() != exclude.resolve()]
    return matches[-1] if matches else None


def resolve_baseline(ref: str, candidate_path: Path) -> Path:
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return path
    found = latest_results(ref, exclude=candidate_path)
    if not found:
        sys.exit(f"No earlier results for baseline '{ref}'. Run it first: "
                 f"python -m src.evals.run_eval --prompt prompts/{ref}.yaml")
    return found


STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "🚨"}


def build_summary(result: dict, status: str, drift: dict, report_url: str | None) -> str:
    """A Markdown summary for the pull request comment and the GitHub Actions run page."""
    b, c = result["baseline"], result["candidate"]
    n, bp, cp = result["shared_cases"], result["baseline_passed"], result["candidate_passed"]
    title = headline(result)
    t = result["thresholds"]
    limit = f" · hard limit {t['max_regressions']} regressions" if t["max_regressions"] is not None else ""
    lines = [
        f"## {STATUS_ICON[status]} {status}: prompt regression check for `{c['prompt_version']}`",
        f"**{title}**",
        "",
        f"Candidate `{c['prompt_version']}` vs baseline `{b['prompt_version']}` · {n} cases · dataset v{c['dataset_version']}",
        "",
        f"**Severity: {result['severity']}** (warning above {t['warning_delta']:g} pts, critical above {t['critical_delta']:g} pts{limit})",
    ]
    lines += [f"- {reason}" for reason in result["reasons"]]
    lines += [
        "",
        "| | Baseline | Candidate | Change |",
        "|---|---|---|---|",
        f"| Pass rate | {pct(bp, n)}% | {pct(cp, n)}% | {result['accuracy_change_pts']:+.1f} pts |",
        f"| Regressions | | {len(result['regressions'])} | |",
        f"| Fixes | | {len(result['fixes'])} | |",
        f"| Still failing | | {len(result['still_failing'])} | |",
    ]
    lines += [f"| {label} | {before} | {after} | {change} |" for label, before, after, change in dimension_rows(result)]
    if result["regressions"]:
        lines += ["", "**Regressed cases**", "", "| Case | Difficulty | Expected | Was | Now | Summary score |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| `{r['id']}` | {r['difficulty']} | {r['expected']} | {r['was']} | {r['now'] or 'error'} | "
                  f"{r['summary_score_now'] if r['summary_score_now'] is not None else '—'} |"
                  for r in result["regressions"]]
    if drift["status"] == "DRIFT":
        lines += ["", f"📉 **Slow drift:** {drift['message']}"]
    if status == "FAIL":
        lines += ["", "This change is blocked: the drop is past the critical threshold. Fix the prompt, "
                  "or raise CRITICAL_DELTA on purpose and say why in this pull request."]
    if report_url:
        lines += ["", f"📄 Full HTML report: [workflow run]({report_url}) (download the `regression-report` artifact)"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run eval, compare, report, drift check and Slack alert.")
    parser.add_argument("--candidate", required=True, help="Prompt file to test, e.g. prompts/v4.yaml")
    parser.add_argument("--baseline", default=os.getenv("BASELINE_PROMPT"),
                        help="Baseline prompt version (e.g. v3) or results file (env: BASELINE_PROMPT)")
    add_threshold_args(parser)
    parser.add_argument("--skip-eval", action="store_true", help="Use the candidate's latest results instead of a new run")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("DRIFT_THRESHOLD", THRESHOLD)),
                        help=f"Drift threshold %% (default {THRESHOLD}, env: DRIFT_THRESHOLD)")
    parser.add_argument("--window", type=int, default=int(os.getenv("DRIFT_WINDOW", WINDOW)),
                        help=f"Runs in the drift average (default {WINDOW}, env: DRIFT_WINDOW)")
    parser.add_argument("--summary-file", help="Also write a Markdown summary to this file (used for PR comments)")
    parser.add_argument("--no-slack", action="store_true", help="Don't post to Slack")
    args = parser.parse_args(argv)
    if not args.baseline:
        parser.error("--baseline is required (or set the BASELINE_PROMPT environment variable)")

    version = PromptConfig.from_yaml(args.candidate).version

    # A missing key is a setup problem, not a regression, so stop before making 60 failing calls
    if not args.skip_eval and not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to your .env file, pass it with "
                 "docker run -e GROQ_API_KEY=..., or add it as a GitHub Actions secret.")

    # 1. Evaluate the candidate
    if args.skip_eval:
        candidate_path = latest_results(version)
        if not candidate_path:
            sys.exit(f"--skip-eval was used, but there are no results for {version} yet.")
        print(f"[1/4] Eval skipped: using {candidate_path}")
    else:
        print(f"[1/4] Running eval for {args.candidate}")
        run = run_eval(args.candidate)
        print_report(run)
        candidate_path = save_results(run)
        print(f"      Saved results to {candidate_path}")
    candidate = json.loads(candidate_path.read_text())

    # 2 and 3. Compare with the baseline and build the HTML report
    baseline_path = resolve_baseline(args.baseline, candidate_path)
    baseline = json.loads(baseline_path.read_text())
    print(f"\n[2/4] Comparing with baseline {baseline_path}")
    page, result, status = build_report(baseline, candidate, **thresholds(args))
    print(f"      {status} (severity: {result['severity']}): {headline(result)}")
    for reason in result["reasons"]:
        print(f"        - {reason}")
    report_path = save_report(page, result)
    print(f"[3/4] Report saved to {report_path}")

    # 4. Check for slow drift across recent runs of this prompt version
    drift = check_drift(load_runs(prompt=version), args.window, args.threshold)
    print(f"[4/4] Drift ({version}, {args.window}-run window): {drift['status']}. {drift['message']}")
    drift_warning = drift["message"] if drift["status"] == "DRIFT" else None

    # Markdown summary for the PR comment and the GitHub Actions run page
    summary = build_summary(result, status, drift, os.getenv("REPORT_URL"))
    if args.summary_file:
        Path(args.summary_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_file).write_text(summary, encoding="utf-8")
        print(f"      Summary written to {args.summary_file}")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary)

    # Slack
    if args.no_slack:
        print("\nSlack: skipped (--no-slack)")
    elif not os.getenv("SLACK_WEBHOOK_URL"):
        print("\nSlack: skipped (SLACK_WEBHOOK_URL is not set)")
    else:
        try:
            send_to_slack(build_message(result, status, report_path, drift_warning), os.environ["SLACK_WEBHOOK_URL"])
            print("\nSlack: alert sent")
        except RuntimeError as err:
            print(f"\nSlack: failed. {err}")

    print(f"\nOverall: {status}" + (" (plus a slow drift warning)" if drift_warning else ""))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())