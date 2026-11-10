"""Send the regression check result to Slack through an incoming webhook.

The message has the PASS/WARN/FAIL status, a headline like "1 regression detected, accuracy
dropped from 94% to 89%", the regressed cases, and a link to the full HTML report.

Usage:
    python -m src.evals.alerts --baseline v3 --candidate v4 --max-regressions 1
    python -m src.evals.alerts --baseline v3 --candidate v4 --dry-run   # show the message, don't send it
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.evals.compare import find_results, pct
from src.evals.report import build_report, save_report

load_dotenv()

STATUS_EMOJI = {"PASS": ":white_check_mark:", "WARN": ":warning:", "FAIL": ":rotating_light:"}
MAX_CASES_LISTED = 5  # keep the Slack message short; the report has the full list


def headline(result: dict) -> str:
    """e.g. '3 regressions detected, accuracy dropped from 94.0% to 89.0%'"""
    n = result["shared_cases"]
    before, after = pct(result["baseline_passed"], n), pct(result["candidate_passed"], n)
    count = len(result["regressions"])
    regressions = f"{count} regression{'s' if count != 1 else ''} detected" if count else "No regressions"
    if after < before:
        accuracy = f"accuracy dropped from {before}% to {after}%"
    elif after > before:
        accuracy = f"accuracy rose from {before}% to {after}%"
    else:
        accuracy = f"accuracy unchanged at {after}%"
    return f"{regressions}, {accuracy}"


def report_link(report_path: Path) -> str | None:
    """A link to the report: REPORT_URL (e.g. the CI run page), or REPORT_BASE_URL plus the file name
    (e.g. a GitHub Pages site). None when neither is set, so the message shows the local path."""
    if os.getenv("REPORT_URL"):
        return os.environ["REPORT_URL"]
    base = os.getenv("REPORT_BASE_URL", "").rstrip("/")
    return f"{base}/{report_path.name}" if base else None


def build_message(result: dict, status: str, report_path: Path, drift_warning: str | None = None) -> dict:
    b, c = result["baseline"], result["candidate"]
    title = headline(result)
    if status == "WARN" and result["regressions"]:
        title += f" (within the allowed limit of {result['max_regressions_allowed']})"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "emoji": True,
                                    "text": f"{STATUS_EMOJI[status]} {status}: model regression check"}},
        {"type": "section", "text": {"type": "mrkdwn", "text":
            f"*{title}*\n`{b['prompt_version']}` → `{c['prompt_version']}` · dataset v{c['dataset_version']}"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Regressions*\n{len(result['regressions'])}"},
            {"type": "mrkdwn", "text": f"*Fixes*\n{len(result['fixes'])}"},
            {"type": "mrkdwn", "text": f"*Still failing*\n{len(result['still_failing'])}"},
            {"type": "mrkdwn", "text": f"*Cases compared*\n{result['shared_cases']}"},
        ]},
    ]

    if result["regressions"]:
        lines = [f"• `{r['id']}` ({r['difficulty']}): expected *{r['expected']}*, was {r['was']}, now *{r['now'] or 'error'}*"
                 for r in result["regressions"][:MAX_CASES_LISTED]]
        extra = len(result["regressions"]) - MAX_CASES_LISTED
        if extra > 0:
            lines.append(f"…and {extra} more in the report")
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Regressed cases*\n" + "\n".join(lines)}})

    if drift_warning:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f":chart_with_downwards_trend: *Slow drift:* {drift_warning}"}})

    link = report_link(report_path)
    if link:
        blocks.append({"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "View full report"}, "url": link}]})
    else:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"Full report (on the machine that ran the check): `{report_path}`"}]})

    blocks.append({"type": "context", "elements": [
        {"type": "mrkdwn", "text": f"Checked {datetime.now().strftime('%Y-%m-%d %H:%M')}"}]})

    # "text" is what Slack shows in notifications and in clients that can't display blocks
    return {"text": f"{status}: {title}", "blocks": blocks}


def send_to_slack(message: dict, webhook_url: str) -> None:
    request = urllib.request.Request(webhook_url, data=json.dumps(message).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except urllib.error.HTTPError as err:  # Slack explains the problem in the response body
        raise RuntimeError(f"Slack rejected the message ({err.code}): {err.read().decode(errors='replace')}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Could not reach Slack: {err.reason}") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post the regression check result to Slack.")
    parser.add_argument("--baseline", required=True, help="Prompt version (e.g. v3) or path to a results file")
    parser.add_argument("--candidate", required=True, help="Prompt version (e.g. v4) or path to a results file")
    parser.add_argument("--max-regressions", type=int, default=0, help="Regressions to tolerate (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print the Slack message instead of sending it")
    args = parser.parse_args()

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url and not args.dry_run:
        sys.exit("SLACK_WEBHOOK_URL is not set. Add it to your .env file, or use --dry-run.")

    baseline = json.loads(find_results(args.baseline).read_text())
    candidate = json.loads(find_results(args.candidate).read_text())
    page, result, status = build_report(baseline, candidate, args.max_regressions)
    report_path = save_report(page, result)
    message = build_message(result, status, report_path)

    print(f"Status: {status}  |  {headline(result)}")
    print(f"Report saved to {report_path}")
    if args.dry_run:
        print("\nDry run: this is the message that would be sent to Slack:\n")
        print(json.dumps(message, indent=2, ensure_ascii=False))
    else:
        try:
            send_to_slack(message, webhook_url)
        except RuntimeError as err:
            sys.exit(str(err))
        print("Alert sent to Slack.")