"""Build an HTML diff report comparing two eval runs.

The report shows run metadata, a scorecard against the baseline, every regressed case with the
old and new output side by side, and a trend chart of scores over recent runs.

Usage:
    python -m src.evals.report --baseline v3 --candidate v4
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

from src.evals.compare import RESULTS_DIR, compare, find_results, pct

GOLDEN_PATH = Path("data/golden.json")
REPORTS_DIR = Path("reports")
TREND_RUNS = 10  # how many recent runs the trend chart shows


def run_status(result: dict, candidate: dict) -> str:
    """FAIL = regression beyond the allowed limit, WARN = needs a look, PASS = all clear."""
    if result["verdict"] == "REGRESSION":
        return "FAIL"
    if result["regressions"] or any(case["error"] for case in candidate["cases"]):
        return "WARN"
    return "PASS"


def load_history(limit: int = TREND_RUNS) -> list[dict]:
    """All eval runs in results/, oldest first, trimmed to the most recent `limit`."""
    runs = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("v*_2*.json")]
    runs.sort(key=lambda run: run["run_at"])
    return runs[-limit:]


def trend_chart_svg(runs: list[dict], highlight_run_at: str) -> str:
    """A simple line chart of accuracy per run, drawn as inline SVG so the report has no dependencies."""
    if len(runs) < 2:
        return "<p class='muted'>The trend chart appears once there are at least two runs.</p>"

    width, height, pad_x, pad_top, pad_bottom = 720, 260, 48, 28, 52
    scores = [run["accuracy"] * 100 for run in runs]
    y_min = max(0, (int(min(scores)) // 5) * 5 - 5)
    plot_w, plot_h = width - 2 * pad_x, height - pad_top - pad_bottom

    def x(i: int) -> float:
        return pad_x + plot_w * i / (len(runs) - 1)

    def y(score: float) -> float:
        return pad_top + plot_h * (100 - score) / (100 - y_min)

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Accuracy over recent runs'>"]
    for tick in range(y_min, 101, 5):  # horizontal grid lines
        parts.append(f"<line x1='{pad_x}' x2='{width - pad_x}' y1='{y(tick):.1f}' y2='{y(tick):.1f}' class='grid'/>")
        parts.append(f"<text x='{pad_x - 8}' y='{y(tick) + 4:.1f}' class='axis' text-anchor='end'>{tick}%</text>")

    points = " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in enumerate(scores))
    parts.append(f"<polyline points='{points}' class='line'/>")

    for i, (run, score) in enumerate(zip(runs, scores)):
        cls = "dot current" if run["run_at"] == highlight_run_at else "dot"
        label = escape(run["prompt_version"])
        tooltip = escape(f"{run['prompt_version']} at {run['run_at']}: {score:.1f}% (dataset v{run['dataset_version']})")
        parts.append(f"<circle cx='{x(i):.1f}' cy='{y(score):.1f}' r='5' class='{cls}'><title>{tooltip}</title></circle>")
        parts.append(f"<text x='{x(i):.1f}' y='{y(score) - 10:.1f}' class='value' text-anchor='middle'>{score:.0f}%</text>")
        parts.append(f"<text x='{x(i):.1f}' y='{height - pad_bottom + 20}' class='axis' text-anchor='middle'>{label}</text>")
        parts.append(f"<text x='{x(i):.1f}' y='{height - pad_bottom + 36}' class='axis small' text-anchor='middle'>"
                     f"{escape(run['run_at'][5:16].replace('T', ' '))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def output_cell(category: str | None, summary: str | None, error: str | None, passed: bool) -> str:
    if error:
        return f"<span class='tag fail'>error</span><div class='muted'>{escape(error[:160])}</div>"
    tag = "pass" if passed else "fail"
    return f"<span class='tag {tag}'>{escape(str(category))}</span><div>{escape(summary or '')}</div>"


def case_table(entries: list[dict], base: dict, cand: dict, inputs: dict) -> str:
    if not entries:
        return "<p class='muted'>None.</p>"
    rows = []
    for entry in entries:
        b, c = base[entry["id"]], cand[entry["id"]]
        rows.append(
            "<tr>"
            f"<td><strong>{escape(entry['id'])}</strong><div class='muted'>{escape(entry['difficulty'])}</div></td>"
            f"<td class='email'>{escape(inputs.get(entry['id'], ''))}</td>"
            f"<td><span class='tag'>{escape(c['expected'])}</span><div>{escape(c.get('expected_summary') or '')}</div></td>"
            f"<td>{output_cell(b['predicted'], b.get('summary'), b['error'], b['passed'])}</td>"
            f"<td>{output_cell(c['predicted'], c.get('summary'), c['error'], c['passed'])}</td>"
            "</tr>"
        )
    return ("<div class='scroll'><table><thead><tr><th>Case</th><th>Email</th><th>Expected</th>"
            "<th>Baseline output</th><th>Candidate output</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def breakdown_table(base: dict, cand: dict, shared: list[str], key: str, order: list[str] | None = None) -> str:
    groups = defaultdict(lambda: [0, 0, 0])  # total, baseline passed, candidate passed
    for cid in shared:
        g = groups[cand[cid][key]]
        g[0] += 1
        g[1] += base[cid]["passed"]
        g[2] += cand[cid]["passed"]
    names = [n for n in (order or sorted(groups)) if n in groups]
    rows = []
    for name in names:
        total, bp, cp = groups[name]
        change = cp - bp
        cls = "up" if change > 0 else "down" if change < 0 else ""
        rows.append(f"<tr><td>{escape(name)}</td><td>{bp}/{total}</td><td>{cp}/{total}</td>"
                    f"<td class='{cls}'>{f'{change:+d}' if change else ''}</td></tr>")
    return ("<table class='compact'><thead><tr><th></th><th>Baseline</th><th>Candidate</th><th>Change</th></tr>"
            "</thead><tbody>" + "".join(rows) + "</tbody></table>")


def build_report(baseline: dict, candidate: dict, max_regressions: int = 0) -> tuple[str, dict, str]:
    result = compare(baseline, candidate, max_regressions)
    status = run_status(result, candidate)
    base = {c["id"]: c for c in baseline["cases"]}
    cand = {c["id"]: c for c in candidate["cases"]}
    shared = [cid for cid in base if cid in cand]
    inputs = {c["id"]: c["input"] for c in json.loads(GOLDEN_PATH.read_text())["cases"]} if GOLDEN_PATH.exists() else {}

    n, bp, cp = result["shared_cases"], result["baseline_passed"], result["candidate_passed"]
    change = result["accuracy_change_pts"]
    b_name, c_name = escape(baseline["prompt_version"]), escape(candidate["prompt_version"])
    dataset_note = ("" if result["same_dataset"] else
                    "<p class='warning'>The two runs used different dataset versions, so scores are not fully comparable.</p>")

    meta_rows = "".join(
        f"<tr><td>{label}</td><td>{escape(str(baseline[key]))}</td><td>{escape(str(candidate[key]))}</td></tr>"
        for label, key in [("Prompt version", "prompt_version"), ("Model", "model"),
                           ("Dataset version", "dataset_version"), ("Run at", "run_at")])

    body = f"""
<header>
  <div>
    <p class="eyebrow">Model regression report</p>
    <h1>{b_name} &rarr; {c_name}</h1>
    <p class="muted">Generated {escape(datetime.now().isoformat(timespec="seconds"))}</p>
  </div>
  <span class="status {status.lower()}">{status}</span>
</header>
{dataset_note}

<section class="cards">
  <div class="card"><p class="label">Accuracy</p><p class="big">{pct(cp, n)}%</p>
    <p class="muted">was {pct(bp, n)}% &middot; <span class="{'up' if change > 0 else 'down' if change < 0 else ''}">{change:+.1f} pts</span></p></div>
  <div class="card"><p class="label">Regressions</p><p class="big {'down' if result['regressions'] else ''}">{len(result['regressions'])}</p>
    <p class="muted">passed before, fail now</p></div>
  <div class="card"><p class="label">Fixes</p><p class="big {'up' if result['fixes'] else ''}">{len(result['fixes'])}</p>
    <p class="muted">failed before, pass now</p></div>
  <div class="card"><p class="label">Still failing</p><p class="big">{len(result['still_failing'])}</p>
    <p class="muted">failed in both runs</p></div>
</section>

<section>
  <h2>Run details</h2>
  <table class="compact"><thead><tr><th></th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>{meta_rows}</tbody></table>
</section>

<section class="two-col">
  <div><h2>By category</h2>{breakdown_table(base, cand, shared, "expected")}</div>
  <div><h2>By difficulty</h2>{breakdown_table(base, cand, shared, "difficulty", ["easy", "medium", "hard"])}</div>
</section>

<section>
  <h2>Regressed cases ({len(result['regressions'])})</h2>
  {case_table(result['regressions'], base, cand, inputs)}
</section>

<section>
  <h2>Fixed cases ({len(result['fixes'])})</h2>
  {case_table(result['fixes'], base, cand, inputs)}
</section>

<section>
  <h2>Still failing ({len(result['still_failing'])})</h2>
  {case_table(result['still_failing'], base, cand, inputs)}
</section>

<section>
  <h2>Accuracy over the last {TREND_RUNS} runs</h2>
  {trend_chart_svg(load_history(), candidate["run_at"])}
</section>
"""
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regression report: {b_name} vs {c_name}</title>
<style>{CSS}</style></head>
<body><main>{body}</main></body></html>"""
    return page, result, status


CSS = """
:root { --bg:#fafaf9; --surface:#ffffff; --text:#1c1917; --muted:#78716c; --border:#e7e5e4;
  --pass:#15803d; --pass-bg:#dcfce7; --warn:#a16207; --warn-bg:#fef9c3; --fail:#b91c1c; --fail-bg:#fee2e2;
  --accent:#4f46e5; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --surface:#292524; --text:#f5f5f4; --muted:#a8a29e;
  --border:#44403c; --pass:#4ade80; --pass-bg:#14532d; --warn:#facc15; --warn-bg:#422006; --fail:#f87171;
  --fail-bg:#450a0a; --accent:#a5b4fc; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 20px 64px; }
header { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:24px; }
h1 { margin:4px 0; font-size:28px; } h2 { font-size:18px; margin:32px 0 12px; }
.eyebrow { margin:0; text-transform:uppercase; letter-spacing:.08em; font-size:12px; color:var(--muted); }
.muted { color:var(--muted); font-size:13px; margin:4px 0 0; }
.status { font-weight:700; padding:8px 16px; border-radius:999px; font-size:16px; }
.status.pass { color:var(--pass); background:var(--pass-bg); } .status.warn { color:var(--warn); background:var(--warn-bg); }
.status.fail { color:var(--fail); background:var(--fail-bg); }
.warning { background:var(--warn-bg); color:var(--warn); padding:10px 14px; border-radius:8px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px; }
.label { margin:0; font-size:13px; color:var(--muted); } .big { margin:4px 0 0; font-size:32px; font-weight:700; }
.up { color:var(--pass); } .down { color:var(--fail); }
.two-col { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:24px; }
.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:top; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
tr:last-child td { border-bottom:none; }
td.email { max-width:280px; font-style:italic; }
td:first-child { white-space:nowrap; }
.tag { display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; background:var(--border); margin-bottom:4px; }
.tag.pass { color:var(--pass); background:var(--pass-bg); } .tag.fail { color:var(--fail); background:var(--fail-bg); }
svg { width:100%; height:auto; background:var(--surface); border:1px solid var(--border); border-radius:12px; }
svg .grid { stroke:var(--border); } svg .axis { fill:var(--muted); font-size:12px; } svg .small { font-size:10px; }
svg .line { fill:none; stroke:var(--accent); stroke-width:2.5; } svg .dot { fill:var(--accent); }
svg .dot.current { fill:var(--fail); stroke:var(--surface); stroke-width:2; } svg .value { fill:var(--text); font-size:12px; font-weight:600; }
"""


def save_report(page: str, result: dict) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"report_{result['baseline']['prompt_version']}_vs_{result['candidate']['prompt_version']}_{stamp}.html"
    path = REPORTS_DIR / name
    path.write_text(page, encoding="utf-8")
    (REPORTS_DIR / "latest.html").write_text(page, encoding="utf-8")  # always points at the newest report
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an HTML diff report for two eval runs.")
    parser.add_argument("--baseline", required=True, help="Prompt version (e.g. v3) or path to a results file")
    parser.add_argument("--candidate", required=True, help="Prompt version (e.g. v4) or path to a results file")
    parser.add_argument("--max-regressions", type=int, default=0, help="Regressions to tolerate (default 0)")
    args = parser.parse_args()

    baseline = json.loads(find_results(args.baseline).read_text())
    candidate = json.loads(find_results(args.candidate).read_text())
    page, result, status = build_report(baseline, candidate, args.max_regressions)
    path = save_report(page, result)
    print(f"Status: {status}  |  {len(result['regressions'])} regression(s), {len(result['fixes'])} fix(es), "
          f"accuracy {result['accuracy_change_pts']:+.1f} pts")
    print(f"Report saved to {path}")
    print(f"Open it with:  open {path}")