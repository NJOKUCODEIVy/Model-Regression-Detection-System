# Model Regression Detection System

![Prompt regression check](https://github.com/NJOKUCODEIVy/Model-Regression-Detection-System/actions/workflows/regression-check.yml/badge.svg)
![Docker build](https://github.com/NJOKUCODEIVy/Model-Regression-Detection-System/actions/workflows/docker.yml/badge.svg)

A CI pipeline that tests an LLM feature against a golden dataset whenever its prompt changes. The feature under test is a customer support email classifier: it reads an email and returns one of four categories (`billing`, `technical`, `account`, `general`) plus a one-sentence summary. Every prompt version lives in its own YAML file. When a pull request adds or changes one, CI scores it on 60 labeled emails across four dimensions (category match, summary relevance rated 1–5 by an LLM judge, latency and token usage), compares it with the current baseline case by case, posts a summary comment on the pull request and sends a Slack alert. Severity comes from how far the pass rate drops: past 3 points it warns, past 8 points it blocks the merge. A rolling average across recent runs catches slow drift that individual checks miss.

**Write-up:** [Prompt changes ship blind. I built CI to catch them.](docs/blog-post.md)

## Current state

| Prompt | Score (dataset v1.0) | Notes |
|---|---|---|
| v1 | 56/60 | Original prompt. Login-related bugs went to `account`; information requests went to `technical` |
| v2 | 59/60 | Fixed v1's four failures, regressed case-020 |
| v3 | 59/60 | Fixed case-020, regressed case-076 |
| **v4** | **59–60/60** | **Baseline.** Fixed case-076. case-049 is flaky: it failed in a local run and passed in a CI run with no change |

Model: `openai/gpt-oss-120b` through Groq's OpenAI-compatible API, temperature 0.

## How a prompt change flows

1. Create `prompts/vN.yaml` as a copy of the baseline prompt. Set `version`, `created_at`, `parent_version` and `change_notes`. Change one thing, so a result can be traced to its cause.
2. Open a pull request. The **Prompt regression check** workflow evaluates every added or modified file in `prompts/` against `BASELINE_PROMPT`.
3. The workflow posts (and on later pushes, updates) a comment with the status, scorecard and regressed cases. The full HTML report is attached to the run as the `regression-report` artifact. Slack gets the same summary.
4. A **FAIL** blocks the merge. Fix the prompt, or raise `CRITICAL_DELTA` on purpose and record why in the pull request.
5. To promote a merged version to baseline: run `python -m src.evals.run_eval --prompt prompts/vN.yaml` locally, commit the new file in `results/`, then set the `BASELINE_PROMPT` repository variable to `vN`. CI does not commit results, so a version cannot become the baseline until its results file is in the repo.

## Statuses

Severity is measured by the drop in pass rate against the baseline, in percentage points.

| Pass-rate drop | Severity | Status | Merge |
|---|---|---|---|
| More than `CRITICAL_DELTA` (8) | critical | **FAIL** | Blocked |
| More than `WARNING_DELTA` (3) | warning | **WARN** | Allowed |
| Within 3 points, but some cases regressed or hit API errors | ok | **WARN** | Allowed; review the listed cases |
| No regressions | ok | **PASS** | Allowed |

With 60 cases, one case is 1.7 points, so a 3-point warning is about two flips and an 8-point block is about five. If `MAX_REGRESSIONS` is set, more regressions than that is also critical; it is unset by default.

Summary relevance, latency and tokens are reported next to the pass rate in every comparison, but they don't decide the status. Only the category match is pass/fail.

## Setup

**Local (Python 3.11+)**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root (it is gitignored and excluded from the Docker image):

```
GROQ_API_KEY=gsk_...                                   # console.groq.com → API Keys
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... # optional; Slack incoming webhook
```

**GitHub**

- Secrets (Settings → Secrets and variables → Actions → Secrets): `GROQ_API_KEY`, `SLACK_WEBHOOK_URL`.
- Variables (same page → Variables), all optional: `BASELINE_PROMPT` (default `v4`), `WARNING_DELTA` (`3`), `CRITICAL_DELTA` (`8`), `MAX_REGRESSIONS` (unset), `DRIFT_THRESHOLD` (`95`), `JUDGE_MODEL` (`openai/gpt-oss-20b`).
- Ruleset **Protect main**: requires a pull request and the `regression-check` status check.

**Docker**

```bash
docker build -t regression-check .
docker run --rm --env-file .env -v "$PWD/reports:/app/reports" regression-check
docker run --rm --env-file .env -e CRITICAL_DELTA=5 \
  -v "$PWD/reports:/app/reports" -v "$PWD/results:/app/results" \
  regression-check --candidate prompts/v5.yaml --baseline v4
```

The default command evaluates `prompts/v4.yaml` against `BASELINE_PROMPT`. Mount `results/` if new results should outlive the container. The container runs as UID 1000, so mounted folders must be writable by it. CI builds the image and smoke-tests it against saved results on every relevant pull request, without calling the API.

## Commands

```bash
python -m src.evals.run_eval --prompt prompts/v4.yaml        # score one version (about 120 API calls with the judge)
python -m src.evals.run_eval --prompt prompts/v4.yaml --no-judge   # categories, latency and tokens only (60 calls)
python -m src.evals.compare  --baseline v3 --candidate v4    # case-by-case diff, no API calls
python -m src.evals.report   --baseline v3 --candidate v4    # HTML report → reports/latest.html
python -m src.evals.alerts   --baseline v3 --candidate v4 --dry-run   # preview the Slack message
python -m src.evals.drift    [--prompt v4] [--simulate]      # rolling-average drift check
python -m src.evals.pipeline --candidate prompts/v5.yaml --baseline v4  # all of the above (what CI runs)

python -m src.storage runs       # every run with all its scores (SQLite, built from results/)
python -m src.storage versions   # averages per prompt version
python -m src.storage case case-049
python -m src.storage flaky      # cases that both passed and failed on the same prompt and data
python -m src.storage weak       # summaries the judge scored 1-2
```

`--baseline` and `--candidate` accept a prompt version (the newest results file for it is used) or a path to a results file. `--skip-eval` on the pipeline reuses the newest saved results instead of calling the API. `compare`, `report`, `alerts` and `pipeline` all take `--warning-delta`, `--critical-delta` and `--max-regressions`.

## Adding test cases

The golden dataset is `data/golden.json`. Each case:

```json
{
  "id": "case-101",
  "input": "The email text",
  "expected": {"category": "billing", "summary": "The ideal one-sentence summary."},
  "expected_difficulty": "easy | medium | hard",
  "notes": "Why this case matters, e.g. 'mentions paying but the issue is a bug'"
}
```

Rules:

- **IDs are permanent.** Never renumber or reuse one. Gaps in the numbering are expected; cases were removed when the set was trimmed from 100 to 60.
- **A person decides every label.** Cases may be drafted with help, but someone must read each one and agree with its category and summary before it goes in.
- **Bump the version for any change.** Increment `dataset_version` (1.0 → 1.1) and add a `changelog` entry. Then re-run the baseline prompt and commit its results, because scores from different dataset versions are not comparable (`compare` warns and only scores shared cases).
- **Never change a label because a run failed it.** Relabel only when the label itself is wrong, and record the reason in `notes` and the changelog.
- **Keep few-shot examples out of the dataset.** A prompt that contains a test case is graded on its own answer key.
- **Start new cases from failures.** Real misclassifications are the best source. `data/reserve_cases.json` holds 40 labeled cases (stable IDs) ready to move in.

## Adjusting thresholds

| Setting | Default | Where to change it | Effect |
|---|---|---|---|
| `BASELINE_PROMPT` | `v4` | Repository variable, env var, or `--baseline` | What each candidate is compared against |
| `WARNING_DELTA` | `3` | Repository variable, env var, or `--warning-delta` | Pass-rate drop (points) that triggers WARN |
| `CRITICAL_DELTA` | `8` | Repository variable, env var, or `--critical-delta` | Pass-rate drop (points) that triggers FAIL and blocks the merge |
| `MAX_REGRESSIONS` | unset | Repository variable, env var, or `--max-regressions` | Optional hard limit: more regressions than this is FAIL |
| `DRIFT_THRESHOLD` | `95` | Repository variable, env var, or `--threshold` | Rolling-average accuracy (%) below which drift is reported |
| `DRIFT_WINDOW` | `7` | Env var or `--window` | Number of runs in the rolling average |
| `JUDGE` | `on` | Env var, or `--no-judge` on `run_eval` | Whether summaries are scored |
| `JUDGE_MODEL` | `openai/gpt-oss-20b` | Repository variable or env var | The model that scores summaries |
| `EVAL_CONCURRENCY` | `4` | Env var | Requests in flight at once |
| `REQUESTS_PER_MINUTE` | `25` | Env var | Requests started per minute, per model |

Keep `DRIFT_THRESHOLD` a few points below normal accuracy (v4 scores about 98%) so one noisy run doesn't trigger it. Changing `CRITICAL_DELTA` or setting `MAX_REGRESSIONS` should be a deliberate, recorded decision, not a way to get a pull request through.

## Architecture decisions

**Prompts are versioned files, never edited in place.** Comparison needs both versions, rollback needs the old one, and `parent_version` plus `change_notes` keep the history readable without digging through git.

**Comparison is case by case, not by total score.** v1 → v2 raised accuracy by 5 points and still broke case-020. A score-only gate would have passed it. Regressions, fixes and still-failing cases are reported separately.

**Severity comes from the size of the drop, not from any single flip.** The first version of the gate failed on any regression. Then case-049 failed and passed on identical runs, which would have blocked unrelated changes at random. A single flip is now WARN and stays listed in the comment and report; only a drop past `CRITICAL_DELTA` blocks. The thresholds sit at roughly two and five flipped cases out of 60.

**Evaluation has four dimensions, and only one decides pass/fail.** The category match is objective, so it drives the status. Summary relevance, latency and tokens are measured per case and compared per run, so a prompt that keeps its pass rate but writes worse summaries, or doubles its token use, still shows up in the comment.

**The judge is a different model from the classifier.** A model shouldn't grade its own output, and on Groq each model has its own rate limit, so judging doesn't spend the classifier's budget. The judge compares each summary with the human reference on a written 1–5 rubric. A judge failure never changes a pass/fail result.

**Cases run concurrently, within the provider's limits.** Up to `EVAL_CONCURRENCY` requests run at once, each model's requests are spaced to `REQUESTS_PER_MINUTE`, and rate-limit errors are retried with the server's `Retry-After`. Latency is measured on the successful attempt only, so waiting on rate limits doesn't count as model latency.

**The required check runs on every pull request.** GitHub never starts a workflow with a path filter on unrelated pull requests, and a required check that never starts leaves them blocked forever. The job therefore always runs, finds the changed prompt files itself, and finishes in seconds when there are none.

**Baselines come from committed results, not a fresh run per pull request.** Re-running the baseline would double API usage, which matters on Groq's free tier. The trade-off is that the baseline reflects the model as it behaved when the results were recorded. If a pull request fails on cases unrelated to its change, re-run the baseline before trusting the result.

**Single-run flips are treated as noise until they repeat.** Temperature 0 is not fully deterministic: case-049 failed in one run and passed in another with the same prompt and data. `python -m src.storage flaky` lists every case like it.

**JSON is the record; SQLite is an index.** Each run is a readable JSON file in `results/`, committed with the code. `results/history.db` is built from those files for cross-run questions, gitignored, and rebuilt automatically whenever its tables change.

**Setup errors fail fast.** A missing `GROQ_API_KEY` stops the pipeline immediately. Before this check, a missing secret in CI produced "59 regressions," which looked like the model breaking rather than a configuration problem.

**Drift warns but does not block.** Drift is a trend across runs, not a fault in the change under review, so it appears in Slack and the pull request comment without failing the check.

**The dataset is 60 balanced cases with every hard case kept.** 15 per category makes per-category accuracy meaningful. The set was trimmed from 100 to fit the free tier's daily token budget; easy cases were cut first because edge cases are where regressions appear.

**Iteration stopped at v4.** Each version from v2 to v4 fixed one case and broke another, mostly because rules keyed on phrasing ("Can I…?") caught emails they weren't written for. v4 was chosen because its only miss is a genuinely ambiguous two-request email (case-049), while v3 missed a clear-cut account question. case-049 keeps its `billing` label; changing it after seeing results would move the goalposts.

**The provider is swappable.** The classifier uses the OpenAI SDK against Groq's OpenAI-compatible endpoint. Changing providers means changing the base URL, the API key and the `model` field in the prompt file.

**The report is a single self-contained HTML file.** The trend chart is inline SVG, so the report needs no network access and works as a CI artifact.

**The container carries no secrets and runs as a regular user.** `.dockerignore` excludes `.env`, CI checks the built image for it, and all keys and thresholds arrive as environment variables. The image is built and tested in CI because the development machine cannot run current Docker Desktop.

## Known limitations

- **Free-tier budget:** a v4 run uses roughly 50,000 classifier input tokens, plus the judge's calls on a separate model, which allows only a few full runs per day.
- **No drift history from CI:** CI results are not committed, and there are no scheduled runs yet, so drift detection only sees runs committed by hand.
- **No hosted report:** Slack and pull request links go to the Actions run, where the report is a downloadable artifact, not a hosted page.
- **Two few-shot examples are near-duplicates of golden cases:** the billing example resembles case-001, and the "Spanish interface" example resembles case-033. A v5 that only replaces them would show whether those cases pass on the rules alone.
- **The judge is a model too:** its scores haven't been checked against human ratings, so treat a change of a few tenths in summary relevance as noise. Runs from before the judge was added have no summary scores.
- **One category per email:** emails with two requests, like case-049, have no single correct label.

## Repository layout

```
prompts/                    versioned prompt files (v1–v4)
data/golden.json            golden dataset (versioned, 60 cases)
data/reserve_cases.json     labeled cases not yet in the golden set
results/                    saved eval runs and comparisons (committed; baselines live here)
results/history.db          SQLite index built from results/ (gitignored)
reports/                    generated HTML reports (gitignored)
docs/                       blog post write-up and its chart
src/config.py               PromptConfig and the output contract (Pydantic)
src/classifier.py           the LLM feature under test
src/llm.py                  shared async client: concurrency, rate limits, retries, latency and tokens
src/storage.py              SQLite index and cross-run queries
src/evals/run_eval.py       scores a prompt version on the golden dataset, every dimension per case
src/evals/judge.py          LLM-as-judge for summary relevance (1-5)
src/evals/compare.py        case-by-case diff between two runs, severity thresholds
src/evals/report.py         HTML diff report and PASS/WARN/FAIL status
src/evals/alerts.py         Slack incoming-webhook alerts
src/evals/drift.py          rolling-average drift detection
src/evals/pipeline.py       eval → compare → report → drift → Slack
.github/workflows/          regression check (required) and Docker build
Dockerfile, .dockerignore   container packaging
```
