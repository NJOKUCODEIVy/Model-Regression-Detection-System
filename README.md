# Model Regression Detection System

A lightweight CI system for detecting prompt regressions in an LLM-based customer support email classifier.

The project evaluates prompt changes against a fixed golden dataset, compares each candidate prompt to the current baseline, and blocks merge decisions when a change causes a real regression.

## What this project does

The classifier reads a customer support email and returns:

- a category: `billing`, `technical`, `account`, or `general`
- a one-sentence summary

Each prompt version is stored in its own YAML file under `prompts/`.

When a pull request changes a prompt:

1. CI runs the evaluation against the golden dataset
2. It compares the candidate prompt to the current baseline
3. It reports pass, warn, or fail status for the change
4. It posts a summary to GitHub and Slack
5. It blocks the merge if the change introduces regressions

## Why it exists

Prompt changes often look safe in review but quietly break edge cases. This project treats prompt quality as a versioned, measurable artifact.

Instead of trusting a single score, it checks:

- which cases passed before and fail now
- which fixes offset regressions
- whether the overall trend is drifting over time

## Current status

The baseline prompt is currently `v4`.

This project uses the `openai/gpt-oss-120b` model through Groq's OpenAI-compatible API, with temperature set to `0`.

## Workflow

1. Create a new prompt file such as `prompts/v5.yaml`
2. Set the required metadata such as version, created date, parent version, and change notes
3. Open a pull request
4. The regression workflow runs automatically
5. Results are compared against the baseline
6. Merge is blocked for failing regressions

## Evaluation process

The system evaluates a prompt against the golden dataset in `data/golden.json`.

Each record includes:

- an ID
- the email text
- the expected category and summary
- difficulty information
- notes for reasoning and maintenance

The system then generates a case-by-case diff to identify:

- fixed cases
- regressed cases
- still-failing cases
- overall pass/warn/fail status

## Status definitions

| Status | Meaning | Merge result |
| --- | --- | --- |
| PASS | No baseline-passing case fails in the candidate | Allowed |
| WARN | Regressions are within the allowed threshold and offset by fixes | Allowed with review |
| FAIL | Too many regressions or the candidate performs worse than the baseline | Blocked |

## Local setup

Requirements:

- Python 3.11+
- A Groq API key
- Optional Slack webhook for alerts

### Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## Useful commands

```bash
python -m src.evals.run_eval --prompt prompts/v4.yaml
python -m src.evals.compare --baseline v3 --candidate v4
python -m src.evals.report --baseline v3 --candidate v4
python -m src.evals.alerts --baseline v3 --candidate v4 --dry-run
python -m src.evals.drift [--prompt v4] [--simulate]
python -m src.evals.pipeline --candidate prompts/v5.yaml --baseline v4
```

## Repository structure

```text
prompts/                  Versioned prompt files
data/                    Golden dataset and reserve cases
results/                 Saved evaluation results
reports/                 Generated HTML reports
src/                     Project source code
.github/workflows/       GitHub Actions for regression checks and Docker build
Dockerfile               Container definition
requirements.txt         Python dependencies
```

## Architecture highlights

- Prompts are versioned files, not edited in place
- Evaluation is case-by-case, not just score-based
- Baselines come from committed result files
- CI runs automatically on pull requests
- Drift detection watches for repeated model quality decay
- The report is stored as a self-contained HTML artifact

## Known limitations

- The free-tier API budget is limited
- Drift history is not yet fully available from CI-only runs
- Summaries are stored but not scored as part of the outcome
- Some edge-case emails are naturally ambiguous

## Contributing

To improve the system:

1. Add or adjust prompt files under `prompts/`
2. Re-run evaluation locally when needed
3. Validate the behavior against the dataset
4. Open a pull request for review
5. Let CI decide whether the change is safe

## License

This project does not currently declare a license file in the repository.
