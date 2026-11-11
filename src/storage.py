"""SQLite index over the JSON results, for questions that span many runs.

The JSON files in results/ stay the source of truth: they're readable and git-friendly.
This builds a SQLite database from them (results/history.db, gitignored), which can be
deleted and rebuilt at any time. It rebuilds itself when its table layout changes.

Usage:
    python -m src.storage sync               # add any new results files to the database
    python -m src.storage runs               # every run: pass rate, summary relevance, latency, tokens
    python -m src.storage versions           # averages per prompt version
    python -m src.storage case case-049      # one case's answer and summary score in every run
    python -m src.storage flaky              # cases that both passed and failed on the same prompt and data
    python -m src.storage weak               # summaries the judge scored 1-2
    python -m src.storage sql "SELECT ..."   # your own read-only query
    python -m src.storage sync --rebuild     # delete the database and build it again
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
DB_PATH = RESULTS_DIR / "history.db"
SCHEMA_VERSION = 2  # bump whenever the tables change; the database then rebuilds itself from the JSON

RUN_COLUMNS = ["run_id", "prompt_version", "model", "judge_model", "dataset_version", "run_at", "passed", "total",
               "accuracy", "errors", "summary_relevance", "latency_mean_ms", "latency_p95_ms", "tokens_total",
               "wall_time_s", "source_file"]
CASE_COLUMNS = ["run_id", "case_id", "difficulty", "expected", "predicted", "passed", "error", "summary",
                "expected_summary", "latency_ms", "prompt_tokens", "completion_tokens", "judge_score",
                "judge_reason", "judge_error", "judge_latency_ms", "judge_prompt_tokens", "judge_completion_tokens"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,   -- the results file name, e.g. v4_20261110-203958
    prompt_version    TEXT NOT NULL,
    model             TEXT,
    judge_model       TEXT,               -- NULL when summaries weren't scored
    dataset_version   TEXT,
    run_at            TEXT NOT NULL,
    passed            INTEGER,
    total             INTEGER,
    accuracy          REAL,
    errors            INTEGER,            -- cases where the classifier call failed
    summary_relevance REAL,               -- mean judge score, 1-5
    latency_mean_ms   REAL,
    latency_p95_ms    REAL,
    tokens_total      INTEGER,            -- classifier + judge, prompt + completion
    wall_time_s       REAL,
    source_file       TEXT
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id                  TEXT NOT NULL REFERENCES runs(run_id),
    case_id                 TEXT NOT NULL,
    difficulty              TEXT,
    expected                TEXT,
    predicted               TEXT,
    passed                  INTEGER,
    error                   TEXT,
    summary                 TEXT,
    expected_summary        TEXT,
    latency_ms              INTEGER,
    prompt_tokens           INTEGER,
    completion_tokens       INTEGER,
    judge_score             INTEGER,
    judge_reason            TEXT,
    judge_error             TEXT,
    judge_latency_ms        INTEGER,
    judge_prompt_tokens     INTEGER,
    judge_completion_tokens INTEGER,
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_case ON case_results(case_id);
"""


def connect() -> sqlite3.Connection:
    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        has_tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
        conn.close()
        if has_tables:  # an older layout: it's only an index, so start over from the JSON files
            print(f"The database layout changed, so {DB_PATH} is being rebuilt from the results files.")
            DB_PATH.unlink()
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.row_factory = sqlite3.Row
    return conn


def run_row(run_id: str, run: dict, path: Path) -> dict:
    relevance = run.get("summary_relevance") or {}
    latency = run.get("latency_ms") or {}
    tokens = run.get("tokens")
    return {
        "run_id": run_id, "prompt_version": run["prompt_version"], "model": run.get("model"),
        "judge_model": run.get("judge_model"), "dataset_version": run.get("dataset_version"),
        "run_at": run["run_at"], "passed": run["passed"], "total": run["total"], "accuracy": run["accuracy"],
        "errors": sum(1 for c in run["cases"] if c.get("error")),
        "summary_relevance": relevance.get("mean"), "latency_mean_ms": latency.get("mean"),
        "latency_p95_ms": latency.get("p95"), "tokens_total": sum(tokens.values()) if tokens else None,
        "wall_time_s": run.get("wall_time_s"), "source_file": str(path),
    }


def case_row(run_id: str, case: dict) -> dict:
    row = {column: case.get(column) for column in CASE_COLUMNS}
    row.update(run_id=run_id, case_id=case["id"], passed=int(case["passed"]))
    return row


def sync(rebuild: bool = False) -> int:
    """Add every results file not yet in the database. Returns how many runs were added."""
    if rebuild and DB_PATH.exists():
        DB_PATH.unlink()
    conn = connect()
    known = {row["run_id"] for row in conn.execute("SELECT run_id FROM runs")}
    insert_run = f"INSERT INTO runs ({', '.join(RUN_COLUMNS)}) VALUES ({', '.join(':' + c for c in RUN_COLUMNS)})"
    insert_case = (f"INSERT INTO case_results ({', '.join(CASE_COLUMNS)}) "
                   f"VALUES ({', '.join(':' + c for c in CASE_COLUMNS)})")
    added = 0
    for path in sorted(RESULTS_DIR.glob("v*_2*.json")):
        if path.stem in known:
            continue
        run = json.loads(path.read_text())
        conn.execute(insert_run, run_row(path.stem, run, path))
        conn.executemany(insert_case, [case_row(path.stem, case) for case in run["cases"]])
        added += 1
    conn.commit()
    conn.close()
    return added


def show(rows: list[sqlite3.Row], empty: str = "No rows.") -> None:
    """Print query results as a simple aligned table."""
    if not rows:
        print(empty)
        return
    headers = rows[0].keys()
    values = [["" if row[h] is None else str(row[h]) for h in headers] for row in rows]
    widths = [min(60, max(len(h), *(len(v[i]) for v in values))) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for v in values:
        print("  ".join(cell[:w].ljust(w) for cell, w in zip(v, widths)))


QUERIES = {
    "runs": """
        SELECT run_at, prompt_version AS prompt, dataset_version AS dataset, passed || '/' || total AS score,
               ROUND(accuracy * 100, 1) AS pass_pct, ROUND(summary_relevance, 2) AS summary_1to5,
               ROUND(latency_mean_ms) AS latency_ms, latency_p95_ms AS p95_ms, tokens_total AS tokens, errors
        FROM runs ORDER BY run_at""",
    "versions": """
        SELECT prompt_version AS prompt, dataset_version AS dataset, COUNT(*) AS runs,
               ROUND(AVG(accuracy) * 100, 1) AS avg_pass_pct, ROUND(MIN(accuracy) * 100, 1) AS worst_pass_pct,
               ROUND(AVG(summary_relevance), 2) AS avg_summary_1to5, ROUND(AVG(latency_mean_ms)) AS avg_latency_ms
        FROM runs WHERE errors < total  -- leave out runs where every call failed
        GROUP BY prompt_version, dataset_version ORDER BY prompt_version""",
    "case": """
        SELECT r.run_at, r.prompt_version AS prompt, c.expected, COALESCE(c.predicted, 'error') AS predicted,
               CASE WHEN c.passed THEN 'PASS' ELSE 'FAIL' END AS result, c.judge_score AS summary_1to5, c.summary
        FROM case_results c JOIN runs r USING (run_id)
        WHERE c.case_id = ? ORDER BY r.run_at""",
    # A case is flaky when the same prompt on the same dataset both passed and failed it.
    # API errors are left out: they say nothing about the model's answer.
    "flaky": """
        SELECT c.case_id, r.prompt_version AS prompt, r.dataset_version AS dataset,
               SUM(c.passed) AS passes, COUNT(*) - SUM(c.passed) AS fails,
               GROUP_CONCAT(DISTINCT c.predicted) AS answers_seen
        FROM case_results c JOIN runs r USING (run_id)
        WHERE c.error IS NULL
        GROUP BY c.case_id, r.prompt_version, r.dataset_version
        HAVING passes > 0 AND fails > 0
        ORDER BY r.prompt_version, c.case_id""",
    "weak": """
        SELECT r.run_at, r.prompt_version AS prompt, c.case_id, c.judge_score AS score, c.judge_reason AS reason,
               c.summary
        FROM case_results c JOIN runs r USING (run_id)
        WHERE c.judge_score <= 2 ORDER BY r.run_at DESC, c.case_id""",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Query eval history stored in SQLite.")
    sub = parser.add_subparsers(dest="command", required=True)
    sync_cmd = sub.add_parser("sync", help="Add new results files to the database")
    sync_cmd.add_argument("--rebuild", action="store_true", help="Delete the database and rebuild it")
    sub.add_parser("runs", help="List every run with all its scores")
    sub.add_parser("versions", help="Averages per prompt version")
    case_cmd = sub.add_parser("case", help="Show one case across every run")
    case_cmd.add_argument("case_id", help="e.g. case-049")
    sub.add_parser("flaky", help="Cases that both passed and failed on the same prompt and data")
    sub.add_parser("weak", help="Summaries the judge scored 1-2")
    sql_cmd = sub.add_parser("sql", help="Run your own read-only SQL query")
    sql_cmd.add_argument("query")
    args = parser.parse_args(argv)

    added = sync(rebuild=getattr(args, "rebuild", False))  # always up to date before querying
    if args.command == "sync":
        total = connect().execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        print(f"Added {added} new run(s). The database now holds {total} run(s): {DB_PATH}")
        return

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)  # read-only for queries
    conn.row_factory = sqlite3.Row
    empty = {
        "case": f"No results for {getattr(args, 'case_id', '')}.",
        "flaky": "No flaky cases yet. A case needs at least two runs of the same prompt to be compared.",
        "weak": "No summaries scored 1-2. (Runs before the LLM judge was added have no summary scores.)",
    }
    try:
        if args.command == "sql":
            show(conn.execute(args.query).fetchall())
        elif args.command == "case":
            show(conn.execute(QUERIES["case"], (args.case_id,)).fetchall(), empty["case"])
        else:
            show(conn.execute(QUERIES[args.command]).fetchall(), empty.get(args.command, "No rows."))
    except sqlite3.Error as err:
        sys.exit(f"SQL error: {err}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()