"""Runs the eval harness (task #9) against real dataset cases through the
live Anthropic API. NOT part of the pytest suite. Costs real money — this
is the single most expensive script in the repo: the full dataset is 155
documents, and a claim_form case alone can run 8-15+ LLM calls across
Classifier/Claims/Policy RAG/Fraud Detection/Orchestrator. DO NOT run the
full dataset without deliberately choosing to; use --limit or --category
for a cheap sample first.

`--combine-from-checkpoints` costs nothing — it reads already-completed
cases straight from the persistent checkpointer (`graph.get_state`, no
`invoke()`) and writes one merged EVAL_REPORT.md. Use it after running
separate `--category` stages to get a single full-dataset report without
paying to re-run anything.

Usage:
    uv run python -m backend.scripts.run_evals --limit 10
    uv run python -m backend.scripts.run_evals --category id_documents
    uv run python -m backend.scripts.run_evals --category id_documents discharge_summaries prescriptions policy_amendments unknown
    uv run python -m backend.scripts.run_evals            # full 155-doc dataset — expensive
    uv run python -m backend.scripts.run_evals --combine-from-checkpoints   # free — merges prior stages
"""

import argparse
import sys

from backend.config import REPO_ROOT, get_settings
from backend.evals.ground_truth import load_eval_cases
from backend.evals.harness import read_case_result_from_checkpoint, run_eval_suite
from backend.evals.report import format_report
from backend.evals.scoring import weighted_score
from backend.graph.pipeline import build_case_graph


def main() -> None:
    # Windows' default console codepage (cp1252) can't encode arbitrary
    # Unicode (a real run crashed here on U+2265 in the report text, after
    # the report file itself had already been written safely in UTF-8) —
    # reconfigure stdout defensively so a stray non-ASCII character in
    # future report text degrades to a replacement char instead of
    # crashing the whole run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (cheap sample).")
    parser.add_argument(
        "--category", type=str, nargs="+", default=None, help="Only run cases from these categories (one or more)."
    )
    parser.add_argument(
        "--combine-from-checkpoints",
        action="store_true",
        help="Free — read every case's already-persisted result from the checkpointer instead of invoking the API.",
    )
    args = parser.parse_args()

    settings = get_settings()

    cases = load_eval_cases(settings=settings)
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        raise SystemExit("No matching eval cases — check --category spelling.")

    graph = build_case_graph(settings=settings)

    if args.combine_from_checkpoints:
        print(f"Reading {len(cases)} already-completed case(s) from checkpoints. No API cost.")
        results = [read_case_result_from_checkpoint(graph, case) for case in cases]
    else:
        if not settings.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in first.")
        print(f"Running {len(cases)} eval case(s) against the live API. This costs real money.")
        results = run_eval_suite(graph, cases)

    score = weighted_score(results)

    report = format_report(results, score)
    output_path = REPO_ROOT / "EVAL_REPORT.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
