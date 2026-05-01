from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import load_environment
from models import PipelineResult, Ticket, TicketTrace
from pipeline import OrchestratePipeline


def read_tickets(csv_path: Path) -> list[Ticket]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        tickets = []
        for row in reader:
            tickets.append(
                Ticket(
                    issue=(row.get("Issue") or "").strip(),
                    subject=(row.get("Subject") or "").strip(),
                    company=(row.get("Company") or "None").strip() or "None",
                )
            )
        return tickets


def write_output(results: list[PipelineResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "product_area", "response", "justification", "request_type"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "product_area": result.product_area,
                    "response": _csv_cell(result.response),
                    "justification": _csv_cell(result.justification),
                    "request_type": result.request_type,
                }
            )


def _csv_cell(value: str) -> str:
    return " ".join(str(value).split())


def write_trace_logs(traces: list[TicketTrace], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_path = log_dir / f"ticket_trace_{timestamp}.json"
    latest_path = log_dir / "ticket_trace_latest.json"

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "ticket_count": len(traces),
        "traces": [asdict(trace) for trace in traces],
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    run_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")
    return run_path


def run_batch_mode(repo_root: Path) -> None:
    input_path = repo_root / "support_tickets" / "support_tickets.csv"
    output_path = repo_root / "support_tickets" / "output.csv"
    trace_dir = repo_root / "log"

    pipeline = OrchestratePipeline(repo_root=repo_root)
    tickets = read_tickets(input_path)
    analyses = [pipeline.analyze(ticket) for ticket in tickets]
    results = [
        PipelineResult(
            status=analysis.final.status,
            product_area=analysis.final.product_area,
            response=analysis.final.response,
            justification=analysis.final.justification,
            request_type=analysis.final.request_type,
            confidence=analysis.final.confidence,
        )
        for analysis in analyses
    ]
    traces = [pipeline.build_trace(analysis, ticket_index=index) for index, analysis in enumerate(analyses, start=1)]
    write_output(results, output_path)
    trace_path = write_trace_logs(traces, trace_dir)
    print(f"Wrote {len(results)} rows to {output_path}")
    print(f"Wrote ticket traces to {trace_path}")


def run_interactive_mode(repo_root: Path) -> None:
    pipeline = OrchestratePipeline(repo_root=repo_root)
    print("Interactive support triage CLI")
    print("Type 'exit' at any prompt to quit.")
    print(pipeline.describe_backend())

    while True:
        company = input("\nCompany [HackerRank/Claude/Visa/None]: ").strip() or "None"
        if company.lower() == "exit":
            break
        subject = input("Subject: ").strip()
        if subject.lower() == "exit":
            break
        issue = input("Issue: ").strip()
        if issue.lower() == "exit":
            break

        ticket = Ticket(issue=issue, subject=subject, company=company)
        result = pipeline.run(ticket)

        print("\nResult")
        print(f"status: {result.status}")
        print(f"product_area: {result.product_area}")
        print(f"request_type: {result.request_type}")
        print(f"justification: {result.justification}")
        print("response:")
        print(result.response)


def run_tui_mode(repo_root: Path) -> None:
    from tui_app import SupportAssistantApp

    app = SupportAssistantApp(repo_root=repo_root)
    app.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate support agent")
    parser.add_argument(
        "--mode",
        choices=["batch", "interactive", "tui"],
        default="batch",
        help="Run the CSV pipeline or start the interactive CLI.",
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    load_environment(repo_root)
    args = build_parser().parse_args()
    if args.mode == "tui":
        run_tui_mode(repo_root)
        return
    if args.mode == "interactive":
        run_interactive_mode(repo_root)
        return
    run_batch_mode(repo_root)


if __name__ == "__main__":
    main()
