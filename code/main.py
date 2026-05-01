from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import load_environment
from models import PipelineResult, Ticket, TicketTrace
from pipeline import OrchestratePipeline, TicketAnalysis


console = Console()
STAGE_LABELS = {
    "language_normalization": "Language",
    "guard": "Guard",
    "triage": "Triage",
    "retrieval": "Retrieval",
    "hallucination_check": "Hallucination",
    "escalation_judge": "Escalation",
    "localization": "Localization",
    "complete": "Complete",
}


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


def write_knowledge_gap_report(analyses: list[TicketAnalysis], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_path = log_dir / f"knowledge_gaps_{timestamp}.json"
    latest_path = log_dir / "knowledge_gaps_latest.json"
    gaps = []
    for index, analysis in enumerate(analyses, start=1):
        top_score = analysis.chunks[0].score if analysis.chunks else 0.0
        reasons = []
        if analysis.final.confidence < 0.52:
            reasons.append("low_final_confidence")
        if not analysis.chunks and not analysis.triage.needs_escalation:
            reasons.append("no_retrieved_context")
        if not analysis.hallucination.is_grounded:
            reasons.append("hallucination_verifier_flag")
        if analysis.retrieval_attempts > 0:
            reasons.append("query_rewrite_needed")
        if top_score and top_score < 0.035:
            reasons.append("weak_top_retrieval_score")
        if not reasons:
            continue
        gaps.append(
            {
                "ticket_index": index,
                "company": analysis.ticket.company,
                "subject": analysis.ticket.subject,
                "product_area": analysis.final.product_area,
                "status": analysis.final.status,
                "confidence": analysis.final.confidence,
                "top_retrieval_score": top_score,
                "reasons": reasons,
                "suggested_corpus_need": _suggest_corpus_need(analysis),
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "gap_count": len(gaps),
        "gaps": gaps,
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    run_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")
    return run_path


def _suggest_corpus_need(analysis: TicketAnalysis) -> str:
    if analysis.final.product_area == "security_compliance":
        return "Add explicit support process docs for infosec, procurement, and vendor security form requests."
    if analysis.triage.intents:
        return f"Add or improve docs for {analysis.triage.domain} / {analysis.final.product_area} / {', '.join(analysis.triage.intents[:3])}."
    return f"Add more documentation for {analysis.triage.domain} / {analysis.final.product_area}."


def run_batch_mode(repo_root: Path) -> None:
    input_path = repo_root / "support_tickets" / "support_tickets.csv"
    output_path = repo_root / "support_tickets" / "output.csv"
    trace_dir = repo_root / "log"

    pipeline = OrchestratePipeline(repo_root=repo_root)
    tickets = read_tickets(input_path)
    analyses: list[TicketAnalysis] = []
    status_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {label: 0 for label in STAGE_LABELS}

    console.print(Panel.fit(f"{pipeline.describe_backend()}\nProcessing {len(tickets)} tickets", title="Support Agent Batch"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        ticket_task = progress.add_task("Batch tickets", total=len(tickets))
        stage_task = progress.add_task("Pipeline stages", total=len(tickets) * len(STAGE_LABELS))
        for index, ticket in enumerate(tickets, start=1):
            progress.update(ticket_task, description=f"Ticket {index}/{len(tickets)}: {ticket.subject or ticket.company}")

            def on_stage(stage: str) -> None:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                progress.update(stage_task, advance=1, description=f"Stage: {STAGE_LABELS.get(stage, stage)}")

            analysis = pipeline.analyze(ticket, stage_callback=on_stage)
            analyses.append(analysis)
            status_counts[analysis.final.status] = status_counts.get(analysis.final.status, 0) + 1
            progress.update(ticket_task, advance=1)

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
    gap_path = write_knowledge_gap_report(analyses, trace_dir)
    _print_batch_summary(results, output_path, trace_path, gap_path, status_counts)


def _print_batch_summary(
    results: list[PipelineResult],
    output_path: Path,
    trace_path: Path,
    gap_path: Path,
    status_counts: dict[str, int],
) -> None:
    table = Table(title="Batch Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Rows written", str(len(results)))
    table.add_row("Replied", str(status_counts.get("replied", 0)))
    table.add_row("Escalated", str(status_counts.get("escalated", 0)))
    table.add_row("Output", str(output_path))
    table.add_row("Trace", str(trace_path))
    table.add_row("Knowledge gaps", str(gap_path))
    console.print(table)


def run_interactive_mode(repo_root: Path) -> None:
    pipeline = OrchestratePipeline(repo_root=repo_root)
    console.print(Panel.fit(pipeline.describe_backend(), title="Interactive Support Triage CLI"))
    console.print("Type 'exit' at any prompt to quit.")

    while True:
        company = console.input("\nCompany [HackerRank/Claude/Visa/None]: ").strip() or "None"
        if company.lower() == "exit":
            break
        subject = console.input("Subject: ").strip()
        if subject.lower() == "exit":
            break
        issue = console.input("Issue: ").strip()
        if issue.lower() == "exit":
            break

        ticket = Ticket(issue=issue, subject=subject, company=company)
        with console.status("Starting pipeline...", spinner="dots") as status:
            def on_stage(stage: str) -> None:
                status.update(f"Passing stage: {STAGE_LABELS.get(stage, stage)}")

            analysis = pipeline.analyze(ticket, stage_callback=on_stage)

        _print_interactive_analysis(analysis)


def _print_interactive_analysis(analysis: TicketAnalysis) -> None:
    table = Table(title="Pipeline Result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("status", analysis.final.status)
    table.add_row("product_area", analysis.final.product_area)
    table.add_row("request_type", analysis.final.request_type)
    table.add_row("confidence", f"{analysis.final.confidence:.2f}")
    table.add_row("hallucination_score", f"{analysis.hallucination.score:.2f}")
    table.add_row("grounded", str(analysis.hallucination.is_grounded))
    table.add_row("retrieved_chunks", str(len(analysis.chunks)))
    console.print(table)
    console.print(Panel(analysis.final.response, title="Response"))
    console.print(Panel(analysis.final.justification, title="Justification"))
    if analysis.hallucination.unsupported_claims:
        console.print(Panel("\n".join(analysis.hallucination.unsupported_claims), title="Unsupported Claims", style="red"))


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
