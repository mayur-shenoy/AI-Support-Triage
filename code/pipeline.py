from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import load_environment
from escalation_judge import EscalationJudge
from guard_agent import GuardAgent
from hallucination_agent import HallucinationAgent
from language_support import LanguageProcessor
from llm_client import LLMClient
from models import GuardResult, HallucinationResult, LanguageState, PipelineResult, RetrievedChunk, ResponseDraft, Ticket, TicketTrace, TriageResult
from response_agent import ResponseAgent
from retriever import HybridRetriever
from triage_agent import TriageAgent


@dataclass(slots=True)
class TicketAnalysis:
    ticket: Ticket
    normalized_ticket: Ticket
    language: LanguageState
    guard: GuardResult
    triage: TriageResult
    chunks: list[RetrievedChunk]
    intent_chunks: dict[str, list[RetrievedChunk]]
    draft: ResponseDraft
    hallucination: HallucinationResult
    final: ResponseDraft
    retrieval_attempts: int
    retrieval_queries: list[str]


class OrchestratePipeline:
    RETRY_TRIGGER_CONFIDENCE = 0.52
    VERY_LOW_CONFIDENCE = 0.26

    def __init__(self, repo_root: Path) -> None:
        load_environment(repo_root)
        self.llm_client = LLMClient()
        self.guard_agent = GuardAgent()
        self.triage_agent = TriageAgent()
        self.retriever = HybridRetriever(repo_root=repo_root)
        self.response_agent = ResponseAgent(llm_client=self.llm_client)
        self.hallucination_agent = HallucinationAgent()
        self.language_processor = LanguageProcessor(llm_client=self.llm_client)
        self.escalation_judge = EscalationJudge()

    def analyze(self, ticket: Ticket, stage_callback: Callable[[str], None] | None = None) -> TicketAnalysis:
        self._emit(stage_callback, "language_normalization")
        normalized_ticket, language_state = self.language_processor.prepare_ticket(ticket)
        self._emit(stage_callback, "guard")
        guard = self.guard_agent.evaluate(normalized_ticket)
        self._emit(stage_callback, "triage")
        triage = self.triage_agent.classify(normalized_ticket)
        self._emit(stage_callback, "retrieval")
        triage, chunks, intent_chunks, draft, attempts_used, retrieval_queries = self._run_retrieval_loop(normalized_ticket, triage, guard)
        self._emit(stage_callback, "hallucination_check")
        draft, hallucination = self.hallucination_agent.apply(normalized_ticket, triage, draft, chunks)
        self._emit(stage_callback, "escalation_judge")
        final = self.escalation_judge.finalize(normalized_ticket, guard, triage, draft, chunks)
        if attempts_used > 0:
            final.justification = (
                f"{final.justification} Retrieval retry attempts used: {attempts_used}."
            ).strip()
        self._emit(stage_callback, "localization")
        localized_final = self.language_processor.localize_draft(final, language_state)
        self._emit(stage_callback, "complete")
        return TicketAnalysis(
            ticket=ticket,
            normalized_ticket=normalized_ticket,
            language=language_state,
            guard=guard,
            triage=triage,
            chunks=chunks,
            intent_chunks=intent_chunks,
            draft=draft,
            hallucination=hallucination,
            final=localized_final,
            retrieval_attempts=attempts_used,
            retrieval_queries=retrieval_queries,
        )

    def run(self, ticket: Ticket) -> PipelineResult:
        analysis = self.analyze(ticket)
        return PipelineResult(
            status=analysis.final.status,
            product_area=analysis.final.product_area,
            response=analysis.final.response,
            justification=analysis.final.justification,
            request_type=analysis.final.request_type,
            confidence=analysis.final.confidence,
        )

    def build_trace(self, analysis: TicketAnalysis, ticket_index: int) -> TicketTrace:
        top_score = analysis.chunks[0].score if analysis.chunks else 0.0
        return TicketTrace(
            ticket_index=ticket_index,
            input_ticket={
                "company": analysis.ticket.company,
                "subject": analysis.ticket.subject,
                "issue": analysis.ticket.issue,
                "language": analysis.ticket.language,
            },
            normalized_ticket={
                "company": analysis.normalized_ticket.company,
                "subject": analysis.normalized_ticket.subject,
                "issue": analysis.normalized_ticket.issue,
                "language": analysis.normalized_ticket.language,
            },
            language=asdict(analysis.language),
            pipeline_passed=analysis.guard.is_safe,
            guard=asdict(analysis.guard),
            triage=asdict(analysis.triage),
            retrieval={
                "attempts_used": analysis.retrieval_attempts,
                "queries": analysis.retrieval_queries,
                "retrieved_count": len(analysis.chunks),
                "top_score": top_score,
                "top_source_path": analysis.chunks[0].source_path if analysis.chunks else "",
                "per_intent": {
                    query: [
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_path": chunk.source_path,
                            "section_title": chunk.section_title,
                            "score": chunk.score,
                        }
                        for chunk in query_chunks
                    ]
                    for query, query_chunks in analysis.intent_chunks.items()
                },
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "section_id": chunk.section_id,
                        "section_title": chunk.section_title,
                        "source_path": chunk.source_path,
                        "source_url": chunk.source_url,
                        "supporting_source_paths": chunk.supporting_source_paths,
                        "graph_concepts": chunk.graph_concepts,
                        "graph_expansion_paths": chunk.graph_expansion_paths,
                        "title": chunk.title,
                        "score": chunk.score,
                    }
                    for chunk in analysis.chunks
                ],
            },
            draft=asdict(analysis.draft),
            hallucination=asdict(analysis.hallucination),
            final=asdict(analysis.final),
        )

    def describe_backend(self) -> str:
        return self.retriever.describe_backend()

    def _retrieve_for_triage(self, ticket: Ticket, triage: TriageResult) -> tuple[list[RetrievedChunk], dict[str, list[RetrievedChunk]]]:
        queries = triage.intent_queries or [ticket.text]
        merged: dict[str, RetrievedChunk] = {}
        per_intent: dict[str, list[RetrievedChunk]] = {}
        for query in queries:
            retrieved = self.retriever.retrieve(query, triage.domain, limit=4)
            per_intent[query] = retrieved
            for chunk in retrieved:
                existing = merged.get(chunk.chunk_id)
                if not existing or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk
        return sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:7], per_intent

    def _run_retrieval_loop(
        self,
        ticket: Ticket,
        triage: TriageResult,
        guard: GuardResult,
    ) -> tuple[TriageResult, list[RetrievedChunk], dict[str, list[RetrievedChunk]], ResponseDraft, int, list[str]]:
        if triage.needs_escalation:
            draft = ResponseDraft(
                status="escalated",
                product_area=triage.product_area,
                response="This request needs human support review.",
                justification="Skipped retrieval because triage required escalation.",
                request_type=triage.request_type,
                confidence=0.90,
            )
            return triage, [], {}, draft, 0, list(triage.intent_queries or [ticket.text])

        chunks, intent_chunks = self._retrieve_for_triage(ticket, triage)
        draft = self.response_agent.draft(ticket, triage, chunks, intent_chunks=intent_chunks)
        best_triad = (triage, chunks, intent_chunks, draft)
        attempts_used = 0
        retrieval_queries = list(triage.intent_queries or [ticket.text])

        if not self._should_retry(guard, triage, draft):
            return triage, chunks, intent_chunks, draft, attempts_used, retrieval_queries

        max_retries = self._max_retry_count(draft.confidence)
        current_triage = triage

        for attempt_index in range(max_retries):
            rewritten_queries = self.triage_agent.rewrite_queries(ticket, current_triage, best_triad[1], attempt_index)
            if not rewritten_queries or [query.lower() for query in rewritten_queries] == [query.lower() for query in current_triage.intent_queries]:
                break

            candidate_triage = replace(
                current_triage,
                intent_queries=rewritten_queries,
                query_rewrite_attempts=attempt_index + 1,
            )
            candidate_chunks, candidate_intent_chunks = self._retrieve_for_triage(ticket, candidate_triage)
            candidate_draft = self.response_agent.draft(
                ticket,
                candidate_triage,
                candidate_chunks,
                intent_chunks=candidate_intent_chunks,
            )
            attempts_used = attempt_index + 1
            retrieval_queries = rewritten_queries

            if self._is_better_attempt(candidate_draft, candidate_chunks, best_triad[3], best_triad[1]):
                best_triad = (candidate_triage, candidate_chunks, candidate_intent_chunks, candidate_draft)
                current_triage = candidate_triage

            if candidate_draft.confidence >= self.RETRY_TRIGGER_CONFIDENCE:
                break

        best_triage, best_chunks, best_intent_chunks, best_draft = best_triad
        return best_triage, best_chunks, best_intent_chunks, best_draft, attempts_used, retrieval_queries

    def _should_retry(self, guard: GuardResult, triage: TriageResult, draft: ResponseDraft) -> bool:
        if not guard.is_safe:
            return False
        if triage.request_type == "invalid" or triage.needs_escalation:
            return False
        return draft.confidence < self.RETRY_TRIGGER_CONFIDENCE

    def _max_retry_count(self, confidence: float) -> int:
        if confidence < self.VERY_LOW_CONFIDENCE:
            return 2
        if confidence < self.RETRY_TRIGGER_CONFIDENCE:
            return 1
        return 0

    @staticmethod
    def _is_better_attempt(
        candidate_draft: ResponseDraft,
        candidate_chunks: list[RetrievedChunk],
        best_draft: ResponseDraft,
        best_chunks: list[RetrievedChunk],
    ) -> bool:
        candidate_top = candidate_chunks[0].score if candidate_chunks else 0.0
        best_top = best_chunks[0].score if best_chunks else 0.0
        if candidate_draft.confidence > best_draft.confidence + 0.02:
            return True
        if abs(candidate_draft.confidence - best_draft.confidence) <= 0.02 and candidate_top > best_top:
            return True
        return False

    @staticmethod
    def _emit(stage_callback: Callable[[str], None] | None, stage: str) -> None:
        if stage_callback:
            stage_callback(stage)
