from __future__ import annotations

import asyncio
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from config import load_environment
from incident_retriever import SimilarIncidentRetriever
from models import IncidentMatch, Ticket
from pipeline import OrchestratePipeline, TicketAnalysis


TokenCallback = Callable[[str], Awaitable[None]]
StageCallback = Callable[[str], None]


@dataclass(slots=True)
class StreamConfig:
    provider: str
    model: str
    max_tokens: int


@dataclass(slots=True)
class SupportStateSummary:
    status: str
    product_area: str
    request_type: str
    risk_level: str
    confidence: float
    hallucination_score: float
    grounded: bool
    retrieval_attempts: int
    top_retrieval_score: float
    references: list[str]


class StreamingSupportAssistant:
    MIN_SUPPORT_CONFIDENCE = 0.025

    def __init__(self, repo_root: Path) -> None:
        load_environment(repo_root)
        self.repo_root = repo_root
        self.pipeline = OrchestratePipeline(repo_root=repo_root)
        self.incident_retriever = SimilarIncidentRetriever(
            repo_root=repo_root,
            embedding_model=self.pipeline.retriever.embedding_model,
        )
        provider = os.getenv("ORCHESTRATE_LLM_PROVIDER", "").strip().lower()
        self.config = StreamConfig(
            provider=provider or "anthropic",
            model=os.getenv("ORCHESTRATE_TUI_MODEL", os.getenv("ORCHESTRATE_LLM_MODEL", "claude-3-5-haiku-latest")),
            max_tokens=int(os.getenv("ORCHESTRATE_TUI_MAX_TOKENS", "700")),
        )
        self.client = self._build_client()

    def describe_backend(self) -> str:
        return (
            f"{self.pipeline.describe_backend()} "
            f"Streaming provider: {self.config.provider}. "
            f"Streaming model: {self.config.model}."
        )

    async def stream_reply(
        self,
        ticket: Ticket,
        on_token: TokenCallback,
    ) -> TicketAnalysis:
        analysis = await self.analyze_ticket(ticket)
        await self.stream_from_analysis(analysis, on_token)
        return analysis

    async def analyze_ticket(self, ticket: Ticket) -> TicketAnalysis:
        return await asyncio.to_thread(self.pipeline.analyze, ticket)

    async def analyze_ticket_with_stages(
        self,
        ticket: Ticket,
        on_stage: StageCallback,
    ) -> TicketAnalysis:
        return await asyncio.to_thread(self.pipeline.analyze, ticket, on_stage)

    async def stream_from_analysis(
        self,
        analysis: TicketAnalysis,
        on_token: TokenCallback,
    ) -> None:
        if os.getenv("ORCHESTRATE_TUI_REWRITE", "").strip().lower() not in {"1", "true", "yes"}:
            await self._stream_plain_text(analysis.final.response, on_token)
            return

        if self._should_use_canned_response(analysis):
            await self._stream_plain_text(self._canned_response(analysis), on_token)
            return

        if not self.client:
            await self._stream_plain_text(analysis.final.response, on_token)
            return

        system_prompt, user_prompt = self._build_prompts(analysis)
        if self.config.provider == "anthropic":
            await self._stream_with_anthropic(system_prompt, user_prompt, on_token)
            return
        await self._stream_with_openai_compatible(system_prompt, user_prompt, on_token)

    async def retrieve_similar_incidents(self, ticket: Ticket, limit: int = 5) -> list[IncidentMatch]:
        return await asyncio.to_thread(self.incident_retriever.retrieve, ticket, limit)

    async def ingest_csv(self, csv_path: Path) -> tuple[Path, int]:
        return await asyncio.to_thread(self._ingest_csv_sync, csv_path)

    async def save_incident(self, ticket: Ticket, analysis: TicketAnalysis) -> Path:
        return await asyncio.to_thread(self._save_incident_sync, ticket, analysis)

    def summarize_state(self, analysis: TicketAnalysis) -> SupportStateSummary:
        return SupportStateSummary(
            status=analysis.final.status,
            product_area=analysis.final.product_area,
            request_type=analysis.final.request_type,
            risk_level=self._risk_level(analysis),
            confidence=analysis.final.confidence,
            hallucination_score=analysis.hallucination.score,
            grounded=analysis.hallucination.is_grounded,
            retrieval_attempts=analysis.retrieval_attempts,
            top_retrieval_score=analysis.chunks[0].score if analysis.chunks else 0.0,
            references=self._references(analysis),
        )

    def _build_client(self) -> AsyncAnthropic | AsyncOpenAI | None:
        if self.config.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key or api_key.startswith("your_"):
                return None
            return AsyncAnthropic(api_key=api_key)

        if self.config.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key or api_key.startswith("your_"):
                return None
            return AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

        if self.config.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key or api_key.startswith("your_"):
                return None
            return AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1",
            )

        return None

    async def _stream_with_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        on_token: TokenCallback,
    ) -> None:
        assert isinstance(self.client, AsyncAnthropic)
        async with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                await on_token(text)

    async def _stream_with_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        on_token: TokenCallback,
    ) -> None:
        assert isinstance(self.client, AsyncOpenAI)
        stream = await self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                await on_token(delta)

    async def _stream_plain_text(self, text: str, on_token: TokenCallback) -> None:
        for token in self._tokenize_for_stream(text):
            await on_token(token)
            await asyncio.sleep(0)

    @staticmethod
    def _build_prompts(analysis: TicketAnalysis) -> tuple[str, str]:
        triage = analysis.triage
        target_language = analysis.language.source_language
        top_chunks = analysis.chunks[:4]
        context_parts = []
        for idx, chunk in enumerate(top_chunks, start=1):
            context_parts.append(
                f"[{idx}] {chunk.title}\nSource: {chunk.source_path}\nContent:\n{chunk.text[:2200]}"
            )
        context_text = "\n\n".join(context_parts) if context_parts else "No corpus context was retrieved."

        system_prompt = (
            "You are Support AI, a careful support engineer assistant. "
            "Answer using only the provided support corpus context. "
            "If the context is insufficient or the request is outside HackerRank, Claude, or Visa support, "
            "say it is out of scope and do not answer from prior knowledge. "
            "Do not invent policies, pricing, account actions, or security steps. "
            f"Respond in language code '{target_language}'."
        )
        user_prompt = (
            f"Original user query:\nSubject: {analysis.ticket.subject or '(none)'}\nIssue: {analysis.ticket.issue}\nCompany: {analysis.ticket.company}\nLanguage: {analysis.language.source_language}\n\n"
            f"Normalized English query for retrieval:\nSubject: {analysis.normalized_ticket.subject or '(none)'}\nIssue: {analysis.normalized_ticket.issue}\n\n"
            f"Triage:\n"
            f"- domain: {triage.domain}\n"
            f"- urgency: {triage.urgency}\n"
            f"- request_type: {triage.request_type}\n"
            f"- product_area: {triage.product_area}\n"
            f"- needs_escalation: {triage.needs_escalation}\n"
            f"- escalation_reason: {triage.escalation_reason or 'none'}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Respond to the user directly in a concise, helpful tone. "
            "If appropriate, mention that the case should be escalated."
        )
        return system_prompt, user_prompt

    def _should_use_canned_response(self, analysis: TicketAnalysis) -> bool:
        top_score = analysis.chunks[0].score if analysis.chunks else 0.0
        if not analysis.guard.is_safe:
            return True
        if analysis.triage.request_type == "invalid":
            return True
        if analysis.triage.domain == "None" and top_score < self.MIN_SUPPORT_CONFIDENCE:
            return True
        return False

    @staticmethod
    def _canned_response(analysis: TicketAnalysis) -> str:
        return analysis.final.response

    @staticmethod
    def _risk_level(analysis: TicketAnalysis) -> str:
        if not analysis.guard.is_safe:
            return "critical"
        if analysis.triage.needs_escalation or analysis.final.status == "escalated":
            return "high" if analysis.triage.urgency in {"low", "medium"} else analysis.triage.urgency
        return analysis.triage.urgency

    @staticmethod
    def _references(analysis: TicketAnalysis) -> list[str]:
        return [
            f"[{idx}] {chunk.title} | {chunk.source_path} | fused_score={chunk.score:.4f}"
            for idx, chunk in enumerate(analysis.chunks[:5], start=1)
        ]

    @staticmethod
    def _tokenize_for_stream(text: str) -> list[str]:
        tokens: list[str] = []
        current = ""
        for char in text:
            current += char
            if char in {" ", "\n"}:
                tokens.append(current)
                current = ""
        if current:
            tokens.append(current)
        return tokens

    def _ingest_csv_sync(self, csv_path: Path) -> tuple[Path, int]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        output_rows: list[dict[str, str]] = []
        for row in rows:
            ticket = Ticket(
                issue=(row.get("Issue") or "").strip(),
                subject=(row.get("Subject") or "").strip(),
                company=(row.get("Company") or "None").strip() or "None",
            )
            result = self.pipeline.run(ticket)
            output_rows.append(
                {
                    "status": result.status,
                    "product_area": result.product_area,
                    "response": result.response,
                    "justification": result.justification,
                    "request_type": result.request_type,
                }
            )

        output_path = csv_path.with_name("output.csv" if csv_path.name == "support_tickets.csv" else f"{csv_path.stem}_output.csv")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["status", "product_area", "response", "justification", "request_type"],
            )
            writer.writeheader()
            writer.writerows(output_rows)
        return output_path, len(output_rows)

    def _save_incident_sync(self, ticket: Ticket, analysis: TicketAnalysis) -> Path:
        output_path = self.repo_root / "support_tickets" / "saved_incidents.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type"]
        should_write_header = not output_path.exists() or output_path.stat().st_size == 0

        with output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if should_write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "Issue": ticket.issue,
                    "Subject": ticket.subject,
                    "Company": ticket.company,
                    "Response": analysis.final.response,
                    "Product Area": analysis.final.product_area,
                    "Status": analysis.final.status,
                    "Request Type": analysis.final.request_type,
                }
            )

        self.incident_retriever = SimilarIncidentRetriever(
            repo_root=self.repo_root,
            embedding_model=self.pipeline.retriever.embedding_model,
        )
        return output_path
