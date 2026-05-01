from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import List, Optional


ALLOWED_STATUSES = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
ALLOWED_URGENCY = {"low", "medium", "high", "critical"}
ALLOWED_DOMAINS = {"HackerRank", "Claude", "Visa", "None"}


@dataclass(slots=True)
class Ticket:
    issue: str
    subject: str
    company: str
    language: str = "en"

    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.subject.strip(), self.issue.strip()) if part)


@dataclass(slots=True)
class GuardResult:
    is_safe: bool
    threat_type: Optional[str] = None
    confidence: float = 0.0
    notes: str = ""


@dataclass(slots=True)
class TriageResult:
    domain: str
    intents: List[str] = field(default_factory=list)
    intent_queries: List[str] = field(default_factory=list)
    query_rewrite_attempts: int = 0
    urgency: str = "medium"
    request_type: str = "product_issue"
    product_area: str = "general_support"
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    domain: str
    source_path: str
    title: str
    text: str
    score: float
    document_id: str = ""
    section_id: str = ""
    section_title: str = ""
    source_url: str = ""
    supporting_source_paths: List[str] = field(default_factory=list)
    graph_concepts: List[str] = field(default_factory=list)
    graph_expansion_paths: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ResponseDraft:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    confidence: float = 0.0


@dataclass(slots=True)
class HallucinationResult:
    is_grounded: bool
    score: float
    unsupported_claims: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class PipelineResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    confidence: float = 0.0


@dataclass(slots=True)
class TicketTrace:
    ticket_index: int
    input_ticket: dict[str, str]
    normalized_ticket: dict[str, str]
    language: dict[str, Any]
    pipeline_passed: bool
    guard: dict[str, Any]
    triage: dict[str, Any]
    retrieval: dict[str, Any]
    draft: dict[str, Any]
    hallucination: dict[str, Any]
    final: dict[str, Any]


@dataclass(slots=True)
class IncidentMatch:
    issue: str
    subject: str
    company: str
    response: str
    product_area: str
    status: str
    request_type: str
    score: float


@dataclass(slots=True)
class LanguageState:
    source_language: str = "en"
    normalized_language: str = "en"
    translated_issue: str = ""
    translated_subject: str = ""
    translation_applied: bool = False
