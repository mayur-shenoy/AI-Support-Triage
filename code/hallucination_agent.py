from __future__ import annotations

import re

from models import HallucinationResult, ResponseDraft, RetrievedChunk, Ticket, TriageResult


class HallucinationAgent:
    MAX_GROUNDED_SCORE = 0.58
    CONTENT_STOP_WORDS = {
        "the", "and", "for", "that", "this", "with", "from", "your", "you", "are", "can", "will",
        "should", "would", "could", "into", "when", "what", "where", "which", "please", "support",
        "documentation", "retrieved", "provided", "case", "issue", "request", "team", "review",
    }
    BOILERPLATE_PATTERNS = (
        "i could not find",
        "this needs human review",
        "i cannot",
        "i am escalating",
        "contact the support team directly",
        "based on the support documentation",
        "based on the retrieved support guidance",
        "if this does not resolve",
        "if any intent remains unsupported",
    )
    CONTACT_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|(?:https?://|mailto:)[^\s)]+", re.IGNORECASE)

    def evaluate(
        self,
        ticket: Ticket,
        triage: TriageResult,
        draft: ResponseDraft,
        chunks: list[RetrievedChunk],
    ) -> HallucinationResult:
        if draft.status == "escalated" or triage.needs_escalation:
            return HallucinationResult(
                is_grounded=True,
                score=0.0,
                notes="Skipped hallucination scoring because the case is already escalated.",
            )
        if not chunks:
            return HallucinationResult(
                is_grounded=False,
                score=1.0,
                unsupported_claims=[draft.response[:180]],
                notes="No retrieved context was available for a replied response.",
            )

        context = "\n".join(chunk.text for chunk in chunks).lower()
        response_sentences = self._response_sentences(draft.response)
        unsupported: list[str] = []
        scored = 0
        unsupported_weight = 0.0

        for sentence in response_sentences:
            lowered = sentence.lower()
            if self._is_boilerplate(lowered):
                continue
            scored += 1
            sentence_terms = self._content_terms(sentence)
            if not sentence_terms:
                continue
            covered_terms = {term for term in sentence_terms if term in context}
            coverage = len(covered_terms) / len(sentence_terms)
            unsupported_contacts = [
                contact for contact in self.CONTACT_RE.findall(sentence) if contact.lower() not in context
            ]
            if coverage < 0.42 or unsupported_contacts:
                unsupported.append(sentence)
                unsupported_weight += 1.0 - coverage
                if unsupported_contacts:
                    unsupported_weight += 0.35

        if scored == 0:
            return HallucinationResult(
                is_grounded=True,
                score=0.0,
                notes="Response only contained escalation or uncertainty boilerplate.",
            )

        score = min(1.0, unsupported_weight / scored)
        return HallucinationResult(
            is_grounded=score <= self.MAX_GROUNDED_SCORE,
            score=score,
            unsupported_claims=unsupported[:4],
            notes=f"Checked {scored} response claim(s) against {len(chunks)} retrieved evidence bundle(s).",
        )

    def apply(
        self,
        ticket: Ticket,
        triage: TriageResult,
        draft: ResponseDraft,
        chunks: list[RetrievedChunk],
    ) -> tuple[ResponseDraft, HallucinationResult]:
        result = self.evaluate(ticket, triage, draft, chunks)
        if result.is_grounded:
            return draft, result

        draft.status = "escalated"
        draft.response = (
            "I could not verify enough of this response against the retrieved support documentation, "
            "so I am escalating it for human review rather than risking an unsupported answer."
        )
        draft.justification = (
            f"{draft.justification} Escalated by hallucination verifier because unsupported claim score "
            f"was {result.score:.2f}."
        ).strip()
        draft.confidence = min(draft.confidence, 0.35)
        return draft, result

    def _response_sentences(self, response: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", response).strip()
        return [
            sentence.strip(" -")
            for sentence in re.split(r"(?<=[.!?])\s+|(?:\s+Intent\s+\d+:)", cleaned)
            if len(sentence.strip()) >= 28
        ]

    def _content_terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9']+", text.lower())
            if len(term) > 3 and term not in self.CONTENT_STOP_WORDS
        }

    def _is_boilerplate(self, lowered_sentence: str) -> bool:
        return any(pattern in lowered_sentence for pattern in self.BOILERPLATE_PATTERNS)
