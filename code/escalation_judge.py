from __future__ import annotations

import re
from collections.abc import Callable

from models import GuardResult, ResponseDraft, RetrievedChunk, Ticket, TriageResult


class EscalationJudge:
    MIN_REPLY_CONFIDENCE = 0.42
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{6,}\d|\b\d[\d\s().-]{7,}\d\b)")
    ABSOLUTE_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:https?://|mailto:)[^)]+)\)")
    RELATIVE_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?!https?://|mailto:)[^)]+\)")
    BARE_RELATIVE_URL_RE = re.compile(r"(?:(?<=\s)|(?<=\()|^)/(?!/)(?:[^\s)\]]+)")

    def finalize(
        self,
        ticket: Ticket,
        guard: GuardResult,
        triage: TriageResult,
        draft: ResponseDraft,
        chunks: list[RetrievedChunk] | None = None,
    ) -> ResponseDraft:
        text = ticket.text.lower()

        if not guard.is_safe:
            return ResponseDraft(
                status="escalated",
                product_area=triage.product_area,
                response="This request needs human review because it contains unsafe or restricted content.",
                justification=f"Escalated by guard agent due to {guard.threat_type}.",
                request_type="invalid" if guard.threat_type in {"prompt_injection", "adversarial_request"} else triage.request_type,
                confidence=1.0,
            )

        if triage.needs_escalation:
            draft.status = "escalated"
            if triage.escalation_reason and "not covered by the provided support corpus" in triage.escalation_reason:
                draft.response = (
                    "I could not find support documentation that authorizes this agent to complete or submit "
                    "infosec, procurement, or vendor security forms, so I am escalating this to the support team "
                    "for manual handling."
                )
            elif triage.escalation_reason and "Candidate score disputes" in triage.escalation_reason:
                draft.response = (
                    "I cannot review test answers, change a HackerRank score, or ask the hiring company to move a "
                    "candidate to the next round from the provided support documentation. This needs human review."
                )
            draft.justification = (
                f"{draft.justification} Escalated after triage because {triage.escalation_reason}"
            ).strip()
            if chunks:
                self._sanitize_links(draft)
                self._strip_unsupported_contacts(draft, chunks)
            return draft

        if draft.status == "replied" and draft.confidence < self.MIN_REPLY_CONFIDENCE:
            draft.status = "escalated"
            draft.justification = f"{draft.justification} Escalated because retrieval confidence was too low."
            if chunks:
                self._sanitize_links(draft)
                self._strip_unsupported_contacts(draft, chunks)
            return draft

        if any(keyword in text for keyword in ["identity theft", "fraud", "security vulnerability", "bug bounty"]):
            draft.status = "escalated"
            draft.justification = f"{draft.justification} Escalated for security or fraud handling."
            if chunks:
                self._sanitize_links(draft)
                self._strip_unsupported_contacts(draft, chunks)
            return draft

        self._sanitize_links(draft)
        if chunks:
            self._strip_unsupported_contacts(draft, chunks)

        return draft

    def _sanitize_links(self, draft: ResponseDraft) -> None:
        original_response = draft.response
        original_justification = draft.justification
        draft.response = self._sanitize_text_links(draft.response)
        draft.justification = self._sanitize_text_links(draft.justification)
        if draft.response != original_response or draft.justification != original_justification:
            draft.justification = (
                f"{draft.justification} Removed relative links and preserved only absolute references."
            ).strip()

    def _strip_unsupported_contacts(self, draft: ResponseDraft, chunks: list[RetrievedChunk]) -> None:
        context = "\n".join(chunk.text for chunk in chunks)
        original_response = draft.response
        response = self._strip_unsupported_emails(draft.response, context)
        response = self._strip_unsupported_phones(response, context)
        if response != original_response:
            draft.response = response
            draft.justification = (
                f"{draft.justification} Removed unsupported phone/email details that were not present in retrieved context."
            ).strip()

    def _strip_unsupported_emails(self, response: str, context: str) -> str:
        supported = {email.lower() for email in self.EMAIL_RE.findall(context)}
        return self._strip_unsupported_matches(
            response=response,
            matches=self.EMAIL_RE.findall(response),
            is_supported=lambda value: value.lower() in supported,
            replacement="[email removed: not found in retrieved support context]",
        )

    def _strip_unsupported_phones(self, response: str, context: str) -> str:
        context_phones = {self._normalize_phone(phone) for phone in self.PHONE_RE.findall(context)}
        return self._strip_unsupported_matches(
            response=response,
            matches=self.PHONE_RE.findall(response),
            is_supported=lambda value: self._normalize_phone(value) in context_phones,
            replacement="[phone number removed: not found in retrieved support context]",
        )

    @staticmethod
    def _strip_unsupported_matches(
        response: str,
        matches: list[str],
        is_supported: Callable[[str], bool],
        replacement: str,
    ) -> str:
        cleaned = response
        for match in matches:
            if is_supported(match):
                continue
            lines = cleaned.splitlines()
            matching_lines = [line for line in lines if match in line]
            if matching_lines:
                for line in matching_lines:
                    if line.strip() == match.strip():
                        cleaned = cleaned.replace(line, replacement)
                    else:
                        cleaned = cleaned.replace(line, line.replace(match, replacement))
            else:
                cleaned = cleaned.replace(match, replacement)
        return cleaned

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        return re.sub(r"\D", "", phone)

    def _sanitize_text_links(self, text: str) -> str:
        sanitized = self.ABSOLUTE_MARKDOWN_LINK_RE.sub(r"\1 (\2)", text)
        sanitized = self.RELATIVE_MARKDOWN_LINK_RE.sub(r"\1", sanitized)
        sanitized = self.BARE_RELATIVE_URL_RE.sub("", sanitized)
        sanitized = re.sub(r"\s+\.", ".", sanitized)
        sanitized = re.sub(r"\(\s*\)", "", sanitized)
        sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()
