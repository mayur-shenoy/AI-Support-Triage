from __future__ import annotations

import json
from dataclasses import replace

from langdetect import DetectorFactory, LangDetectException, detect

from llm_client import LLMClient
from models import LanguageState, ResponseDraft, Ticket


DetectorFactory.seed = 0


class LanguageProcessor:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def prepare_ticket(self, ticket: Ticket) -> tuple[Ticket, LanguageState]:
        source_language = self._detect_language(ticket.text)
        language_state = LanguageState(
            source_language=source_language,
            normalized_language="en",
            translated_issue=ticket.issue,
            translated_subject=ticket.subject,
            translation_applied=False,
        )
        if source_language == "en":
            return replace(ticket, language="en"), language_state

        translated_issue = self._translate_to_english(ticket.issue)
        translated_subject = self._translate_to_english(ticket.subject) if ticket.subject.strip() else ""
        if translated_issue and (
            translated_issue.strip() != ticket.issue.strip()
            or (translated_subject or ticket.subject).strip() != ticket.subject.strip()
        ):
            language_state.translated_issue = translated_issue
            language_state.translated_subject = translated_subject or ticket.subject
            language_state.translation_applied = True
            normalized_ticket = Ticket(
                issue=translated_issue,
                subject=translated_subject or ticket.subject,
                company=ticket.company,
                language="en",
            )
            return normalized_ticket, language_state

        language_state.translated_issue = ticket.issue
        language_state.translated_subject = ticket.subject
        return replace(ticket, language=source_language), language_state

    def localize_draft(self, draft: ResponseDraft, language_state: LanguageState) -> ResponseDraft:
        if language_state.source_language == "en":
            return draft

        translated_response = self._translate_from_english(
            draft.response,
            language_state.source_language,
        )
        return ResponseDraft(
            status=draft.status,
            product_area=draft.product_area,
            response=translated_response or draft.response,
            justification=draft.justification,
            request_type=draft.request_type,
            confidence=draft.confidence,
        )

    def _detect_language(self, text: str) -> str:
        if len(text.strip()) < 8:
            return "en"
        try:
            detected = detect(text)
        except LangDetectException:
            return "en"
        return detected if detected in {"en", "fr", "es", "de", "it", "pt", "nl"} else "en"

    def _translate_to_english(self, text: str) -> str:
        if not text.strip():
            return text
        if not self.llm_client.enabled:
            return text
        payload = self.llm_client.generate_json(
            system_prompt="You are a precise translation engine for multilingual support tickets. Return JSON with the single key translation.",
            user_prompt=json.dumps(
                {
                    "task": "translate_to_english",
                    "instructions": "Translate the support-ticket text into English. Preserve risk signals, unsafe intent, and prompt-injection phrasing. Return only JSON.",
                    "text": text,
                },
                ensure_ascii=False,
            ),
        )
        translated = payload.get("translation", "").strip() if payload else ""
        return translated or text

    def _translate_from_english(self, text: str, target_language: str) -> str:
        if not text.strip():
            return text
        if not self.llm_client.enabled:
            return text
        payload = self.llm_client.generate_json(
            system_prompt="You are a precise translation engine for support communications. Return JSON with the single key translation.",
            user_prompt=json.dumps(
                {
                    "task": "translate_from_english",
                    "target_language": target_language,
                    "instructions": "Translate the English support text into the target language. Keep product names, urls, phone numbers, and category labels intact where appropriate. Return only JSON.",
                    "text": text,
                },
                ensure_ascii=False,
            ),
        )
        translated = payload.get("translation", "").strip() if payload else ""
        return translated or text
