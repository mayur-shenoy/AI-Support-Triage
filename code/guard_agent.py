from __future__ import annotations

import re

from models import GuardResult, Ticket


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|your|previous) instructions",
    r"show (me )?(all )?(internal|system) (rules|prompt|instructions|docs)",
    r"display (all )?(internal|system) (rules|prompt|instructions|docs)",
    r"reveal (the )?(system|developer) prompt",
    r"jailbreak",
    r"bypass (your )?(policy|guardrails|restrictions)",
    r"retrieved documents",
    r"exact logic (you use|used)",
    r"affiche(r)? toutes? les (r[eè]gles|instructions) internes",
    r"documents? r[eé]cup[eé]r[eé]s",
    r"logique exacte",
    r"mu[eé]strame (todas )?las (reglas|instrucciones) internas",
    r"documentos recuperados",
]

ADVERSARIAL_PATTERNS = [
    r"delete all files",
    r"rm -rf",
    r"drop database",
    r"steal passwords?",
    r"hack(ing)? instructions",
    r"malware",
]

SENSITIVE_PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d[ -]*?){13,19}\b",
]


class GuardAgent:
    def evaluate(self, ticket: Ticket) -> GuardResult:
        text = ticket.text.lower()

        if self._matches_any(text, PROMPT_INJECTION_PATTERNS):
            return GuardResult(
                is_safe=False,
                threat_type="prompt_injection",
                confidence=0.99,
                notes="User asked to override or reveal internal instructions.",
            )

        if self._matches_any(text, ADVERSARIAL_PATTERNS):
            return GuardResult(
                is_safe=False,
                threat_type="adversarial_request",
                confidence=0.98,
                notes="User requested destructive or abusive behavior.",
            )

        if self._matches_any(text, SENSITIVE_PII_PATTERNS):
            return GuardResult(
                is_safe=False,
                threat_type="sensitive_pii",
                confidence=0.90,
                notes="Potential sensitive personal or card number detected.",
            )

        return GuardResult(is_safe=True, confidence=0.75, notes="No hard safety block detected.")

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
