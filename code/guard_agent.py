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

ACCOUNT_TAKEOVER_PATTERNS = [
    r"\bbypass (all )?(verification|auth|authentication|security) (steps|checks|process)?\b",
    r"\btransfer full admin ownership\b",
    r"\btransfer (admin|owner|ownership) (access|rights|role|privileges)?\b",
    r"\breset all admin passwords?\b",
    r"\bcancel all active tests?\b",
    r"\b(account|workspace).{0,40}(compromised|breach|breached)\b",
    r"\b(ceo|executive).{0,80}\b(bypass|override|transfer|reset all)\b",
]

SECURITY_INCIDENT_PATTERNS = [
    r"\bphishing\b",
    r"\bsuspicious (email|link|login|message|activity|charge|request)\b",
    r"\b(is this|was this|does this look) (legitimate|real|a scam|fraudulent)\b",
    r"\b(account|workspace|api key|token|password|credit card|payment details?).{0,60}(compromised|breach|breached|leaked|exposed|stolen)\b",
    r"\b(compromised|breached|leaked|exposed|stolen).{0,60}(account|workspace|api key|token|password|credit card|payment details?)\b",
    r"\b(enter|provide|verify|confirm|share).{0,40}(api key|password|credit card|card number|payment details?|mfa code|2fa code|otp)\b",
    r"\b(suspended|locked|disabled).{0,60}(unless|if you do not|within 24 hours|click)\b",
    r"\b(click|open|follow).{0,40}(suspicious|unknown|untrusted|email|link)\b",
    r"\bunauthorized (login|access|charge|transaction|change|admin|user)\b",
    r"\bunknown (login|device|session|user|admin)\b",
    r"\baccount takeover\b",
    r"\bcredential (theft|stuffing|leak|dump|exposure)\b",
    r"\b(api key|token|secret|password).{0,40}(rotate|revoke|leaked|exposed|stolen)\b",
    r"\bmalware|ransomware|virus|trojan|keylogger|spyware\b",
    r"\bspoof(ed|ing)?|impersonat(e|ion|ing)|scam|fraudulent\b",
    r"\bdata (breach|leak|exfiltration|exposure)\b",
    r"\bsecurity (breach|incident|vulnerability|exploit)\b",
    r"\bbug bounty|xss|cross-site scripting|sql injection|sqli|csrf|ssrf|rce|remote code execution\b",
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

        if self._matches_any(text, ACCOUNT_TAKEOVER_PATTERNS):
            return GuardResult(
                is_safe=False,
                threat_type="account_takeover_attempt",
                confidence=0.97,
                notes="User requested privileged account changes, ownership transfer, or verification bypass.",
            )

        if self._matches_any(text, SECURITY_INCIDENT_PATTERNS):
            return GuardResult(
                is_safe=False,
                threat_type="security_incident",
                confidence=0.95,
                notes="User described a potential phishing, credential, account, payment, or security incident.",
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
