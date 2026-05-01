from __future__ import annotations

import re
from collections import Counter

from models import RetrievedChunk, Ticket, TriageResult
from taxonomy import map_product_area


DOMAIN_KEYWORDS = {
    "HackerRank": [
        "hackerrank",
        "assessment",
        "candidate",
        "test",
        "screen",
        "interview",
        "plagiarism",
        "score",
        "recruiter",
        "question bank",
        "employee",
        "hiring account",
        "user management",
    ],
    "Claude": [
        "claude",
        "anthropic",
        "workspace",
        "conversation",
        "chat",
        "team plan",
        "pro plan",
        "api",
        "artifacts",
        "bedrock",
    ],
    "Visa": [
        "visa",
        "card",
        "credit",
        "debit",
        "merchant",
        "chargeback",
        "traveller",
        "traveler's cheque",
        "stolen",
        "fraud",
        "atm",
    ],
}

PRODUCT_AREA_RULES = [
    ("security_compliance", ["infosec", "security questionnaire", "security review", "procurement", "vendor form", "fill in the forms"]),
    ("privacy", ["privacy", "delete conversation", "incognito", "sensitive data", "delete my account"]),
    ("billing", ["refund", "invoice", "billing", "charge", "payment", "subscription"]),
    ("account_access", ["login", "sign in", "workspace", "access", "account", "admin", "seat"]),
    ("travel_support", ["travel", "traveller", "traveler's cheque", "cheque", "abroad"]),
    ("card_support", ["lost card", "stolen card", "card stolen", "replace card", "blocked card"]),
    ("security", ["security", "vulnerability", "bug bounty", "fraud", "identity theft", "phishing", "suspicious email", "api key", "credit card number", "compromised", "breach"]),
    ("screen", ["assessment", "test", "candidate", "screen", "interview", "score"]),
    ("account_access", ["employee has left", "remove them", "remove user", "deactivate user", "user management", "hiring account"]),
    ("conversation_management", ["conversation", "rename", "delete chat", "chat history"]),
]

INVALID_PATTERNS = [
    "iron man",
    "ironman",
    "thank you",
    "thanks",
    "hello",
    "who is the actor",
    "name of the actor",
    "actor ironman",
    "weather",
    "i am a recruiter",
    "i'm a recruiter",
    "i am a student",
    "i am a candidate",
]

SCORE_DISPUTE_PATTERNS = [
    r"\breview\s+my\s+answers\b",
    r"\bin\s*crease\s+my\s+score\b",
    r"\bincrease\s+my\s+score\b",
    r"\bincrease\s+(?:a|the)\s+candidate'?s\s+score\b",
    r"\bmanually\s+increase\s+.*\bscore\b",
    r"\bupdate\s+.*\bscore\s+on\s+(?:my|our)\s+behalf\b",
    r"\bchange\s+.*\bcandidate'?s\s+score\b",
    r"\bautomated\s+grader\s+marked\b",
    r"\bgrader\s+marked\s+.*\bincorrectly\b",
    r"\bchange\s+my\s+score\b",
    r"\badjust\s+my\s+score\b",
    r"\badjust\s+.*\bcandidate'?s\s+score\b",
    r"\boverride\s+my\s+score\b",
    r"\boverride\s+.*\bscore\b",
    r"\bmove\s+me\s+to\s+the\s+next\s+round\b",
    r"\bgraded\s+me\s+unfairly\b",
    r"\bplatform\s+must\s+have\s+graded\s+me\b",
    r"\brecruiter\s+rejected\s+me\b",
    r"\bappeal\s+(?:my\s+)?(?:score|result|grade|assessment|test|plagiarism\s+flag|flag)\b",
    r"\bappeal\s+the\s+(?:score|result|grade|assessment|test|plagiarism\s+flag|flag)\b",
    r"\bappeal\s+.*\b(candidate|test|score|plagiarism|flag)\b",
    r"\b(candidate|test|score|plagiarism|flag)\b.*\bappeal\b",
    r"\bdispute\s+(?:my\s+)?(?:score|result|grade|plagiarism\s+flag|flag)\b",
    r"\bplagiarism\s+.*\b(appeal|dispute|overturn|reverse)\b",
    r"\b(overturn|reverse)\s+.*\b(plagiarism|flag|score|result)\b",
    # Third-person / possessive score manipulation (e.g. "increase the score of my students")
    r"\b(increase|boost|bump|raise|inflate|improve|edit|change|update|fix|adjust|override)\s+the\s+score\s+of\b",
    r"\b(increase|boost|bump|raise|inflate|improve|edit|change|update|fix|adjust|override)\s+(?:my|their|his|her|our)\s+(students?|candidates?|users?|team|employees?)'?s?\s+score\b",
    r"\bgive\s+(?:them|him|her|my\s+(?:students?|candidates?|users?))\s+(?:full\s+)?marks?\b",
    r"\bgive\s+(?:full\s+)?marks?\s+to\b",
    r"\b(set|make)\s+(?:the|their|his|her|my)\s+score\s+(?:higher|to\s+\d+|full|max)\b",
]

ACCOUNT_TAKEOVER_ESCALATION_PATTERNS = [
    r"\bbypass (all )?(verification|auth|authentication|security) (steps|checks|process)?\b",
    r"\btransfer full admin ownership\b",
    r"\btransfer (admin|owner|ownership) (access|rights|role|privileges)?\b",
    r"\breset all admin passwords?\b",
    r"\bcancel all active tests?\b",
    r"\b(account|workspace).{0,40}(compromised|breach|breached)\b",
]

VAGUE_SUPPORT_PATTERNS = [
    r"^(help|help needed|it'?s not working|its not working|not working|broken|please help)[.! ]*$",
    r"\b(it'?s|its|this is|system is|site is|app is)\s+not\s+working\b",
    r"\bhelp\b.{0,20}\b(not working|broken|issue|problem)\b",
]

SECURITY_INCIDENT_ESCALATION_PATTERNS = [
    r"\bphishing\b",
    r"\bsuspicious (email|link|login|message|activity|charge|request)\b",
    r"\b(is this|was this|does this look) (legitimate|real|a scam|fraudulent)\b",
    r"\b(account|workspace|api key|token|password|credit card|payment details?).{0,60}(compromised|breach|breached|leaked|exposed|stolen)\b",
    r"\b(compromised|breached|leaked|exposed|stolen).{0,60}(account|workspace|api key|token|password|credit card|payment details?)\b",
    r"\b(enter|provide|verify|confirm|share).{0,40}(api key|password|credit card|card number|payment details?|mfa code|2fa code|otp)\b",
    r"\b(suspended|locked|disabled).{0,60}(unless|if you do not|within 24 hours|click)\b",
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


class TriageAgent:
    def classify(self, ticket: Ticket) -> TriageResult:
        text = ticket.text.lower()
        domain = self._detect_domain(ticket.company, text)
        request_type = self._detect_request_type(text)
        urgency = self._detect_urgency(text)
        intent_queries = self._split_intent_queries(ticket.issue or ticket.text)
        intents = self._extract_intents(text, intent_queries)
        product_area = self._detect_product_area(text, domain, intents)
        needs_escalation, reason = self._needs_escalation(text, domain)

        return TriageResult(
            domain=domain,
            intents=intents,
            intent_queries=intent_queries,
            urgency=urgency,
            request_type=request_type,
            product_area=product_area,
            needs_escalation=needs_escalation,
            escalation_reason=reason,
        )

    def rewrite_queries(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        attempt_index: int,
    ) -> list[str]:
        rewritten: list[str] = []
        current_queries = triage.intent_queries or [ticket.text]
        hint_terms = self._top_context_terms(chunks)
        domain_hint = triage.domain if triage.domain != "None" else ""
        product_hint = triage.product_area if triage.product_area != "general_support" else ""

        for query in current_queries:
            keywords = self._compress_query(query)
            pieces = [part for part in [domain_hint, product_hint, keywords] if part]
            if attempt_index == 0 and hint_terms:
                pieces.append(hint_terms)
            elif attempt_index >= 1:
                subject_hint = self._compress_query(ticket.subject)
                if subject_hint and subject_hint.lower() not in " ".join(pieces).lower():
                    pieces.append(subject_hint)
            rewritten_query = " ".join(pieces).strip()
            if rewritten_query:
                rewritten.append(rewritten_query)

        if not rewritten:
            fallback = " ".join(
                part for part in [domain_hint, product_hint, self._compress_query(ticket.text)] if part
            ).strip()
            if fallback:
                rewritten.append(fallback)

        deduped: list[str] = []
        seen: set[str] = set()
        for query in rewritten:
            normalized = query.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(query)
        return deduped[:5]

    def _detect_domain(self, company: str, text: str) -> str:
        normalized = (company or "").strip()
        if normalized in DOMAIN_KEYWORDS:
            return normalized
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for keyword in keywords if keyword in text)
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "None"

    @staticmethod
    def _is_gibberish_or_too_short(text: str) -> bool:
        """Return True when the issue text is too short or looks like gibberish.

        Rules:
        - Fewer than 3 whitespace-separated tokens → too short.
        - Of those tokens, if fewer than 40 % contain at least 2 consecutive
          alphabetic characters the text is treated as gibberish (e.g. 'asd qwe'
          or 'xyz 123 !!!').
        """
        MIN_WORDS = 3
        MIN_ALPHA_RATIO = 0.40

        tokens = text.split()
        if len(tokens) < MIN_WORDS:
            return True

        alpha_tokens = sum(1 for t in tokens if re.search(r'[a-zA-Z]{2,}', t))
        if len(tokens) > 0 and (alpha_tokens / len(tokens)) < MIN_ALPHA_RATIO:
            return True

        return False

    @staticmethod
    def _detect_request_type(text: str) -> str:
        if TriageAgent._is_gibberish_or_too_short(text):
            return "invalid"
        if any(pattern in text for pattern in INVALID_PATTERNS):
            return "invalid"
        if any(keyword in text for keyword in ["feature request", "can you add", "please add", "would love to have"]):
            return "feature_request"
        if any(keyword in text for keyword in ["bug", "broken", "not working", "site is down", "error", "issue with page"]):
            return "bug"
        return "product_issue"

    @staticmethod
    def _detect_urgency(text: str) -> str:
        if any(keyword in text for keyword in ["identity theft", "stolen", "fraud", "security vulnerability", "site is down"]):
            return "critical"
        if any(keyword in text for keyword in ["urgent", "immediately", "asap", "cannot access"]):
            return "high"
        if any(keyword in text for keyword in ["whenever", "question", "wondering"]):
            return "low"
        return "medium"

    def _detect_product_area(self, text: str, domain: str, intents: list[str]) -> str:
        taxonomy_area = map_product_area(domain, text)
        if taxonomy_area != "general_support":
            return taxonomy_area
        for area, keywords in PRODUCT_AREA_RULES:
            if any(keyword in text for keyword in keywords):
                return area
        return {
            "HackerRank": "screen",
            "Claude": "conversation_management",
            "Visa": "general_support",
            "None": "general_support",
        }[domain]

    def _extract_intents(self, text: str, intent_queries: list[str]) -> list[str]:
        intents = []
        if "refund" in text or "billing" in text:
            intents.append("billing")
        if "access" in text or "login" in text:
            intents.append("account_access")
        if "delete" in text and "conversation" in text:
            intents.append("privacy_cleanup")
        if "test" in text or "assessment" in text:
            intents.append("assessment_support")
        if self._is_score_dispute(text):
            intents.append("score_dispute")
        if any(keyword in text for keyword in ["infosec", "security questionnaire", "security review", "procurement", "fill in the forms"]):
            intents.append("security_compliance")
        if self._is_security_incident(text):
            intents.append("security")
        if "dispute" in text or "chargeback" in text or "wrong product" in text:
            intents.append("dispute_support")
        if "blocked card" in text or "card blocked" in text or "lost card" in text or "stolen card" in text:
            intents.append("card_support")
        if len(intent_queries) > 1:
            for idx, query in enumerate(intent_queries, start=1):
                label = self._label_intent_query(query)
                if label not in intents:
                    intents.append(label)
        return intents or ["general_support"]

    @staticmethod
    def _split_intent_queries(text: str) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        parts = re.split(r"(?:\?+|;\s+|\n+|(?:\s+and\s+)|(?:\s+also\s+)|(?:\s+plus\s+))", cleaned, flags=re.IGNORECASE)
        queries = []
        for part in parts:
            candidate = part.strip(" .,:;-")
            if len(candidate.split()) >= 3:
                queries.append(candidate)

        if len(queries) <= 1 and "," in cleaned:
            comma_parts = [part.strip(" .,:;-") for part in cleaned.split(",") if len(part.strip().split()) >= 3]
            if len(comma_parts) > 1 and any(
                any(keyword in part.lower() for keyword in ["refund", "billing", "payment", "fix", "restore", "review", "help"])
                for part in comma_parts[1:]
            ):
                queries = comma_parts

        if len(queries) <= 1:
            return [cleaned]

        deduped: list[str] = []
        for query in queries:
            lowered = query.lower()
            if lowered not in {existing.lower() for existing in deduped}:
                deduped.append(query)
        return deduped[:5]

    @staticmethod
    def _label_intent_query(query: str) -> str:
        text = query.lower()
        if any(keyword in text for keyword in ["refund", "billing", "payment", "invoice", "subscription"]):
            return "billing"
        if any(keyword in text for keyword in ["login", "access", "admin", "workspace", "seat", "account"]):
            return "account_access"
        if any(keyword in text for keyword in ["privacy", "delete", "export", "data", "conversation"]):
            return "privacy_cleanup"
        if any(keyword in text for keyword in ["dispute", "merchant", "wrong product", "chargeback", "seller"]):
            return "dispute_support"
        if any(keyword in text for keyword in ["fraud", "identity theft", "security", "stolen", "phishing", "suspicious email", "api key", "credit card"]):
            return "security"
        if any(keyword in text for keyword in ["test", "assessment", "candidate", "score", "reinvite"]):
            if TriageAgent._is_score_dispute(text):
                return "score_dispute"
            return "assessment_support"
        if any(keyword in text for keyword in ["infosec", "security questionnaire", "security review", "procurement", "forms"]):
            return "security_compliance"
        return "general_support"

    @staticmethod
    def _compress_query(text: str) -> str:
        terms = re.findall(r"[a-zA-Z0-9']+", text.lower())
        filtered = [term for term in terms if len(term) > 2 and term not in {"please", "today", "help", "issue", "problem"}]
        return " ".join(filtered[:10])

    @staticmethod
    def _top_context_terms(chunks: list[RetrievedChunk]) -> str:
        stop_words = {
            "the", "and", "for", "that", "with", "from", "this", "your", "have", "into", "when",
            "then", "will", "must", "their", "about", "using", "only", "after", "more", "than",
            "what", "where", "which", "please", "contact", "click", "https", "http",
        }
        counter: Counter[str] = Counter()
        for chunk in chunks[:3]:
            for term in re.findall(r"[a-zA-Z0-9']+", chunk.text.lower()):
                if len(term) <= 3 or term in stop_words:
                    continue
                counter[term] += 1
        return " ".join(term for term, _ in counter.most_common(4))

    @staticmethod
    def _needs_escalation(text: str, domain: str) -> tuple[bool, str | None]:
        if domain == "HackerRank" and TriageAgent._is_score_dispute(text):
            return True, "Candidate score disputes, answer reviews, plagiarism flag appeals, and next-round outcome changes require human review."
        if domain == "HackerRank" and TriageAgent._is_account_takeover_request(text):
            return True, "Privileged account changes, admin ownership transfers, and verification bypass requests require human review."
        if TriageAgent._is_security_incident(text):
            return True, "Potential phishing, credential, account, payment, or security incident requires human review."
        if domain == "HackerRank" and any(
            keyword in text
            for keyword in ["infosec", "security questionnaire", "security review", "procurement", "fill in the forms"]
        ):
            return True, "Infosec or vendor security form handling is not covered by the provided support corpus."
        if any(keyword in text for keyword in ["identity theft", "fraud", "stolen", "security vulnerability", "bug bounty"]):
            return True, "Sensitive security or fraud scenario."
        if "not the workspace owner" in text or "not admin" in text:
            return True, "Account restoration requires administrator verification."
        if domain == "None" and (len(text.split()) < 8 or TriageAgent._is_vague_support_request(text)):
            return True, "Insufficient information to route safely."
        return False, None

    @staticmethod
    def _is_score_dispute(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return any(re.search(pattern, normalized) for pattern in SCORE_DISPUTE_PATTERNS)

    @staticmethod
    def _is_account_takeover_request(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return any(re.search(pattern, normalized) for pattern in ACCOUNT_TAKEOVER_ESCALATION_PATTERNS)

    @staticmethod
    def _is_security_incident(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return any(re.search(pattern, normalized) for pattern in SECURITY_INCIDENT_ESCALATION_PATTERNS)

    @staticmethod
    def _is_vague_support_request(text: str) -> bool:
        normalized = " ".join(text.lower().split())
        return any(re.search(pattern, normalized) for pattern in VAGUE_SUPPORT_PATTERNS)
