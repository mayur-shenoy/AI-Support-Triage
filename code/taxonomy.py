from __future__ import annotations


DOMAIN_PRODUCT_TAXONOMY = {
    "HackerRank": {
        "screen": [
            "assessment",
            "test",
            "candidate",
            "reinvite",
            "extra time",
            "score",
            "proctor",
            "interview",
            "question",
            "variant",
            "diagram",
            "draw.io",
        ],
        "library": ["library", "question bank", "create question", "diagram question", "coding question"],
        "community": ["community", "challenge", "certification", "skill certificate", "profile", "resume"],
        "billing": ["subscription", "billing", "payment", "refund", "invoice", "mock interview"],
        "account_access": [
            "login",
            "password",
            "account",
            "delete my account",
            "sso",
            "access",
            "employee has left",
            "remove them",
            "remove user",
            "deactivate user",
            "user management",
            "hiring account",
        ],
        "integrations": ["greenhouse", "lever", "ashby", "workday", "ats", "integration"],
        "security_compliance": ["infosec", "security review", "security questionnaire", "procurement", "vendor", "forms"],
    },
    "Claude": {
        "account_access": ["login", "workspace", "seat", "admin", "owner", "access", "account"],
        "conversation_management": ["conversation", "chat", "rename", "delete", "share", "history", "incognito"],
        "billing": ["billing", "subscription", "invoice", "payment", "refund", "plan", "pro", "team"],
        "privacy": ["privacy", "data", "sensitive", "export", "delete account", "retention"],
        "api_console": ["api", "console", "rate limit", "key", "usage", "bedrock"],
        "security": ["security", "vulnerability", "bug bounty", "phishing", "compromised"],
    },
    "Visa": {
        "card_support": ["lost card", "stolen card", "blocked card", "card blocked", "replacement card"],
        "fraud_security": ["fraud", "identity theft", "scam", "phishing", "unauthorized", "security"],
        "disputes": ["dispute", "chargeback", "wrong product", "merchant", "seller", "refund"],
        "travel_support": ["travel", "traveller", "traveler", "abroad", "emergency cash", "cheque"],
        "merchant_rules": ["minimum", "maximum", "merchant", "surcharge", "rules", "us virgin islands"],
        "general_support": ["visa", "card", "atm", "contact", "phone"],
    },
    "None": {
        "general_support": [],
        "out_of_scope": ["actor", "weather", "thank you", "hello"],
    },
}


AREA_ALIASES = {
    "security": "fraud_security",
    "travel_support": "travel_support",
    "card_support": "card_support",
    "privacy": "privacy",
    "billing": "billing",
    "account_access": "account_access",
    "conversation_management": "conversation_management",
    "screen": "screen",
}


def map_product_area(domain: str, text: str, fallback: str = "general_support") -> str:
    normalized_text = text.lower()
    taxonomy = DOMAIN_PRODUCT_TAXONOMY.get(domain, DOMAIN_PRODUCT_TAXONOMY["None"])
    scores: dict[str, int] = {}
    for area, keywords in taxonomy.items():
        scores[area] = sum(1 for keyword in keywords if keyword in normalized_text)
    if scores:
        best_area = max(scores, key=scores.get)
        if scores[best_area] > 0:
            return best_area
    if fallback in taxonomy:
        return fallback
    return AREA_ALIASES.get(fallback, "general_support")


def normalize_product_area(domain: str, area: str, text: str) -> str:
    taxonomy = DOMAIN_PRODUCT_TAXONOMY.get(domain, DOMAIN_PRODUCT_TAXONOMY["None"])
    if area in taxonomy:
        return area
    alias = AREA_ALIASES.get(area)
    if alias and alias in taxonomy:
        return alias
    return map_product_area(domain, text)
