from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


SUPPORT_CONCEPTS = {
    "refund": ["refund", "reimburse", "money back", "not satisfied"],
    "billing": ["billing", "invoice", "payment", "purchase", "subscription", "pay"],
    "mock_interview": ["mock interview", "interview credits", "ai-powered mock"],
    "candidate_assessment": ["assessment", "candidate", "test", "score", "reinvite"],
    "account_access": ["access", "login", "admin", "owner", "workspace", "seat"],
    "user_management": [
        "user management",
        "manage users",
        "team member",
        "employee",
        "deactivate user",
        "deactivating a user",
        "reactivate user",
        "remove users",
        "left the company",
    ],
    "security_review": ["infosec", "security review", "security questionnaire", "vendor", "procurement"],
    "privacy": ["privacy", "delete", "export", "data retention", "sensitive data"],
    "fraud": ["fraud", "identity theft", "stolen", "unauthorized", "scam"],
    "card_support": ["blocked card", "lost card", "stolen card", "emergency cash"],
    "merchant_dispute": ["merchant", "seller", "dispute", "chargeback", "wrong product"],
    "contact": ["contact", "email", "phone", "support team"],
    "policy": ["policy", "rules", "must", "cannot", "not available", "not supported"],
}


@dataclass(slots=True)
class GraphExpansion:
    section_id: str
    concepts: list[str]
    path: str
    score: float


class SupportGraphStore:
    def __init__(self, records: list[Any], section_members: dict[str, list[Any]]) -> None:
        self.records = records
        self.section_members = section_members
        self.section_concepts = self._build_section_concepts()
        self.concept_sections = self._build_concept_sections()

    def concepts_for_text(self, text: str) -> list[str]:
        normalized = text.lower()
        concepts = []
        for concept, aliases in SUPPORT_CONCEPTS.items():
            if any(alias in normalized for alias in aliases):
                concepts.append(concept)
        return concepts

    def concepts_for_section(self, section_id: str) -> list[str]:
        return sorted(self.section_concepts.get(section_id, set()))

    def related_section_ids(
        self,
        primary: Any,
        query_terms: set[str],
        target_terms: set[str],
        limit: int = 4,
    ) -> list[GraphExpansion]:
        query_concepts = set(self._concepts_from_terms(query_terms))
        primary_concepts = self.section_concepts.get(primary.section_id, set())
        desired_concepts = query_concepts | primary_concepts
        if not desired_concepts:
            return []

        candidates: dict[str, GraphExpansion] = {}
        for concept in desired_concepts:
            for section_id in self.concept_sections.get(concept, set()):
                if section_id == primary.section_id:
                    continue
                section_records = self.section_members.get(section_id, [])
                if not section_records:
                    continue
                candidate = section_records[0]
                if candidate.domain != primary.domain:
                    continue
                candidate_terms = self._section_terms(section_records)
                target_overlap = len(candidate_terms & target_terms)
                concept_overlap = len(self.section_concepts.get(section_id, set()) & desired_concepts)
                if target_overlap == 0 and concept_overlap < 2:
                    continue
                score = (0.006 * target_overlap) + (0.004 * concept_overlap)
                path = f"{primary.section_id} -> {concept} -> {section_id}"
                existing = candidates.get(section_id)
                if not existing or score > existing.score:
                    candidates[section_id] = GraphExpansion(
                        section_id=section_id,
                        concepts=sorted(self.section_concepts.get(section_id, set()) & desired_concepts),
                        path=path,
                        score=score,
                    )

        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:limit]

    def _build_section_concepts(self) -> dict[str, set[str]]:
        section_concepts: dict[str, set[str]] = {}
        for section_id, records in self.section_members.items():
            text = " ".join(
                " ".join([record.title, record.section_title, record.source_path, record.text])
                for record in records
            )
            concepts = set(self.concepts_for_text(text))
            section_concepts[section_id] = concepts
        return section_concepts

    def _build_concept_sections(self) -> dict[str, set[str]]:
        concept_sections: dict[str, set[str]] = defaultdict(set)
        for section_id, concepts in self.section_concepts.items():
            for concept in concepts:
                concept_sections[concept].add(section_id)
        return dict(concept_sections)

    def _concepts_from_terms(self, terms: set[str]) -> list[str]:
        text = " ".join(sorted(terms))
        return self.concepts_for_text(text)

    @staticmethod
    def _section_terms(records: list[Any]) -> set[str]:
        text = " ".join(record.text for record in records)
        return {term for term in re.findall(r"[a-zA-Z0-9']+", text.lower()) if len(term) > 2}
