from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from models import IncidentMatch, Ticket


class SimilarIncidentRetriever:
    def __init__(self, repo_root: Path, embedding_model: object) -> None:
        self.repo_root = repo_root
        self.embedding_model = embedding_model
        self.incidents = self._load_incidents()
        self._texts = [self._incident_text(incident) for incident in self.incidents]
        self._embeddings = self.embedding_model.encode(self._texts, normalize_embeddings=True)

    def retrieve(self, ticket: Ticket, limit: int = 5) -> list[IncidentMatch]:
        if not self.incidents:
            return []

        query = ticket.text or ticket.issue
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self._embeddings, query_embedding)

        ranked: list[IncidentMatch] = []
        for index in np.argsort(scores)[::-1]:
            incident = self.incidents[int(index)]
            score = float(scores[int(index)])
            if ticket.company != "None" and incident.company == ticket.company:
                score += 0.03
            ranked.append(
                IncidentMatch(
                    issue=incident.issue,
                    subject=incident.subject,
                    company=incident.company,
                    response=incident.response,
                    product_area=incident.product_area,
                    status=incident.status,
                    request_type=incident.request_type,
                    score=score,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _load_incidents(self) -> list[IncidentMatch]:
        sample_path = self.repo_root / "support_tickets" / "sample_support_tickets.csv"
        with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [
            IncidentMatch(
                issue=(row.get("Issue") or "").strip(),
                subject=(row.get("Subject") or "").strip(),
                company=(row.get("Company") or "None").strip() or "None",
                response=(row.get("Response") or "").strip(),
                product_area=(row.get("Product Area") or "").strip(),
                status=(row.get("Status") or "").strip(),
                request_type=(row.get("Request Type") or "").strip(),
                score=0.0,
            )
            for row in rows
        ]

    @staticmethod
    def _incident_text(incident: IncidentMatch) -> str:
        return "\n".join(
            part
            for part in (
                incident.company,
                incident.subject,
                incident.issue,
                incident.product_area,
                incident.response,
            )
            if part
        )
