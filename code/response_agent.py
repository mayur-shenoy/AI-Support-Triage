from __future__ import annotations

import json
import re
from dataclasses import asdict

from llm_client import LLMClient
from models import ResponseDraft, RetrievedChunk, Ticket, TriageResult
from taxonomy import normalize_product_area


class ResponseAgent:
    LOW_CONFIDENCE_TEMPLATE_THRESHOLD = 0.58
    EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    RESPONSE_RE = re.compile(r"RESPONSE:\s*(.*?)(?=\n[A-Z_ ]+:\s|\Z)", re.DOTALL)
    JUSTIFICATION_RE = re.compile(r"JUSTIFICATION:\s*(.*?)(?=\n[A-Z_ ]+:\s|\Z)", re.DOTALL)
    STATUS_RE = re.compile(r"STATUS:\s*(.+)", re.IGNORECASE)
    PRODUCT_AREA_RE = re.compile(r"PRODUCT_AREA:\s*(.+)", re.IGNORECASE)
    REQUEST_TYPE_RE = re.compile(r"REQUEST_TYPE:\s*(.+)", re.IGNORECASE)
    CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE)
    ACTION_START_RE = re.compile(
        r"^(?:contact|ask|go to|navigate|sign in|click|use|report|review|ensure|enable|pause|resume|submit|"
        r"follow|reach out|visit|complete|provide|select|enter|check|open|remove|create|request|update)\b",
        re.IGNORECASE,
    )
    LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)|https?://\S+|mailto:\S+", re.IGNORECASE)

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def draft(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> ResponseDraft:
        if self.llm_client.enabled and chunks:
            llm_result = self._draft_with_llm(ticket, triage, chunks, intent_chunks=intent_chunks)
            if llm_result:
                return llm_result
        return self._draft_with_templates(ticket, triage, chunks, intent_chunks=intent_chunks)

    def _draft_with_llm(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> ResponseDraft | None:
        fallback_confidence = self._estimate_confidence(triage, chunks)
        system_prompt = (
            "You are a support triage assistant. Reply only from the provided corpus context. "
            "If the answer is missing or unsafe, recommend escalation. If there are multiple intents, "
            "answer each intent in a short labeled section using only relevant context. Do not paste raw document "
            "chunks, markdown tables, images, or relative links. Synthesize a concise support reply using only the "
            "facts and steps present in context. Return strict JSON with keys status, product_area, response, "
            "justification, request_type, confidence."
        )
        user_prompt = self._build_llm_user_prompt(ticket, triage, chunks, intent_chunks=intent_chunks)
        payload = self.llm_client.generate_json(system_prompt, user_prompt)
        if payload:
            return self._draft_from_payload(ticket, triage, payload, fallback_confidence)
        return self._draft_with_llm_text(ticket, triage, chunks, fallback_confidence, intent_chunks=intent_chunks)

    def _draft_with_templates(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> ResponseDraft:
        if triage.request_type == "invalid":
            return ResponseDraft(
                status="replied",
                product_area=triage.product_area,
                response="I am sorry, this request is outside the scope of this support agent.",
                justification="Detected an out-of-scope or non-support request.",
                request_type="invalid",
                confidence=0.88,
            )

        if not chunks:
            return ResponseDraft(
                status="escalated",
                product_area=triage.product_area,
                response="I could not find this in the provided support documentation, so I am escalating it to a human support team member.",
                justification="No relevant local-corpus context was retrieved.",
                request_type=triage.request_type,
                confidence=0.15,
            )

        lead = chunks[0]
        confidence = self._estimate_confidence(triage, chunks)
        response = self._build_template_response(triage, chunks, confidence, intent_chunks=intent_chunks)
        justification = f"Used top local-corpus match from {lead.source_path} with retrieval score {lead.score:.4f}."
        if lead.supporting_source_paths:
            support_paths = ", ".join(lead.supporting_source_paths[:2])
            justification = f"{justification} Included supporting context from {support_paths}."
        if lead.graph_expansion_paths:
            justification = f"{justification} Used graph expansion paths: {'; '.join(lead.graph_expansion_paths[:2])}."
        if confidence < self.LOW_CONFIDENCE_TEMPLATE_THRESHOLD:
            justification = f"{justification} Response framed as a low-confidence close match."
        return ResponseDraft(
            status="replied",
            product_area=normalize_product_area(triage.domain, triage.product_area, ticket.text),
            response=response,
            justification=justification,
            request_type=triage.request_type,
            confidence=confidence,
        )

    @staticmethod
    def _fallback_response(triage: TriageResult) -> str:
        if triage.needs_escalation:
            return "This case needs a human support team member because it involves access, safety, or verification requirements."
        return "I could not find a fully grounded answer in the provided support corpus."

    @staticmethod
    def _safe_confidence(raw_confidence: object) -> float | None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return None
        if 0.05 <= confidence <= 1.0:
            return confidence
        return None

    def _estimate_confidence(self, triage: TriageResult, chunks: list[RetrievedChunk]) -> float:
        if triage.request_type == "invalid":
            return 0.90
        if not chunks:
            return 0.12

        top_score = chunks[0].score
        second_score = chunks[1].score if len(chunks) > 1 else 0.0
        score_span = max(0.0, min(1.0, (top_score - 0.018) / 0.018))
        gap_span = max(0.0, min(1.0, (top_score - second_score) / 0.01))
        domain_bonus = 0.05 if triage.domain != "None" else 0.0
        escalation_penalty = 0.10 if triage.needs_escalation else 0.0

        confidence = 0.30 + (0.42 * score_span) + (0.13 * gap_span) + domain_bonus - escalation_penalty
        return max(0.12, min(0.95, confidence))

    def _build_template_response(
        self,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        confidence: float,
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> str:
        if confidence < self.LOW_CONFIDENCE_TEMPLATE_THRESHOLD:
            return self._build_low_confidence_response(triage, chunks)

        if len(triage.intent_queries) <= 1:
            context_chunks = self._prefer_supporting_context(chunks, triage.intent_queries[0] if triage.intent_queries else "")
            steps = self._guidance_points_from_chunks(
                context_chunks,
                max_points=5,
                priority_text=triage.intent_queries[0] if triage.intent_queries else "",
            )
            lines = ["Based on the support documentation, these are the recommended next steps:"]
            lines.extend(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))
            lines.append("")
            lines.append("If this does not fully resolve your case, I recommend escalating it for manual review.")
            return "\n".join(lines)

        sections = ["This ticket appears to contain multiple intents. Based on the retrieved support guidance:"]
        for idx, query in enumerate(triage.intent_queries, start=1):
            query_chunks = (intent_chunks or {}).get(query) or [chunks[min(idx - 1, len(chunks) - 1)]]
            query_chunks = self._prefer_supporting_context(query_chunks, query)
            points = self._guidance_points_from_chunks(query_chunks, max_points=2, priority_text=query)
            sections.append(f"\nIntent {idx}: {query}")
            sections.extend(f"{point_index}. {point}" for point_index, point in enumerate(points, start=1))
        sections.append("\nIf any intent remains unsupported by the documentation, escalate that portion for human review.")
        return "\n".join(sections)

    def _build_low_confidence_response(self, triage: TriageResult, chunks: list[RetrievedChunk]) -> str:
        points = self._guidance_points_from_chunks(chunks, max_points=2)
        contact_email = self._extract_support_email(chunks)

        response = (
            "Based on a close match from the support documents we retrieved, the most relevant guidance appears to be:\n\n"
            + "\n".join(f"{idx}. {point}" for idx, point in enumerate(points, start=1))
            + "\n\n"
            "I am not fully confident that this is an exact match for your case."
        )
        if contact_email:
            response += f" If this does not resolve the issue, please contact {contact_email}."
        else:
            response += " If this does not resolve the issue, please contact the support team directly."

        if len(triage.intent_queries) > 1:
            response = (
                "This ticket appears to contain multiple intents, and the closest matching documentation is only a partial fit.\n\n"
                + response
            )
        return response

    def _extract_support_email(self, chunks: list[RetrievedChunk]) -> str | None:
        for chunk in chunks:
            matches = self.EMAIL_RE.findall(chunk.text)
            if matches:
                return matches[0]
        return None

    def _prefer_supporting_context(self, chunks: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        primary = chunks[0]
        if not self._is_link_hub(primary.text):
            return chunks

        query_terms = self._priority_terms(query)
        procedural = [
            chunk
            for chunk in chunks[1:]
            if self._chunk_has_procedural_content(chunk, query_terms)
        ]
        return procedural + [primary] + [chunk for chunk in chunks[1:] if chunk not in procedural]

    def _is_link_hub(self, text: str) -> bool:
        clean_text = self._clean_context_text(text)
        link_count = len(self.LINK_RE.findall(text))
        action_count = len(re.findall(r"\b(?:click|select|log in|deactivate|remove|save|confirm|navigate)\b", clean_text.lower()))
        return link_count >= 4 and action_count <= 3

    def _chunk_has_procedural_content(self, chunk: RetrievedChunk, query_terms: set[str]) -> bool:
        cleaned = self._clean_context_text(chunk.text).lower()
        action_hits = len(re.findall(r"\b(?:click|select|log in|deactivate|remove|save|confirm|navigate|search)\b", cleaned))
        query_hits = sum(1 for term in query_terms if term in cleaned)
        section_hit = any(term in chunk.section_title.lower() for term in ["deactivating", "user management", "accessing user"])
        return action_hits >= 2 and (query_hits >= 1 or section_hit)

    def _draft_from_payload(
        self,
        ticket: Ticket,
        triage: TriageResult,
        payload: dict[str, object],
        fallback_confidence: float,
    ) -> ResponseDraft:
        status = str(payload.get("status", "replied")).lower()
        request_type = str(payload.get("request_type", triage.request_type))
        llm_confidence = self._safe_confidence(payload.get("confidence"))
        response_text = self._coerce_response_text(payload.get("response", ""))
        justification_text = self._coerce_response_text(
            payload.get("justification", "LLM-generated corpus-grounded reply."),
        )
        confidence = llm_confidence if llm_confidence is not None else fallback_confidence
        status = self._normalize_llm_status(status, triage, response_text, confidence)
        return ResponseDraft(
            status=status,
            product_area=normalize_product_area(
                triage.domain,
                payload.get("product_area", triage.product_area),
                ticket.text,
            ),
            response=response_text or self._fallback_response(triage),
            justification=justification_text or "LLM-generated corpus-grounded reply.",
            request_type=request_type if request_type in {"product_issue", "feature_request", "bug", "invalid"} else triage.request_type,
            confidence=confidence,
        )

    def _draft_with_llm_text(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        fallback_confidence: float,
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> ResponseDraft | None:
        system_prompt = (
            "You are a support triage assistant. Use only the provided support context. "
            "Do not paste raw chunks, markdown tables, image markup, or relative URLs. "
            "Synthesize a concise answer in plain text. If the context is weak, say that it is a close match. "
            "Return exactly this plain-text format:\n"
            "STATUS: replied|escalated\n"
            "PRODUCT_AREA: <value>\n"
            "REQUEST_TYPE: <value>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "RESPONSE: <constructed reply>\n"
            "JUSTIFICATION: <short corpus-grounded reason>"
        )
        rendered = self.llm_client.generate_text(
            system_prompt,
            self._build_llm_user_prompt(ticket, triage, chunks, intent_chunks=intent_chunks),
        )
        if not rendered:
            return None

        response_text = self._extract_section(self.RESPONSE_RE, rendered)
        justification_text = self._extract_section(self.JUSTIFICATION_RE, rendered)
        status = self._extract_scalar(self.STATUS_RE, rendered, "replied").lower()
        product_area = self._extract_scalar(self.PRODUCT_AREA_RE, rendered, triage.product_area)
        request_type = self._extract_scalar(self.REQUEST_TYPE_RE, rendered, triage.request_type)
        confidence = self._safe_confidence(self._extract_scalar(self.CONFIDENCE_RE, rendered, ""))
        final_confidence = confidence if confidence is not None else fallback_confidence

        if not response_text:
            return None

        status = self._normalize_llm_status(status, triage, response_text, final_confidence)

        return ResponseDraft(
            status=status,
            product_area=normalize_product_area(triage.domain, product_area, ticket.text),
            response=response_text,
            justification=justification_text or "LLM-generated corpus-grounded reply.",
            request_type=request_type if request_type in {"product_issue", "feature_request", "bug", "invalid"} else triage.request_type,
            confidence=final_confidence,
        )

    def _normalize_llm_status(
        self,
        status: str,
        triage: TriageResult,
        response_text: str,
        confidence: float,
    ) -> str:
        if status not in {"replied", "escalated"}:
            status = "replied"
        if triage.needs_escalation:
            return "escalated"
        if status == "replied":
            return "replied"

        lowered = response_text.lower()
        escalation_language = any(
            phrase in lowered
            for phrase in (
                "escalat",
                "human review",
                "manual review",
                "support team",
                "cannot",
                "could not find",
                "insufficient",
                "not covered",
                "not authorized",
            )
        )
        if confidence >= self.LOW_CONFIDENCE_TEMPLATE_THRESHOLD and not escalation_language:
            return "replied"
        return "escalated"

    def _build_llm_user_prompt(
        self,
        ticket: Ticket,
        triage: TriageResult,
        chunks: list[RetrievedChunk],
        intent_chunks: dict[str, list[RetrievedChunk]] | None = None,
    ) -> str:
        return json.dumps(
            {
                "ticket": {
                    "issue": ticket.issue,
                    "subject": ticket.subject,
                    "company": ticket.company,
                },
                "triage": asdict(triage),
                "intent_queries": triage.intent_queries,
                "intent_context": {
                    query: [
                        {
                            "title": chunk.title,
                            "section_title": chunk.section_title,
                            "source_path": chunk.source_path,
                            "score": chunk.score,
                            "guidance_points": self._guidance_points_from_chunks([chunk], max_points=2, priority_text=query),
                        }
                        for chunk in query_chunks[:3]
                    ]
                    for query, query_chunks in (intent_chunks or {}).items()
                },
                "context": [
                    {
                        "title": chunk.title,
                        "section_title": chunk.section_title,
                        "source_path": chunk.source_path,
                        "source_url": chunk.source_url,
                        "supporting_source_paths": chunk.supporting_source_paths,
                        "graph_concepts": chunk.graph_concepts,
                        "graph_expansion_paths": chunk.graph_expansion_paths,
                        "score": chunk.score,
                        "guidance_points": self._guidance_points_from_chunks([chunk], max_points=2),
                        "excerpt": self._clean_context_text(chunk.text)[:450],
                    }
                    for chunk in chunks[:4]
                ],
            },
            ensure_ascii=True,
        )

    def _guidance_points_from_chunks(
        self,
        chunks: list[RetrievedChunk],
        max_points: int,
        priority_text: str = "",
    ) -> list[str]:
        points: list[str] = []
        seen: set[str] = set()
        priority_terms = self._priority_terms(priority_text)
        procedural_points = self._procedural_points_from_chunks(chunks, priority_terms)
        if procedural_points:
            return procedural_points[:max_points]
        for chunk in chunks:
            candidate_points = self._priority_sentences(chunk.text, priority_terms)
            candidate_points.extend(self._extract_guidance_points(chunk.text))
            if priority_terms:
                candidate_points = sorted(
                    candidate_points,
                    key=lambda point: self._priority_score(point, priority_terms),
                    reverse=True,
                )
            for point in candidate_points:
                normalized = point.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                points.append(point)
                if len(points) >= max_points:
                    return points
        if points:
            return points
        fallback = self._clean_context_text(chunks[0].text) if chunks else ""
        if fallback:
            return [fallback[:220].rstrip(" .") + "."]
        return ["Review the closest matching support documentation and escalate if the issue remains unresolved."]

    def _procedural_points_from_chunks(
        self,
        chunks: list[RetrievedChunk],
        priority_terms: set[str],
    ) -> list[str]:
        if not chunks or not (priority_terms & {"employee", "remove", "user", "users", "hiring", "account"}):
            return []

        combined = "\n".join(self._clean_context_text(chunk.text) for chunk in chunks[:3]).lower()
        if "user management" not in combined or "deactivate" not in combined:
            return []

        points: list[str] = []
        if "hackerrank for work" in combined:
            points.append("Log in to your HackerRank for Work account.")
        if "admin panel" in combined:
            points.append("Click your profile icon in the upper-right corner and select Admin Panel.")
        if "user management" in combined:
            points.append("From the admin panel, open User Management and use the search bar or filters to locate the employee's user account.")
        if "deactivate user" in combined or "deactivating a user" in combined:
            points.append("Click the ellipsis next to the user's name, select Deactivate User, and confirm by clicking Deactivate.")
        if "status changes to" in combined and "deactivated" in combined:
            points.append("Confirm that the user's status changes to Deactivated.")
        return points

    def _extract_guidance_points(self, text: str) -> list[str]:
        cleaned = self._clean_context_text(text)
        actionable: list[str] = []
        informational: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip(" -*\t")
            if not line or len(line) < 20:
                continue
            if line.lower().startswith(("subject:", "email body:", "last updated:", "last modified:")):
                continue
            if "colspan" in line.lower() or "rowspan" in line.lower():
                continue
            segments = re.split(r"(?<=[.!?])\s+|\s{2,}", line)
            for segment in segments:
                point = re.sub(r"^\d+[.)]\s*", "", segment).strip(" -")
                point = re.sub(r"\s+", " ", point).strip()
                if not point or len(point) < 20:
                    continue
                if point.endswith(":"):
                    continue
                if point.lower().endswith((" to", " and", " or")):
                    continue
                if self._looks_like_heading(point):
                    continue
                target = actionable if self._is_actionable_point(point) else informational
                if point not in actionable and point not in informational:
                    target.append(point)
        return (actionable + informational)[:4]

    @staticmethod
    def _clean_context_text(text: str) -> str:
        text = re.sub(r"^\[[^\n]+\]\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
        text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[([^\]]+)\]\(((?:https?://|mailto:)[^)]+)\)", r"\1 (\2)", text)
        text = re.sub(r"\[([^\]]+)\]\((?!https?://|mailto:)[^)]+\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = text.replace(" > ", " to ")
        text = re.sub(r"[_`*]", " ", text)
        text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _looks_like_heading(cls, point: str) -> bool:
        if len(point.split()) > 12:
            return False
        if any(char in point for char in ".!?"):
            return False
        words = [word for word in re.findall(r"[A-Za-z]+", point) if word]
        if not words:
            return False
        capitalized = sum(1 for word in words if word[0].isupper())
        return point.isupper() or (capitalized / len(words)) >= 0.6

    @classmethod
    def _is_actionable_point(cls, point: str) -> bool:
        lowered = point.lower()
        if cls.ACTION_START_RE.match(point):
            return True
        return any(
            phrase in lowered
            for phrase in (
                "you can",
                "to fix this",
                "to use",
                "to report",
                "to view",
                "to remove",
                "to update",
                "to pause",
                "to resume",
                "https://",
                "http://",
            )
        )

    @staticmethod
    def _priority_terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z0-9']+", text.lower())
            if len(term) > 3 and term not in {"please", "help", "issue", "problem", "today", "asap"}
        }

    @staticmethod
    def _priority_score(point: str, priority_terms: set[str]) -> int:
        lowered = point.lower()
        score = sum(1 for term in priority_terms if term in lowered)
        if any(term in priority_terms for term in {"refund", "billing", "payment"}) and any(
            keyword in lowered for keyword in ["refund", "contact", "support team", "not satisfied", "purchase"]
        ):
            score += 4
        if priority_terms & {"employee", "remove", "user", "users", "hiring", "account"} and any(
            keyword in lowered for keyword in ["deactivate", "deactivating", "user management", "admin panel", "ellipsis", "deactivated"]
        ):
            score += 5
        if any(term in priority_terms for term in {"infosec", "security", "forms"}) and any(
            keyword in lowered for keyword in ["infosec", "security", "form", "support"]
        ):
            score += 4
        return score

    def _priority_sentences(self, text: str, priority_terms: set[str]) -> list[str]:
        if not priority_terms:
            return []
        cleaned = self._clean_context_text(text)
        sentences = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if len(sentence.strip()) >= 20
        ]
        matches = []
        for sentence in sentences:
            if self._looks_like_heading(sentence):
                continue
            lowered = sentence.lower()
            if "refund" in priority_terms and any(
                keyword in lowered for keyword in ["refund", "not satisfied", "support team", "contact"]
            ):
                matches.append(sentence)
            elif priority_terms & {"employee", "remove", "user", "users", "hiring", "account"} and any(
                keyword in lowered for keyword in ["deactivate", "deactivating", "user management", "admin panel", "ellipsis", "deactivated"]
            ):
                matches.append(sentence)
            elif priority_terms & set(re.findall(r"[a-zA-Z0-9']+", lowered)):
                matches.append(sentence)
        return matches[:3]

    @staticmethod
    def _extract_section(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _extract_scalar(pattern: re.Pattern[str], text: str, default: str) -> str:
        match = pattern.search(text)
        if not match:
            return default
        return match.group(1).strip()

    @staticmethod
    def _coerce_response_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(ResponseAgent._coerce_response_text(item) for item in value).strip()
        if isinstance(value, dict):
            lines = []
            for key, item in value.items():
                rendered = ResponseAgent._coerce_response_text(item)
                if rendered:
                    lines.append(f"{key}: {rendered}")
            return "\n".join(lines).strip()
        return str(value).strip()
