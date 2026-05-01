from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from graph_store import SupportGraphStore
from models import RetrievedChunk


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    section_id: str
    domain: str
    source_path: str
    source_url: str
    title: str
    section_title: str
    text: str
    section_index: int
    chunk_index: int


@dataclass(slots=True)
class SectionRecord:
    section_id: str
    document_id: str
    domain: str
    source_path: str
    source_url: str
    title: str
    section_title: str
    section_index: int


class HybridRetriever:
    QUERY_STOP_WORDS = {
        "the", "and", "for", "with", "that", "this", "have", "your", "from", "into", "when", "what",
        "please", "help", "issue", "problem", "asap", "today", "about", "then", "they", "them", "want",
        "need", "would", "could", "should", "been", "stopped", "give", "make", "between", "really",
    }
    FACET_TERMS = {
        "billing", "refund", "payment", "pay", "purchase", "subscription", "invoice", "charge",
        "access", "login", "admin", "owner", "workspace", "fraud", "security", "dispute", "merchant",
        "delete", "export", "conversation", "privacy", "blocked", "stolen", "cash", "interview",
    }

    def __init__(
        self,
        repo_root: Path,
        chunk_size: int = 1100,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.repo_root = repo_root
        self.data_root = repo_root / "data"
        self.chunk_size = chunk_size
        self.dense_model_name = dense_model_name
        self.cache_dir = repo_root / "code"
        self.chroma_dir = self.cache_dir / ".chroma_db"
        self.manifest_path = self.cache_dir / ".retrieval_manifest.json"
        self.collection_name = "support_chunks_v3"

        self.records = self._build_records()
        self.records_by_id = {record.chunk_id: record for record in self.records}
        self.section_members = self._build_section_members()
        self.section_records = self._build_section_records()
        self.graph_store = SupportGraphStore(self.records, self.section_members)
        self.tokenized_texts = [self._tokenize(record.text) for record in self.records]
        self.bm25_global = BM25Okapi(self.tokenized_texts)
        self.domain_indexes = self._build_domain_indexes()

        self.embedding_model = SentenceTransformer(self.dense_model_name)
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self._sync_chroma_collection()

    def retrieve(self, query: str, domain: str, limit: int = 5) -> list[RetrievedChunk]:
        target_domain = domain if domain in {"HackerRank", "Claude", "Visa"} else None
        search_limit = max(limit * 8, 24)
        bm25_results = self._bm25_search(query, target_domain=target_domain, limit=search_limit)
        dense_results = self._dense_search(query, target_domain=target_domain, limit=search_limit)
        fused_scores = self._rrf_fuse(
            bm25_results=bm25_results,
            dense_results=dense_results,
            bm25_weight=1.0,
            dense_weight=1.2,
            k=60,
        )
        if not fused_scores:
            return []
        bundles = self._assemble_evidence_bundles(query, fused_scores, limit=limit)
        return bundles

    def describe_backend(self) -> str:
        return (
            f"Section-aware hybrid retrieval using BM25Okapi + Chroma dense search "
            f"({self.dense_model_name}) with graph-expanded evidence bundles."
        )

    def _build_records(self) -> list[ChunkRecord]:
        records: list[ChunkRecord] = []
        for md_file in sorted(self.data_root.rglob("*.md")):
            domain = self._domain_from_path(md_file)
            raw_content = md_file.read_text(encoding="utf-8", errors="ignore")
            frontmatter, content = self._split_frontmatter(raw_content)
            title = self._extract_title(content, frontmatter.get("title", md_file.stem))
            source_url = frontmatter.get("source_url", "")
            relative_path = str(md_file.relative_to(self.repo_root))
            path_key = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
            document_id = f"{domain}-{path_key}"
            sections = self._split_sections(content, title)
            for section_index, (section_title, section_text) in enumerate(sections):
                section_id = f"{document_id}-s{section_index}"
                chunk_texts = self._chunk_section(section_text)
                for chunk_index, text in enumerate(chunk_texts):
                    records.append(
                        ChunkRecord(
                            chunk_id=f"{section_id}-c{chunk_index}",
                            document_id=document_id,
                            section_id=section_id,
                            domain=domain,
                            source_path=relative_path,
                            source_url=source_url,
                            title=title,
                            section_title=section_title,
                            text=text,
                            section_index=section_index,
                            chunk_index=chunk_index,
                        )
                    )
        return records

    def _build_section_members(self) -> dict[str, list[ChunkRecord]]:
        section_members: dict[str, list[ChunkRecord]] = {}
        for record in self.records:
            section_members.setdefault(record.section_id, []).append(record)
        for records in section_members.values():
            records.sort(key=lambda item: item.chunk_index)
        return section_members

    def _build_section_records(self) -> dict[str, SectionRecord]:
        sections: dict[str, SectionRecord] = {}
        for record in self.records:
            if record.section_id not in sections:
                sections[record.section_id] = SectionRecord(
                    section_id=record.section_id,
                    document_id=record.document_id,
                    domain=record.domain,
                    source_path=record.source_path,
                    source_url=record.source_url,
                    title=record.title,
                    section_title=record.section_title,
                    section_index=record.section_index,
                )
        return sections

    def _build_domain_indexes(self) -> dict[str, dict[str, Any]]:
        indexes: dict[str, dict[str, Any]] = {}
        for domain in ("HackerRank", "Claude", "Visa"):
            domain_records = [record for record in self.records if record.domain == domain]
            tokenized = [self._tokenize(record.text) for record in domain_records]
            if not tokenized:
                continue
            indexes[domain] = {
                "records": domain_records,
                "bm25": BM25Okapi(tokenized),
            }
        return indexes

    def _sync_chroma_collection(self) -> None:
        manifest = self._current_manifest()
        existing_manifest = None
        if self.manifest_path.exists():
            existing_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        if existing_manifest == manifest and self.collection.count() == len(self.records):
            return

        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except NotFoundError:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)

        batch_size = 128
        for start in range(0, len(self.records), batch_size):
            batch = self.records[start:start + batch_size]
            documents = [record.text for record in batch]
            embeddings = self.embedding_model.encode(documents, normalize_embeddings=True).tolist()
            self.collection.add(
                ids=[record.chunk_id for record in batch],
                documents=documents,
                embeddings=embeddings,
                metadatas=[
                    {
                        "domain": record.domain,
                        "source_path": record.source_path,
                        "title": record.title,
                        "section_title": record.section_title,
                        "section_id": record.section_id,
                        "document_id": record.document_id,
                    }
                    for record in batch
                ],
            )

        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _current_manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(record.chunk_id.encode("utf-8"))
            digest.update(record.source_path.encode("utf-8"))
            digest.update(record.section_id.encode("utf-8"))
            digest.update(record.section_title.encode("utf-8"))
            digest.update(record.text.encode("utf-8"))
        return {
            "collection_name": self.collection_name,
            "dense_model_name": self.dense_model_name,
            "chunk_size": self.chunk_size,
            "record_count": len(self.records),
            "digest": digest.hexdigest(),
        }

    def _bm25_search(self, query: str, target_domain: str | None, limit: int) -> list[tuple[str, float]]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        if target_domain and target_domain in self.domain_indexes:
            domain_index = self.domain_indexes[target_domain]
            scores = domain_index["bm25"].get_scores(query_terms)
            records = domain_index["records"]
        else:
            scores = self.bm25_global.get_scores(query_terms)
            records = self.records

        ranked_pairs = sorted(zip(records, scores), key=lambda item: item[1], reverse=True)
        return [
            (record.chunk_id, float(score))
            for record, score in ranked_pairs[:limit]
            if score > 0
        ]

    def _dense_search(self, query: str, target_domain: str | None, limit: int) -> list[tuple[str, float]]:
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        where = {"domain": target_domain} if target_domain else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["distances", "metadatas"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        dense_results: list[tuple[str, float]] = []
        for chunk_id, distance in zip(ids, distances):
            score = 1.0 / (1.0 + float(distance))
            dense_results.append((chunk_id, score))
        return dense_results

    @staticmethod
    def _rrf_fuse(
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
        bm25_weight: float,
        dense_weight: float,
        k: int,
    ) -> dict[str, float]:
        fused: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (bm25_weight / (k + rank))
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (dense_weight / (k + rank))
        return fused

    def _assemble_evidence_bundles(
        self,
        query: str,
        fused_scores: dict[str, float],
        limit: int,
    ) -> list[RetrievedChunk]:
        ranked_ids = [chunk_id for chunk_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]
        query_terms = self._query_terms(query)
        bundles: list[tuple[float, RetrievedChunk]] = []
        seen_sections: set[str] = set()

        for chunk_id in ranked_ids[: max(limit * 6, 16)]:
            primary = self.records_by_id[chunk_id]
            if primary.section_id in seen_sections:
                continue
            bundle_records = self._records_for_section(primary.section_id)
            support_records = self._select_support_records(
                query_terms=query_terms,
                ranked_ids=ranked_ids,
                fused_scores=fused_scores,
                primary=primary,
                bundle_records=bundle_records,
            )
            graph_expansions = self.graph_store.related_section_ids(
                primary=primary,
                query_terms=query_terms,
                target_terms=query_terms - self._coverage_terms(bundle_records + support_records),
                limit=3,
            )
            graph_records: list[ChunkRecord] = []
            graph_paths: list[str] = []
            for expansion in graph_expansions:
                if expansion.section_id == primary.section_id:
                    continue
                graph_records.extend(self._records_for_section(expansion.section_id))
                graph_paths.append(expansion.path)
            all_records = self._dedupe_records(bundle_records + support_records + graph_records)
            bundle_score = self._bundle_score(primary, all_records, fused_scores, query_terms)
            bundles.append((bundle_score, self._build_bundle(primary, all_records, bundle_score, graph_paths)))
            seen_sections.add(primary.section_id)

        bundles.sort(key=lambda item: item[0], reverse=True)
        return [bundle for _, bundle in bundles[:limit]]

    def _records_for_section(self, section_id: str) -> list[ChunkRecord]:
        return list(self.section_members.get(section_id, []))

    def _select_support_records(
        self,
        query_terms: set[str],
        ranked_ids: list[str],
        fused_scores: dict[str, float],
        primary: ChunkRecord,
        bundle_records: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        covered_terms = self._coverage_terms(bundle_records)
        uncovered_terms = query_terms - covered_terms
        coverage_ratio = (len(query_terms & covered_terms) / len(query_terms)) if query_terms else 1.0
        if coverage_ratio >= 0.7:
            return []
        if not uncovered_terms:
            return []

        primary_terms = self._record_terms(primary)
        support_candidates: list[tuple[float, ChunkRecord]] = []
        for chunk_id in ranked_ids[:20]:
            candidate = self.records_by_id[chunk_id]
            if candidate.document_id == primary.document_id:
                continue
            candidate_terms = self._record_terms(candidate)
            candidate_uncovered = len(candidate_terms & uncovered_terms)
            candidate_primary_overlap = len(candidate_terms & primary_terms)
            facet_bonus = 1 if candidate_terms & (query_terms & self.FACET_TERMS) else 0
            if candidate_uncovered == 0:
                continue
            if candidate_primary_overlap < 2 and facet_bonus == 0:
                continue
            score = fused_scores.get(candidate.chunk_id, 0.0) + (0.003 * candidate_uncovered) + (0.002 * candidate_primary_overlap) + (0.002 * facet_bonus)
            support_candidates.append((score, candidate))

        support_candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[ChunkRecord] = []
        seen_sections: set[str] = set()
        for _, candidate in support_candidates[:3]:
            if candidate.section_id in seen_sections:
                continue
            selected.extend(self._records_for_section(candidate.section_id))
            seen_sections.add(candidate.section_id)
            if len(seen_sections) >= 2:
                break
        return selected

    def _build_bundle(
        self,
        primary: ChunkRecord,
        records: list[ChunkRecord],
        bundle_score: float,
        graph_paths: list[str],
    ) -> RetrievedChunk:
        grouped_by_section: dict[str, list[ChunkRecord]] = {}
        for record in records:
            grouped_by_section.setdefault(record.section_id, []).append(record)

        ordered_sections = sorted(
            grouped_by_section.values(),
            key=lambda section_records: (section_records[0].document_id != primary.document_id, section_records[0].section_index),
        )
        parts: list[str] = []
        supporting_paths: list[str] = []
        for section_records in ordered_sections:
            section_records.sort(key=lambda item: item.chunk_index)
            section = section_records[0]
            label = "Primary Evidence" if section.section_id == primary.section_id else "Supporting Evidence"
            parts.append(
                f"[{label} | {section.title} | {section.section_title} | {section.source_path}]\n"
                + "\n\n".join(record.text for record in section_records)
            )
            if section.source_path != primary.source_path and section.source_path not in supporting_paths:
                supporting_paths.append(section.source_path)

        return RetrievedChunk(
            chunk_id=primary.chunk_id,
            domain=primary.domain,
            source_path=primary.source_path,
            title=primary.title,
            text="\n\n".join(parts),
            score=bundle_score,
            document_id=primary.document_id,
            section_id=primary.section_id,
            section_title=primary.section_title,
            source_url=primary.source_url,
            supporting_source_paths=supporting_paths,
            graph_concepts=self.graph_store.concepts_for_section(primary.section_id),
            graph_expansion_paths=graph_paths,
        )

    def _bundle_score(
        self,
        primary: ChunkRecord,
        records: list[ChunkRecord],
        fused_scores: dict[str, float],
        query_terms: set[str],
    ) -> float:
        primary_score = fused_scores.get(primary.chunk_id, 0.0)
        support_scores = [fused_scores.get(record.chunk_id, 0.0) for record in records if record.chunk_id != primary.chunk_id]
        coverage_bonus = 0.003 * len(self._coverage_terms(records) & query_terms)
        support_bonus = min(sum(sorted(support_scores, reverse=True)[:3]) * 0.35, 0.012)
        return primary_score + support_bonus + coverage_bonus

    def _coverage_terms(self, records: list[ChunkRecord]) -> set[str]:
        terms: set[str] = set()
        for record in records:
            terms.update(self._record_terms(record))
        return terms

    def _record_terms(self, record: ChunkRecord) -> set[str]:
        combined = " ".join([record.title, record.section_title, record.source_path, record.text[:500]])
        return self._query_terms(combined)

    def _query_terms(self, text: str) -> set[str]:
        return {
            term
            for term in self._tokenize(text)
            if term not in self.QUERY_STOP_WORDS
        }

    @staticmethod
    def _dedupe_records(records: list[ChunkRecord]) -> list[ChunkRecord]:
        ordered: list[ChunkRecord] = []
        seen: set[str] = set()
        for record in records:
            if record.chunk_id in seen:
                continue
            seen.add(record.chunk_id)
            ordered.append(record)
        return ordered

    def _split_frontmatter(self, content: str) -> tuple[dict[str, str], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, flags=re.DOTALL)
        if not match:
            return {}, content
        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
        return metadata, match.group(2)

    def _split_sections(self, content: str, fallback_title: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, list[str]]] = []
        current_title = fallback_title
        current_lines: list[str] = []
        for line in content.splitlines():
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = heading_match.group(2).strip() or fallback_title
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            sections.append((current_title, current_lines))
        return [
            (title, "\n".join(lines).strip())
            for title, lines in sections
            if "\n".join(lines).strip()
        ]

    def _chunk_section(self, section_text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
        if not paragraphs:
            return []
        chunks: list[str] = []
        current_parts: list[str] = []
        current_length = 0
        overlap_paragraphs = 1
        for paragraph in paragraphs:
            paragraph_length = len(paragraph) + 2
            if current_parts and current_length + paragraph_length > self.chunk_size:
                chunks.append("\n\n".join(current_parts))
                current_parts = current_parts[-overlap_paragraphs:] if overlap_paragraphs and len(current_parts) > overlap_paragraphs else current_parts[-1:]
                current_length = sum(len(part) + 2 for part in current_parts)
            current_parts.append(paragraph)
            current_length += paragraph_length
        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks

    @staticmethod
    def _extract_title(content: str, default: str) -> str:
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return default

    @staticmethod
    def _domain_from_path(path: Path) -> str:
        if "hackerrank" in path.parts:
            return "HackerRank"
        if "claude" in path.parts:
            return "Claude"
        if "visa" in path.parts:
            return "Visa"
        return "None"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [term for term in re.findall(r"[a-zA-Z0-9']+", text.lower()) if len(term) > 2]
