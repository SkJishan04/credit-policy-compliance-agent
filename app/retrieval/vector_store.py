"""Chroma-backed persistent vector store for indexed policy chunks."""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import PolicyChunk, PolicyChunkMetadata, RetrievedChunk

logger = get_logger(__name__)


def _metadata_to_dict(metadata: PolicyChunkMetadata) -> dict:
    """Chroma metadata values must be str/int/float/bool/None -- flatten accordingly."""
    return {
        "section_id": metadata.section_id,
        "title": metadata.title,
        "borrower_type": metadata.borrower_type.value,
        "max_ltv_ratio": metadata.max_ltv_ratio,
        "max_unsecured_loan_lakhs": metadata.max_unsecured_loan_lakhs,
        "min_credit_score": metadata.min_credit_score,
        "required_documents": ", ".join(metadata.required_documents),
        "policy_version": metadata.policy_version,
        "source_file": metadata.source_file,
    }


def _dict_to_metadata(d: dict) -> PolicyChunkMetadata:
    docs = d.get("required_documents") or ""
    return PolicyChunkMetadata(
        section_id=d["section_id"],
        title=d["title"],
        borrower_type=d["borrower_type"],
        max_ltv_ratio=d.get("max_ltv_ratio"),
        max_unsecured_loan_lakhs=d.get("max_unsecured_loan_lakhs"),
        min_credit_score=d.get("min_credit_score"),
        required_documents=[s.strip() for s in docs.split(",") if s.strip()],
        policy_version=d["policy_version"],
        source_file=d["source_file"],
    )


class VectorStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def upsert_chunks(self, chunks: List[PolicyChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_metadata_to_dict(c.metadata) for c in chunks],
        )
        logger.info("Upserted %d chunks into collection '%s'", len(chunks), self.collection_name)

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        borrower_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        where = {"borrower_type": borrower_type} if borrower_type else None
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for id_, doc, meta, distance in zip(ids, docs, metas, distances):
            similarity = 1.0 - float(distance)  # cosine distance -> similarity
            chunk = PolicyChunk(id=id_, text=doc, metadata=_dict_to_metadata(meta))
            retrieved.append(RetrievedChunk(chunk=chunk, similarity_score=similarity))

        return retrieved