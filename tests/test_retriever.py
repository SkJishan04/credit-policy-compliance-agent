"""Unit tests for PolicyRetriever using fake embedding/vector-store collaborators."""

from __future__ import annotations

from typing import List, Optional

from app.models.schemas import PolicyChunk, PolicyChunkMetadata, RetrievedChunk
from app.retrieval.retriever import PolicyRetriever


class FakeEmbeddingModel:
    def embed_query(self, text: str) -> List[float]:
        return [float(len(text))]


class FakeVectorStore:
    def __init__(self, canned_results: List[RetrievedChunk]) -> None:
        self.canned_results = canned_results
        self.last_call_kwargs: dict = {}

    def query(self, query_embedding: List[float], top_k: int = 4, borrower_type: Optional[str] = None):
        self.last_call_kwargs = {"query_embedding": query_embedding, "top_k": top_k, "borrower_type": borrower_type}
        return self.canned_results[:top_k]


def _make_chunk(section_id: str) -> PolicyChunk:
    metadata = PolicyChunkMetadata(
        section_id=section_id,
        title=f"Title for {section_id}",
        borrower_type="MSME",
        max_ltv_ratio=0.75,
        policy_version="1.0",
        source_file="test.md",
    )
    return PolicyChunk(id=section_id, text="some narrative text", metadata=metadata)


def test_retriever_passes_borrower_type_filter_through_to_vector_store():
    canned = [RetrievedChunk(chunk=_make_chunk("MSME-4.1"), similarity_score=0.9)]
    vector_store = FakeVectorStore(canned)
    retriever = PolicyRetriever(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)

    results = retriever.retrieve("secured MSME loan LTV", borrower_type="MSME", top_k=2)

    assert vector_store.last_call_kwargs["borrower_type"] == "MSME"
    assert vector_store.last_call_kwargs["top_k"] == 2
    assert len(results) == 1
    assert results[0].chunk.metadata.section_id == "MSME-4.1"


def test_retriever_respects_default_top_k_when_not_specified(monkeypatch):
    canned = [RetrievedChunk(chunk=_make_chunk(f"SEC-{i}"), similarity_score=1.0 - i * 0.1) for i in range(5)]
    vector_store = FakeVectorStore(canned)
    retriever = PolicyRetriever(embedding_model=FakeEmbeddingModel(), vector_store=vector_store)

    results = retriever.retrieve("any query")

    # default top_k_retrieval from Settings is 4
    assert vector_store.last_call_kwargs["top_k"] == 4
    assert len(results) == 4