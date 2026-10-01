"""FastAPI dependency providers -- wires config -> embeddings -> vector store ->
retriever -> LLM provider -> compliance agent -> compliance service as singletons.
"""

from __future__ import annotations

from functools import lru_cache

from app.llm.compliance_agent import ComplianceAgent
from app.llm.provider import AnthropicProvider
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.retriever import PolicyRetriever
from app.retrieval.vector_store import VectorStore
from app.services.compliance_service import ComplianceService


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_retriever() -> PolicyRetriever:
    return PolicyRetriever(embedding_model=get_embedding_model(), vector_store=get_vector_store())


@lru_cache
def get_compliance_agent() -> ComplianceAgent:
    return ComplianceAgent(provider=AnthropicProvider())


@lru_cache
def get_compliance_service() -> ComplianceService:
    return ComplianceService(retriever=get_retriever(), agent=get_compliance_agent())