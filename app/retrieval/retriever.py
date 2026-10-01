"""High-level retrieval service combining embedding + vector search."""

from __future__ import annotations

from typing import List, Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import RetrievedChunk
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class PolicyRetriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        borrower_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        settings = get_settings()
        k = top_k or settings.top_k_retrieval
        query_embedding = self._embedding_model.embed_query(query)
        results = self._vector_store.query(query_embedding, top_k=k, borrower_type=borrower_type)
        logger.info(
            "Retrieved %d chunks for borrower_type=%s (query_len=%d)",
            len(results),
            borrower_type,
            len(query),
        )
        return results