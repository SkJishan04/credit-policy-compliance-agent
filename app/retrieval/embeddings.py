"""Local embedding model wrapper used for both indexing and querying.

Using a local sentence-transformers model (rather than a hosted embeddings API)
keeps the ingestion/retrieval path free of external API cost and latency, and
means the vector index is fully reproducible offline once the model is cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: np.ndarray = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Process-wide singleton so the (relatively expensive) model load happens once."""
    return EmbeddingModel()