"""CLI script to (re)build the vector index from the policy markdown source files.

Usage:
    python scripts/build_index.py [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingestion.chunker import ChunkerError, load_and_chunk_directory  # noqa: E402
from app.logging_config import configure_logging, get_logger  # noqa: E402
from app.retrieval.embeddings import get_embedding_model  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh the credit policy vector index.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection before indexing.")
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()

    logger.info("Loading and chunking policy documents from %s", settings.policy_data_path)
    try:
        chunks = load_and_chunk_directory(settings.policy_data_path)
    except ChunkerError as exc:
        logger.error("Failed to chunk policy documents: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Parsed %d policy chunks.", len(chunks))

    embedding_model = get_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = embedding_model.embed_texts(texts)

    vector_store = VectorStore()
    if args.reset:
        logger.info("Resetting existing collection '%s'.", vector_store.collection_name)
        vector_store.reset()

    vector_store.upsert_chunks(chunks, embeddings)
    logger.info("Index build complete. Total chunks in collection: %d", vector_store.count())


if __name__ == "__main__":
    main()