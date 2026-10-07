"""Computes Recall@k and Mean Reciprocal Rank (MRR) for the retriever against
the labeled evaluation set in eval_dataset.py. Run as a script:

    python -m app.evaluation.retrieval_eval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.evaluation.eval_dataset import EVAL_CASES, EvalCase
from app.logging_config import configure_logging, get_logger
from app.retrieval.embeddings import get_embedding_model
from app.retrieval.retriever import PolicyRetriever
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalEvalReport:
    recall_at_k: float
    mrr: float
    num_cases: int
    k: int


def evaluate_retriever(retriever: PolicyRetriever, cases: List[EvalCase] = EVAL_CASES, k: int = 4) -> RetrievalEvalReport:
    hits = 0
    reciprocal_ranks = []

    for case in cases:
        results = retriever.retrieve(case.query, borrower_type=case.borrower_type, top_k=k)
        retrieved_ids = [r.chunk.metadata.section_id for r in results]

        found_rank = None
        for rank, section_id in enumerate(retrieved_ids, start=1):
            if section_id in case.expected_section_ids:
                found_rank = rank
                break

        if found_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
            logger.warning("MISS: query=%r expected=%s retrieved=%s", case.query, case.expected_section_ids, retrieved_ids)

    recall_at_k = hits / len(cases)
    mrr = sum(reciprocal_ranks) / len(cases)
    return RetrievalEvalReport(recall_at_k=recall_at_k, mrr=mrr, num_cases=len(cases), k=k)


def main() -> None:
    configure_logging()
    retriever = PolicyRetriever(embedding_model=get_embedding_model(), vector_store=VectorStore())
    report = evaluate_retriever(retriever)
    print(f"Retrieval Evaluation Report (n={report.num_cases}, k={report.k})")
    print(f"  Recall@{report.k}: {report.recall_at_k:.2%}")
    print(f"  MRR:       {report.mrr:.3f}")


if __name__ == "__main__":
    main()