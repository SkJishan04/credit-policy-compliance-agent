"""A small hand-labeled evaluation set mapping natural-language queries to the
policy section_ids that should be retrieved. Used to measure retrieval quality
(recall@k, MRR) independently of the LLM generation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    query: str
    borrower_type: str
    expected_section_ids: List[str]


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        query="What is the maximum LTV for a secured MSME loan?",
        borrower_type="MSME",
        expected_section_ids=["MSME-4.1"],
    ),
    EvalCase(
        query="Is there a cap on unsecured MSME lending without collateral?",
        borrower_type="MSME",
        expected_section_ids=["MSME-4.2"],
    ),
    EvalCase(
        query="What documents does an MSME applicant need for Udyam registration?",
        borrower_type="MSME",
        expected_section_ids=["MSME-4.3"],
    ),
    EvalCase(
        query="What is the LTV limit for a secured retail individual loan backed by property?",
        borrower_type="Retail Individual",
        expected_section_ids=["RETAIL-2.1"],
    ),
    EvalCase(
        query="What is the cap for an unsecured personal loan to a retail customer?",
        borrower_type="Retail Individual",
        expected_section_ids=["RETAIL-2.2"],
    ),
    EvalCase(
        query="What is the LTV cap for a secured corporate lending facility?",
        borrower_type="Corporate",
        expected_section_ids=["CORP-6.1"],
    ),
    EvalCase(
        query="Does an unsecured corporate credit line require a minimum credit rating?",
        borrower_type="Corporate",
        expected_section_ids=["CORP-6.2"],
    ),
    EvalCase(
        query="Is a board resolution required for corporate borrowing?",
        borrower_type="Corporate",
        expected_section_ids=["CORP-6.3"],
    ),
]