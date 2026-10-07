"""Unit tests for ComplianceService's rule-check and LLM/rule reconciliation logic."""

from __future__ import annotations

from typing import List

import pytest

from app.models.schemas import (
    ComplianceDecision,
    ComplianceRequest,
    LLMComplianceVerdict,
    PolicyChunk,
    PolicyChunkMetadata,
    RetrievedChunk,
)
from app.services.compliance_service import ComplianceService


def _msme_secured_chunk() -> RetrievedChunk:
    metadata = PolicyChunkMetadata(
        section_id="MSME-4.1",
        title="LTV Limits for Secured MSME Loans",
        borrower_type="MSME",
        max_ltv_ratio=0.75,
        policy_version="2.3",
        source_file="msme_lending_policy.md",
    )
    chunk = PolicyChunk(id="MSME-4.1", text="LTV must not exceed 75%.", metadata=metadata)
    return RetrievedChunk(chunk=chunk, similarity_score=0.95)


class FakeRetriever:
    def __init__(self, results: List[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, query: str, borrower_type=None, top_k=None):
        return self.results


class FakeAgent:
    def __init__(self, verdict: LLMComplianceVerdict) -> None:
        self.verdict = verdict

    def get_verdict(self, request, retrieved):
        return self.verdict


def test_rule_engine_overrides_llm_when_ltv_breach_detected():
    # LLM incorrectly says COMPLIANT, but loan_amount/collateral_value implies LTV > 75%.
    verdict = LLMComplianceVerdict(
        decision=ComplianceDecision.COMPLIANT,
        cited_section_ids=["MSME-4.1"],
        violated_section_ids=[],
        rationale="Looks fine.",
        confidence=0.8,
    )
    service = ComplianceService(retriever=FakeRetriever([_msme_secured_chunk()]), agent=FakeAgent(verdict))

    request = ComplianceRequest(
        borrower_type="MSME",
        loan_amount_lakhs=90,
        collateral_type="Commercial property",
        collateral_value_lakhs=100,  # LTV = 0.90 > 0.75 limit
    )

    response = service.evaluate(request)

    assert response.decision == ComplianceDecision.NON_COMPLIANT
    assert response.rule_check.ltv_compliant is False
    assert any("overriding to NON_COMPLIANT" in w for w in response.warnings)


def test_rule_engine_confirms_compliant_llm_decision():
    verdict = LLMComplianceVerdict(
        decision=ComplianceDecision.COMPLIANT,
        cited_section_ids=["MSME-4.1"],
        violated_section_ids=[],
        rationale="LTV is within the allowed limit.",
        confidence=0.9,
    )
    service = ComplianceService(retriever=FakeRetriever([_msme_secured_chunk()]), agent=FakeAgent(verdict))

    request = ComplianceRequest(
        borrower_type="MSME",
        loan_amount_lakhs=50,
        collateral_type="Commercial property",
        collateral_value_lakhs=100,  # LTV = 0.50 <= 0.75 limit
    )

    response = service.evaluate(request)

    assert response.decision == ComplianceDecision.COMPLIANT
    assert response.rule_check.ltv_compliant is True
    assert response.warnings == []


def test_review_required_when_deterministic_pass_but_llm_flags_concern():
    verdict = LLMComplianceVerdict(
        decision=ComplianceDecision.NON_COMPLIANT,
        cited_section_ids=["MSME-4.1"],
        violated_section_ids=[],
        rationale="Missing Udyam registration certificate.",
        confidence=0.7,
    )
    service = ComplianceService(retriever=FakeRetriever([_msme_secured_chunk()]), agent=FakeAgent(verdict))

    request = ComplianceRequest(
        borrower_type="MSME",
        loan_amount_lakhs=50,
        collateral_type="Commercial property",
        collateral_value_lakhs=100,
    )

    response = service.evaluate(request)

    assert response.decision == ComplianceDecision.REVIEW_REQUIRED
    assert any("routed to manual review" in w for w in response.warnings)


def test_no_retrieved_chunks_returns_review_required():
    verdict = LLMComplianceVerdict(
        decision=ComplianceDecision.COMPLIANT,
        cited_section_ids=[],
        violated_section_ids=[],
        rationale="n/a",
        confidence=0.0,
    )
    service = ComplianceService(retriever=FakeRetriever([]), agent=FakeAgent(verdict))

    request = ComplianceRequest(
        borrower_type="Corporate",
        loan_amount_lakhs=10,
        collateral_type="None",
    )

    response = service.evaluate(request)

    assert response.decision == ComplianceDecision.REVIEW_REQUIRED
    assert response.cited_sections == []