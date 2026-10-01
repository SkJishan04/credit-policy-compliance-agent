"""Orchestrates retrieval + LLM generation + a deterministic rule cross-check.

Design rationale (hallucination mitigation):
LLMs are good at synthesizing qualitative rationale and citing sources, but are
unreliable at exact arithmetic and can silently misapply a threshold. This
service therefore treats the LLM's *decision* as advisory and recomputes the
numeric compliance facts (LTV ratio, unsecured cap) directly from the same
retrieved chunk metadata using plain Python. When the deterministic check and
the LLM's decision disagree, the deterministic check wins and the response is
flagged for manual review, rather than silently trusting either side.
"""

from __future__ import annotations

import time
from typing import List, Optional

from app.llm.compliance_agent import ComplianceAgent, ComplianceAgentError
from app.logging_config import get_logger
from app.models.schemas import (
    CitedSection,
    ComplianceDecision,
    ComplianceRequest,
    ComplianceResponse,
    LLMComplianceVerdict,
    PolicyChunkMetadata,
    RetrievedChunk,
    RuleCheckResult,
)
from app.retrieval.retriever import PolicyRetriever

logger = get_logger(__name__)


class ComplianceService:
    def __init__(self, retriever: PolicyRetriever, agent: ComplianceAgent) -> None:
        self._retriever = retriever
        self._agent = agent

    @staticmethod
    def _build_query(request: ComplianceRequest) -> str:
        return (
            f"{request.borrower_type.value} loan of {request.loan_amount_lakhs} lakhs "
            f"with collateral: {request.collateral_type}"
        )

    @staticmethod
    def _select_threshold_metadata(retrieved: List[RetrievedChunk]) -> Optional[PolicyChunkMetadata]:
        """Pick the most relevant chunk that actually carries a numeric threshold."""
        for r in retrieved:
            if r.chunk.metadata.max_ltv_ratio is not None or r.chunk.metadata.max_unsecured_loan_lakhs is not None:
                return r.chunk.metadata
        return None

    def _rule_check(self, request: ComplianceRequest, retrieved: List[RetrievedChunk]) -> RuleCheckResult:
        metadata = self._select_threshold_metadata(retrieved)
        result = RuleCheckResult(loan_amount_lakhs=request.loan_amount_lakhs)

        if metadata is None:
            result.warnings.append("No retrieved policy section carried a numeric threshold to check against.")
            return result

        has_collateral = (
            request.collateral_value_lakhs is not None
            and request.collateral_value_lakhs > 0
            and request.collateral_type.strip().lower() not in {"", "none", "no collateral"}
        )

        if has_collateral and metadata.max_ltv_ratio is not None:
            computed_ltv = round(request.loan_amount_lakhs / request.collateral_value_lakhs, 4)
            result.computed_ltv = computed_ltv
            result.max_allowed_ltv = metadata.max_ltv_ratio
            result.ltv_compliant = computed_ltv <= metadata.max_ltv_ratio
            if not result.ltv_compliant:
                result.warnings.append(
                    f"Computed LTV {computed_ltv:.0%} exceeds the {metadata.max_ltv_ratio:.0%} limit "
                    f"under policy section {metadata.section_id}."
                )
        elif not has_collateral and metadata.max_unsecured_loan_lakhs is not None:
            result.max_unsecured_loan_lakhs = metadata.max_unsecured_loan_lakhs
            result.unsecured_compliant = request.loan_amount_lakhs <= metadata.max_unsecured_loan_lakhs
            if not result.unsecured_compliant:
                result.warnings.append(
                    f"Unsecured loan amount {request.loan_amount_lakhs} lakhs exceeds the "
                    f"{metadata.max_unsecured_loan_lakhs} lakh cap under policy section {metadata.section_id}."
                )
        else:
            result.warnings.append(
                "Collateral value was not provided, so LTV could not be computed. "
                "Provide 'Collateral Value (₹ Lakhs)' to enable the deterministic LTV check."
            )

        return result

    @staticmethod
    def _reconcile(
        verdict: LLMComplianceVerdict, rule_check: RuleCheckResult
    ) -> tuple[ComplianceDecision, List[str]]:
        warnings: List[str] = []
        rule_says_noncompliant = rule_check.ltv_compliant is False or rule_check.unsecured_compliant is False
        rule_has_definitive_pass = (
            rule_check.ltv_compliant is True or rule_check.unsecured_compliant is True
        ) and not rule_says_noncompliant

        if rule_says_noncompliant:
            final_decision = ComplianceDecision.NON_COMPLIANT
            if verdict.decision != ComplianceDecision.NON_COMPLIANT:
                warnings.append(
                    f"LLM suggested {verdict.decision.value} but the deterministic rule engine detected "
                    "a policy breach; overriding to NON_COMPLIANT."
                )
        elif rule_has_definitive_pass and verdict.decision == ComplianceDecision.NON_COMPLIANT:
            final_decision = ComplianceDecision.REVIEW_REQUIRED
            warnings.append(
                "Deterministic numeric checks passed but the LLM flagged a qualitative concern "
                "(e.g. missing documentation); routed to manual review."
            )
        else:
            final_decision = verdict.decision

        return final_decision, warnings

    def evaluate(self, request: ComplianceRequest) -> ComplianceResponse:
        start = time.perf_counter()
        query = self._build_query(request)
        retrieved = self._retriever.retrieve(query, borrower_type=request.borrower_type.value)

        if not retrieved:
            latency_ms = (time.perf_counter() - start) * 1000
            return ComplianceResponse(
                decision=ComplianceDecision.REVIEW_REQUIRED,
                verdict_summary="No matching policy sections were found for this borrower type. Manual review required.",
                cited_sections=[],
                rule_check=RuleCheckResult(loan_amount_lakhs=request.loan_amount_lakhs),
                confidence=0.0,
                latency_ms=latency_ms,
                warnings=["Retrieval returned zero results."],
            )

        try:
            verdict = self._agent.get_verdict(request, retrieved)
        except ComplianceAgentError as exc:
            logger.error("Compliance agent failed: %s", exc)
            latency_ms = (time.perf_counter() - start) * 1000
            return ComplianceResponse(
                decision=ComplianceDecision.REVIEW_REQUIRED,
                verdict_summary="Automated evaluation failed; routed to manual review.",
                cited_sections=[],
                rule_check=RuleCheckResult(loan_amount_lakhs=request.loan_amount_lakhs),
                confidence=0.0,
                latency_ms=latency_ms,
                warnings=[str(exc)],
            )

        rule_check = self._rule_check(request, retrieved)
        final_decision, reconciliation_warnings = self._reconcile(verdict, rule_check)

        cited_sections = [
            CitedSection(section_id=r.chunk.metadata.section_id, title=r.chunk.metadata.title, excerpt=r.chunk.text[:280])
            for r in retrieved
            if r.chunk.metadata.section_id in verdict.cited_section_ids
        ]
        if not cited_sections:
            cited_sections = [
                CitedSection(section_id=r.chunk.metadata.section_id, title=r.chunk.metadata.title, excerpt=r.chunk.text[:280])
                for r in retrieved[:2]
            ]

        latency_ms = (time.perf_counter() - start) * 1000
        all_warnings = [*rule_check.warnings, *reconciliation_warnings]

        return ComplianceResponse(
            decision=final_decision,
            verdict_summary=verdict.rationale,
            cited_sections=cited_sections,
            rule_check=rule_check,
            confidence=verdict.confidence,
            latency_ms=latency_ms,
            warnings=all_warnings,
        )