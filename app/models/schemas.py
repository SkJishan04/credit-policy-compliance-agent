"""Pydantic schemas shared across ingestion, retrieval, LLM, service and API layers."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BorrowerType(str, Enum):
    MSME = "MSME"
    RETAIL_INDIVIDUAL = "Retail Individual"
    CORPORATE = "Corporate"


class ComplianceDecision(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PolicyChunkMetadata(BaseModel):
    """Structured facts extracted from a policy section's YAML frontmatter."""

    section_id: str
    title: str
    borrower_type: BorrowerType
    max_ltv_ratio: Optional[float] = None
    max_unsecured_loan_lakhs: Optional[float] = None
    min_credit_score: Optional[int] = None
    required_documents: List[str] = Field(default_factory=list)
    policy_version: str
    source_file: str


class PolicyChunk(BaseModel):
    """A single retrievable unit of policy text plus its structured metadata."""

    id: str
    text: str
    metadata: PolicyChunkMetadata


class RetrievedChunk(BaseModel):
    chunk: PolicyChunk
    similarity_score: float


class ComplianceRequest(BaseModel):
    borrower_type: BorrowerType
    loan_amount_lakhs: float = Field(..., gt=0, description="Requested loan amount in INR Lakhs")
    collateral_type: str = Field(..., description="Free-text description of collateral offered, or 'None'")
    collateral_value_lakhs: Optional[float] = Field(
        default=None,
        ge=0,
        description="Assessed value of the collateral in INR Lakhs, if any. Required to compute LTV.",
    )


class LLMComplianceVerdict(BaseModel):
    """Structured output produced by the LLM via forced tool use."""

    decision: ComplianceDecision
    cited_section_ids: List[str]
    violated_section_ids: List[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RuleCheckResult(BaseModel):
    """Deterministic, code-computed cross-check used for hallucination mitigation."""

    computed_ltv: Optional[float] = None
    max_allowed_ltv: Optional[float] = None
    ltv_compliant: Optional[bool] = None
    loan_amount_lakhs: Optional[float] = None
    max_unsecured_loan_lakhs: Optional[float] = None
    unsecured_compliant: Optional[bool] = None
    warnings: List[str] = Field(default_factory=list)


class CitedSection(BaseModel):
    section_id: str
    title: str
    excerpt: str


class ComplianceResponse(BaseModel):
    decision: ComplianceDecision
    verdict_summary: str
    cited_sections: List[CitedSection]
    rule_check: RuleCheckResult
    confidence: float
    latency_ms: float
    warnings: List[str] = Field(default_factory=list)