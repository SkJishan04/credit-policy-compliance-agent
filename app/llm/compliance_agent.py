"""Builds prompts from retrieved policy context and produces a structured,
citation-grounded compliance verdict via forced tool use.

The agent is explicitly instructed to only reason over the supplied context
and to cite section_ids it relied on. Numeric compliance thresholds are never
trusted from the LLM's free-text rationale alone -- they are independently
recomputed by ComplianceService against the same retrieved metadata (see
services/compliance_service.py) as a hallucination-mitigation safeguard.
"""

from __future__ import annotations

from typing import List

from pydantic import ValidationError

from app.llm.provider import AnthropicProvider, LLMProviderError
from app.logging_config import get_logger
from app.models.schemas import ComplianceRequest, LLMComplianceVerdict, RetrievedChunk

logger = get_logger(__name__)

_TOOL_NAME = "emit_compliance_verdict"
_TOOL_DESCRIPTION = (
    "Emit a structured credit policy compliance verdict based strictly on the provided policy context."
)
_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["COMPLIANT", "NON_COMPLIANT", "REVIEW_REQUIRED"],
        },
        "cited_section_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "section_id values from the provided context that support this verdict.",
        },
        "violated_section_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "section_id values whose rules appear to be breached, if any.",
        },
        "rationale": {
            "type": "string",
            "description": "Concise (2-4 sentence) explanation grounded only in the provided context.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": ["decision", "cited_section_ids", "violated_section_ids", "rationale", "confidence"],
}

_SYSTEM_PROMPT = """You are a credit policy compliance assistant for a commercial bank.
You must evaluate a loan application strictly against the policy excerpts provided to you.
Rules:
- Only use facts present in the provided policy context. Never invent thresholds, document
  requirements, or section numbers that are not present in the context.
- If the context does not clearly cover the applicant's situation, choose REVIEW_REQUIRED.
- Always cite the section_id(s) you relied on.
- Be concise and precise; loan officers rely on this output for regulatory sign-off."""


class ComplianceAgentError(Exception):
    pass


class ComplianceAgent:
    def __init__(self, provider: AnthropicProvider) -> None:
        self._provider = provider

    @staticmethod
    def _build_context_block(retrieved: List[RetrievedChunk]) -> str:
        parts = []
        for r in retrieved:
            m = r.chunk.metadata
            parts.append(
                f"[section_id: {m.section_id}] {m.title} (policy v{m.policy_version}, "
                f"source: {m.source_file})\n"
                f"max_ltv_ratio: {m.max_ltv_ratio}\n"
                f"max_unsecured_loan_lakhs: {m.max_unsecured_loan_lakhs}\n"
                f"min_credit_score: {m.min_credit_score}\n"
                f"required_documents: {', '.join(m.required_documents) or 'none specified'}\n"
                f"text: {r.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def get_verdict(
        self, request: ComplianceRequest, retrieved: List[RetrievedChunk]
    ) -> LLMComplianceVerdict:
        context_block = self._build_context_block(retrieved)
        user_prompt = f"""POLICY CONTEXT:
{context_block}

LOAN APPLICATION:
- Borrower type: {request.borrower_type.value}
- Requested loan amount: {request.loan_amount_lakhs} INR Lakhs
- Collateral offered: {request.collateral_type}
- Collateral value: {request.collateral_value_lakhs if request.collateral_value_lakhs is not None else 'not provided'} INR Lakhs

Evaluate this application against the policy context above and call the
{_TOOL_NAME} tool with your structured verdict."""

        try:
            result = self._provider.generate_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tool_name=_TOOL_NAME,
                tool_description=_TOOL_DESCRIPTION,
                tool_schema=_TOOL_SCHEMA,
            )
        except LLMProviderError as exc:
            raise ComplianceAgentError(f"LLM provider failed: {exc}") from exc

        try:
            verdict = LLMComplianceVerdict.model_validate(result.data)
        except ValidationError as exc:
            raise ComplianceAgentError(f"LLM returned malformed structured output: {exc}") from exc

        logger.info(
            "LLM verdict: decision=%s confidence=%.2f cited=%s",
            verdict.decision.value,
            verdict.confidence,
            verdict.cited_section_ids,
        )
        return verdict