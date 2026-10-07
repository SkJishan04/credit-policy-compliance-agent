"""Integration tests for the FastAPI compliance API using dependency overrides."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.dependencies import get_compliance_service, get_vector_store
from app.api.main import app
from app.models.schemas import (
    CitedSection,
    ComplianceDecision,
    ComplianceResponse,
    RuleCheckResult,
)


class FakeComplianceService:
    def evaluate(self, request):
        return ComplianceResponse(
            decision=ComplianceDecision.COMPLIANT,
            verdict_summary="Test verdict summary.",
            cited_sections=[CitedSection(section_id="MSME-4.1", title="Test Section", excerpt="Test excerpt.")],
            rule_check=RuleCheckResult(
                computed_ltv=0.5,
                max_allowed_ltv=0.75,
                ltv_compliant=True,
                loan_amount_lakhs=request.loan_amount_lakhs,
            ),
            confidence=0.9,
            latency_ms=42.0,
            warnings=[],
        )


class FakeVectorStore:
    def count(self) -> int:
        return 8


def _client() -> TestClient:
    app.dependency_overrides[get_compliance_service] = lambda: FakeComplianceService()
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStore()
    return TestClient(app)


def test_health_endpoint_returns_indexed_chunk_count():
    client = _client()
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "indexed_chunks": 8}


def test_evaluate_endpoint_returns_compliance_decision():
    client = _client()
    payload = {
        "borrower_type": "MSME",
        "loan_amount_lakhs": 50,
        "collateral_type": "Commercial property",
        "collateral_value_lakhs": 100,
    }
    response = client.post("/api/v1/compliance/evaluate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "COMPLIANT"
    assert body["cited_sections"][0]["section_id"] == "MSME-4.1"
    assert body["rule_check"]["ltv_compliant"] is True


def test_evaluate_endpoint_rejects_invalid_loan_amount():
    client = _client()
    payload = {
        "borrower_type": "MSME",
        "loan_amount_lakhs": -5,
        "collateral_type": "None",
    }
    response = client.post("/api/v1/compliance/evaluate", json=payload)
    assert response.status_code == 422