"""API routes for the credit policy compliance service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_compliance_service, get_vector_store
from app.logging_config import get_logger
from app.models.schemas import ComplianceRequest, ComplianceResponse
from app.services.compliance_service import ComplianceService

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", tags=["system"])
def health_check() -> dict:
    try:
        indexed = get_vector_store().count()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc
    return {"status": "ok", "indexed_chunks": indexed}


@router.post("/compliance/evaluate", response_model=ComplianceResponse, tags=["compliance"])
def evaluate_compliance(
    request: ComplianceRequest,
    service: ComplianceService = Depends(get_compliance_service),
) -> ComplianceResponse:
    try:
        return service.evaluate(request)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unhandled error during compliance evaluation")
        raise HTTPException(status_code=500, detail="Internal error during compliance evaluation.") from exc