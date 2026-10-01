"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Bank Credit Policy Compliance Agent",
    description=(
        "RAG-based compliance engine that indexes bank credit policy documents and evaluates "
        "loan applications against them, with a deterministic rule cross-check for numeric thresholds."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Credit Policy Compliance Agent starting up.")