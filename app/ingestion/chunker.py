"""Parses policy markdown files into structured, retrievable PolicyChunk objects.

Each policy file is a Markdown document where every '## ' section is optionally
preceded/followed by a fenced ```yaml block containing structured metadata
(section_id, borrower_type, numeric thresholds, required documents). This lets
downstream components (retrieval filters, the rule-based compliance cross-check)
work off exact structured facts rather than relying on the LLM to extract numbers
from prose, which is a key hallucination-mitigation strategy in this system.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

from app.models.schemas import PolicyChunk, PolicyChunkMetadata

_SECTION_SPLIT_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_YAML_BLOCK_RE = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)
_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_VERSION_RE = re.compile(r"^Policy Version:\s*(.+)$", re.MULTILINE)


class ChunkerError(Exception):
    """Raised when a policy document cannot be parsed into valid chunks."""


def _extract_policy_version(text: str) -> str:
    match = _VERSION_RE.search(text)
    return match.group(1).strip() if match else "unknown"


def parse_policy_file(path: Path) -> List[PolicyChunk]:
    """Parse a single policy markdown file into a list of PolicyChunk objects."""
    raw = path.read_text(encoding="utf-8")
    policy_version = _extract_policy_version(raw)

    # Split the document on '## ' section headings, keeping the heading text.
    parts = _SECTION_SPLIT_RE.split(raw)
    # parts[0] is preamble before the first '## '; subsequent items alternate
    # [heading, body, heading, body, ...]
    chunks: List[PolicyChunk] = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""

        yaml_match = _YAML_BLOCK_RE.search(body)
        if not yaml_match:
            raise ChunkerError(
                f"Section '{heading}' in {path.name} is missing a required ```yaml metadata block."
            )

        metadata_raw = yaml.safe_load(yaml_match.group(1)) or {}
        metadata_raw.setdefault("title", heading)
        metadata_raw["policy_version"] = policy_version
        metadata_raw["source_file"] = path.name
        metadata_raw.setdefault("required_documents", [])

        try:
            metadata = PolicyChunkMetadata(**metadata_raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ChunkerError(f"Invalid metadata in section '{heading}' of {path.name}: {exc}") from exc

        narrative = _YAML_BLOCK_RE.sub("", body).strip()
        if not narrative:
            raise ChunkerError(f"Section '{heading}' in {path.name} has no narrative text.")

        chunk_id = f"{metadata.section_id}"
        chunks.append(PolicyChunk(id=chunk_id, text=narrative, metadata=metadata))

    if not chunks:
        raise ChunkerError(f"No sections found in {path.name}. Expected at least one '## ' section.")

    return chunks


def load_and_chunk_directory(directory: Path) -> List[PolicyChunk]:
    """Parse every .md policy file in a directory into a flat list of PolicyChunk objects."""
    if not directory.exists():
        raise ChunkerError(f"Policy data directory does not exist: {directory}")

    all_chunks: List[PolicyChunk] = []
    for path in sorted(directory.glob("*.md")):
        all_chunks.extend(parse_policy_file(path))

    if not all_chunks:
        raise ChunkerError(f"No policy markdown files found in {directory}")

    return all_chunks