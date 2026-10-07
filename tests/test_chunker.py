"""Unit tests for the policy markdown chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import ChunkerError, load_and_chunk_directory, parse_policy_file


def test_parse_policy_file_extracts_expected_number_of_chunks(sample_policy_file: Path):
    chunks = parse_policy_file(sample_policy_file)
    assert len(chunks) == 2


def test_parse_policy_file_extracts_correct_metadata(sample_policy_file: Path):
    chunks = parse_policy_file(sample_policy_file)
    first = next(c for c in chunks if c.metadata.section_id == "SAMPLE-1.1")

    assert first.metadata.borrower_type.value == "MSME"
    assert first.metadata.max_ltv_ratio == 0.75
    assert first.metadata.max_unsecured_loan_lakhs is None
    assert first.metadata.min_credit_score == 650
    assert first.metadata.required_documents == ["Sample document A"]
    assert first.metadata.policy_version == "1.0"
    assert "LTV rule" in first.text


def test_parse_policy_file_raises_on_missing_yaml_block(tmp_path: Path):
    bad_file = tmp_path / "bad_policy.md"
    bad_file.write_text("# Bad Policy\n\n## A Section\n\nJust text, no metadata block.\n", encoding="utf-8")

    with pytest.raises(ChunkerError):
        parse_policy_file(bad_file)


def test_load_and_chunk_directory_aggregates_all_files(sample_policy_file: Path):
    chunks = load_and_chunk_directory(sample_policy_file.parent)
    section_ids = {c.metadata.section_id for c in chunks}
    assert section_ids == {"SAMPLE-1.1", "SAMPLE-1.2"}


def test_load_and_chunk_directory_raises_on_missing_directory(tmp_path: Path):
    with pytest.raises(ChunkerError):
        load_and_chunk_directory(tmp_path / "does_not_exist")