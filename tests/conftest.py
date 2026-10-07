"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_policy_markdown() -> str:
    return """# Sample Lending Policy

Policy Version: 1.0
Effective Date: 2025-01-01

## Loan-to-Value Limits for Secured Sample Loans

```yaml
section_id: SAMPLE-1.1
title: Loan-to-Value Limits for Secured Sample Loans
borrower_type: MSME
max_ltv_ratio: 0.75
max_unsecured_loan_lakhs: null
min_credit_score: 650
required_documents:
  - Sample document A
```

This is the narrative text describing the LTV rule for secured sample loans.

## Unsecured Sample Lending Cap

```yaml
section_id: SAMPLE-1.2
title: Unsecured Sample Lending Cap
borrower_type: MSME
max_ltv_ratio: null
max_unsecured_loan_lakhs: 10
min_credit_score: 700
required_documents:
  - Sample document B
```

This is the narrative text describing the unsecured cap rule.
"""


@pytest.fixture
def sample_policy_file(tmp_path: Path, sample_policy_markdown: str) -> Path:
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    file_path = policy_dir / "sample_policy.md"
    file_path.write_text(sample_policy_markdown, encoding="utf-8")
    return file_path