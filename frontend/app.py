"""Gradio frontend for loan officers -- calls the FastAPI compliance backend."""

from __future__ import annotations

import httpx
import gradio as gr

from app.config import get_settings

settings = get_settings()
BACKEND_URL = settings.backend_url


def evaluate_credit_compliance(
    borrower_type: str,
    loan_amount: float,
    collateral_type: str,
    collateral_value: float,
) -> str:
    """Calls the compliance API and formats the result for the loan officer."""
    payload = {
        "borrower_type": borrower_type,
        "loan_amount_lakhs": loan_amount,
        "collateral_type": collateral_type or "None",
        "collateral_value_lakhs": collateral_value if collateral_value and collateral_value > 0 else None,
    }

    try:
        response = httpx.post(f"{BACKEND_URL}/api/v1/compliance/evaluate", json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        return f"ERROR: Backend returned {exc.response.status_code}: {exc.response.text}"
    except httpx.RequestError as exc:
        return f"ERROR: Could not reach compliance backend at {BACKEND_URL} ({exc}). Is the API running?"

    lines = [f"VERDICT: {data['decision']}", "", data["verdict_summary"], ""]

    if data["cited_sections"]:
        lines.append("Cited policy sections:")
        for section in data["cited_sections"]:
            lines.append(f"  - [{section['section_id']}] {section['title']}")
        lines.append("")

    rc = data["rule_check"]
    if rc.get("computed_ltv") is not None:
        lines.append(
            f"Deterministic LTV check: {rc['computed_ltv']:.1%} computed vs "
            f"{rc['max_allowed_ltv']:.1%} allowed -> "
            f"{'PASS' if rc['ltv_compliant'] else 'FAIL'}"
        )
    if rc.get("max_unsecured_loan_lakhs") is not None:
        lines.append(
            f"Deterministic unsecured cap check: {rc['loan_amount_lakhs']} lakhs vs "
            f"{rc['max_unsecured_loan_lakhs']} lakh cap -> "
            f"{'PASS' if rc['unsecured_compliant'] else 'FAIL'}"
        )

    if data["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in data["warnings"]:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append(f"Confidence: {data['confidence']:.0%}  |  Latency: {data['latency_ms']:.0f} ms")

    return "\n".join(lines)


demo = gr.Interface(
    fn=evaluate_credit_compliance,
    inputs=[
        gr.Dropdown(["MSME", "Retail Individual", "Corporate"], label="Borrower Type"),
        gr.Number(label="Loan Amount (in ₹ Lakhs)"),
        gr.Textbox(label="Collateral Offered", placeholder="e.g. Commercial property, Fixed deposit, None"),
        gr.Number(
            label="Collateral Value (in ₹ Lakhs, optional)",
            info="Required to compute the deterministic Loan-to-Value check. Leave 0 for unsecured loans.",
        ),
    ],
    outputs=gr.Textbox(label="Compliance Verdict & Citation", lines=14),
    title="Automated Bank Credit Policy Compliance Agent",
    description=(
        "RAG-based compliance engine indexing internal credit policy documents. "
        "Retrieves the exact regulatory criteria for the borrower type and cross-checks "
        "numeric thresholds deterministically to reduce LLM hallucination risk."
    ),
)

if __name__ == "__main__":
    demo.launch()