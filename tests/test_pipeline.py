import pytest
import json
from decimal import Decimal
from src.pipeline import run_claim_pipeline
from src.models import ClaimInput, ClaimResponse

def load_test_cases() -> list:
    from pathlib import Path
    path = Path(__file__).parent.parent / "Docs" / "test_cases.json"
    with open(path, "r") as f:
        return json.load(f)["test_cases"]

@pytest.mark.parametrize("case_data", load_test_cases())
def test_claim_pipeline(case_data: dict, request: pytest.FixtureRequest) -> None:
    case_id = case_data["case_id"]
    expected = case_data["expected"]
    claim_input = ClaimInput(**case_data["input"])

    result: ClaimResponse = run_claim_pipeline(claim_input, use_mock=True)
    request.node.pipeline_result = result

    if expected.get("decision") is None:
        # TC001–TC003: expect early stop with specific message
        assert result.outcome_type == "EARLY_STOP", f"{case_id}: expected EARLY_STOP"
        assert result.early_stop is not None
        msg = result.early_stop.user_message
        assert msg, f"{case_id}: user_message must not be empty"
        assert len(msg) > 40, f"{case_id}: message too generic"
        
        # TC001 checks for document type keywords
        if case_id == "TC001":
            assert "prescription" in msg.lower()
            assert "hospital bill" in msg.lower() or "hospital_bill" in msg.lower()
            assert result.early_stop.stop_stage == "DOCUMENT_SET_VALIDATION"
            
        # TC002 verifies stop_stage == "QUALITY_CHECK" and non-empty unreadable list
        if case_id == "TC002":
            assert "re-upload" in msg.lower() or "resubmit" in msg.lower()
            assert result.early_stop.stop_stage == "QUALITY_CHECK"
            assert len(result.early_stop.unreadable_documents) > 0
            
        # TC003 asserts both patient names appear in error message
        if case_id == "TC003":
            assert "rajesh" in msg.lower()
            assert "arjun" in msg.lower()
            assert result.early_stop.stop_stage == "CONSISTENCY_CHECK"
            
    else:
        assert result.outcome_type == "DECISION", f"{case_id}: expected DECISION"
        dec = result.decision
        assert dec is not None
        assert dec.decision == expected["decision"],             f"{case_id}: got {dec.decision}, expected {expected['decision']}"

        if "approved_amount" in expected:
            assert dec.approved_amount == Decimal(str(expected["approved_amount"])),                 f"{case_id}: amount mismatch — got {dec.approved_amount}"

        if "confidence_score" in expected:
            threshold_str = expected["confidence_score"]  # e.g. "above 0.85"
            threshold = float(threshold_str.split()[-1])
            assert dec.confidence_score > threshold,                 f"{case_id}: confidence {dec.confidence_score} not {threshold_str}"

        # TC011 specific
        if case_id == "TC011":
            assert dec.confidence_score < 0.90, "TC011: confidence should be reduced"
            assert len(dec.pipeline_warnings) > 0, "TC011: must have pipeline warnings"
            assert any("manual review" in w.lower() for w in dec.pipeline_warnings)

        # TC009 specific
        if case_id == "TC009":
            assert dec.decision == "MANUAL_REVIEW"
            assert len(dec.audit_trace) > 0
            fraud_steps = [s for s in dec.audit_trace if s.rule_applied == "FRAUD_SAME_DAY_CLAIMS_EXCEEDED"]
            assert len(fraud_steps) > 0, "TC009: fraud signal must appear in trace"

        # TC006 specific
        if case_id == "TC006":
            assert dec.decision == "PARTIAL"
            excluded = [li for li in dec.line_item_decisions if li.status == "EXCLUDED"]
            assert len(excluded) == 1
            assert excluded[0].rejection_reason == "COSMETIC_DENTAL_EXCLUSION"

        # Confidence reconciliation check (all cases with DECISION outcome)
        delta_sum = sum(entry.confidence_delta for entry in dec.audit_trace)
        assert abs(delta_sum - (1.0 - dec.confidence_score)) < 0.001