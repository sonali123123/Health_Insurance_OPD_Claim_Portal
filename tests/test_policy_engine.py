import json
import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Any, Dict, List

from src.policy_engine import adjudicate_claim, AdjudicationValidationError
from src.models import (
    ClaimInput,
    FactExtractionPayload,
    ItemizedLine,
    DocumentType,
    RejectionReason,
)
from src.utils import normalize_diagnosis

def load_policy() -> Dict[str, Any]:
    from pathlib import Path
    path = Path(__file__).parent.parent / "Docs" / "policy_terms.json"
    with open(path, "r") as f:
        return json.load(f)

def load_test_cases() -> List[Dict[str, Any]]:
    from pathlib import Path
    path = Path(__file__).parent.parent / "Docs" / "test_cases.json"
    with open(path, "r") as f:
        return json.load(f)["test_cases"]

def make_extraction(doc: Dict[str, Any]) -> FactExtractionPayload:
    content = doc.get("content") or {}
    line_items = None
    if "line_items" in content:
        line_items = [
            ItemizedLine(
                description=item["description"],
                amount=Decimal(str(item["amount"])),
                status="APPROVED"
            )
            for item in content["line_items"]
        ]
        
    bill_date = None
    if "date" in content:
        bill_date = datetime.strptime(content["date"], "%Y-%m-%d").date()
        
    return FactExtractionPayload(
        file_id=doc["file_id"],
        document_type=DocumentType(doc["actual_type"]),
        doctor_name=content.get("doctor_name"),
        doctor_registration=content.get("doctor_registration"),
        diagnosis=content.get("diagnosis"),
        diagnosis_normalized=normalize_diagnosis(content.get("diagnosis")),
        medicines=content.get("medicines"),
        tests_ordered=content.get("tests_ordered"),
        hospital_name=content.get("hospital_name"),
        patient_name=content.get("patient_name"),
        bill_date=bill_date,
        line_items=line_items,
        bill_total=Decimal(str(content["total"])) if "total" in content else None
    )

@pytest.mark.parametrize("case_data", [
    c for c in load_test_cases() if c["expected"].get("decision") is not None
])
def test_policy_engine_adjudication(case_data: Dict[str, Any]) -> None:
    case_id = case_data["case_id"]
    expected = case_data["expected"]
    claim_input = ClaimInput(**case_data["input"])
    policy = load_policy()
    
    # Map raw input documents to FactExtractionPayload list
    extractions = [make_extraction(doc) for doc in case_data["input"].get("documents", [])]
    
    # Run adjudicate_claim directly
    output = adjudicate_claim(claim_input, policy, extractions)
    
    # Assert decision matches
    assert output.decision == expected["decision"], f"{case_id}: expected {expected['decision']}, got {output.decision}"
    
    # Assert approved amount matches if expected
    if "approved_amount" in expected:
        expected_amt = Decimal(str(expected["approved_amount"]))
        assert output.approved_amount == expected_amt, f"{case_id}: expected approved_amount {expected_amt}, got {output.approved_amount}"
        
    # Assert rejection reasons match
    if "rejection_reasons" in expected:
        expected_reasons = [RejectionReason(r) for r in expected["rejection_reasons"]]
        for r in expected_reasons:
            assert r in output.rejection_reasons, f"{case_id}: expected rejection reason {r} in {output.rejection_reasons}"

    # Assert specific case invariants
    if case_id == "TC011":
        assert output.confidence_score == 0.75, "TC011: confidence score should be degraded to 0.75"
        assert any("manual review" in w.lower() for w in output.pipeline_warnings), "TC011: warnings should mention manual review"
        
    if case_id == "TC009":
        assert output.decision == "MANUAL_REVIEW", "TC009: decision should be MANUAL_REVIEW"
        assert any("FRAUD" in (s.rule_applied or "") for s in output.audit_trace), "TC009: audit trace should include FRAUD check"

def test_negative_approved_amount_validation() -> None:
    # Test that AdjudicationValidationError is raised if final approved amount is negative
    claim_input = ClaimInput(
        member_id="EMP001",
        policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION",
        treatment_date=date(2024, 11, 1),
        claimed_amount=Decimal("1500"),
        documents=[]
    )
    policy = load_policy()
    
    # Artificially set a negative copay percent or construct input that causes negative payout
    policy_copy = json.loads(json.dumps(policy))
    policy_copy["opd_categories"]["consultation"]["copay_percent"] = 150  # This makes payout negative
    
    extractions: List[FactExtractionPayload] = []
    
    with pytest.raises(AdjudicationValidationError):
        adjudicate_claim(claim_input, policy_copy, extractions)