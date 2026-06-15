import os
import pytest
from decimal import Decimal
from typing import List, Dict, Any
from src.pipeline import run_claim_pipeline
from src.models import ClaimInput

results_cache: List[Dict[str, Any]] = []

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    outcome = yield
    rep = outcome.get_result()
    
    # Only capture on the call phase of test_claim_pipeline
    if rep.when == "call" and hasattr(item, "callspec") and "case_data" in item.callspec.params:
        if "test_claim_pipeline" not in item.name:
            return
        case_data = item.callspec.params["case_data"]
        case_id = case_data["case_id"]
        case_name = case_data["case_name"]
        scenario = case_data.get("description", "")
        
        # Get expected
        expected = case_data["expected"]
        exp_decision = expected.get("decision")
        exp_payout = expected.get("approved_amount")
        
        # Format expected outcome string and setup payout
        if exp_payout is None:
            expected_outcome_str = f"{exp_decision}" if exp_decision is not None else "EARLY_STOP"
        else:
            expected_payout = Decimal(str(exp_payout))
            expected_outcome_str = f"{exp_decision} (₹{expected_payout})"
            
        # Run actual pipeline to get actual outcome
        try:
            claim_input = ClaimInput(**case_data["input"])
            result = run_claim_pipeline(claim_input, use_mock=True)
            
            if result.outcome_type == "EARLY_STOP":
                actual_payout = Decimal("0")
                actual_outcome_str = "EARLY_STOP"
            else:
                dec = result.decision
                assert dec is not None
                actual_decision = dec.decision
                actual_payout = dec.approved_amount
                actual_outcome_str = f"{actual_decision}"
                if actual_decision in ["APPROVED", "PARTIAL"]:
                    actual_outcome_str += f" (₹{actual_payout})"
                
            if exp_payout is None:
                variance = Decimal("0")
            else:
                variance = actual_payout - expected_payout
                
            assertion_result = "Pass" if rep.outcome == "passed" else "Fail"
        except Exception as e:
            actual_outcome_str = f"ERROR: {str(e)}"
            variance = Decimal("0")
            assertion_result = "Fail"
            
        results_cache.append({
            "case_id": case_id,
            "name": case_name,
            "scenario": scenario,
            "expected_outcome": expected_outcome_str,
            "actual_outcome": actual_outcome_str,
            "variance": variance,
            "assertion_result": assertion_result
        })

def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    # Ensure Docs/ directory exists
    os.makedirs("Docs", exist_ok=True)
    
    # Sort results_cache by case_id to ensure they are in order
    results_cache.sort(key=lambda x: x["case_id"])
    
    report_content = "# Claims Adjudication Evaluation Report\n\n"
    report_content += "| Case ID & Name | Scenario Description | Expected Outcome (Decision & Payout) | Actual Outcome (Decision & Payout) | Financial Variance (Target: ₹0) | Assertion Result (Pass / Fail) |\n"
    report_content += "|---|---|---|---|---|---|\n"
    
    for r in results_cache:
        status_icon = "✅ Pass" if r["assertion_result"] == "Pass" else "❌ Fail"
        variance_str = f"₹{r['variance']}"
        report_content += f"| {r['case_id']}: {r['name']} | {r['scenario']} | {r['expected_outcome']} | {r['actual_outcome']} | {variance_str} | {status_icon} |\n"
        
    with open("Docs/EVAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_content)
