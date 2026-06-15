"""
OPD Claims Adjudication Portal — Streamlit front-end
======================================================
Tab 1: Mock Test Cases  — pre-loaded JSON fixtures, zero API calls
Tab 2: Live AI Upload   — real file uploads → gemini-3.1-flash-lite pipeline
"""

import json
import os
import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from src.models import (
    ClaimInput,
    ClaimsHistoryEntry,
    DocumentUpload,
    EarlyStopResponse,
    RejectionReason,
    ClaimResponse
)
from src.pipeline import run_claim_pipeline, compiled_graph

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OPD Claims Adjudication Portal",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS Styling Injection
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Custom Fonts & Page Layout */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1A1F3C 0%, #2A3158 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #A0AEC0;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    
    /* Card Design */
    .custom-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Timeline Styling */
    .timeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #F8FAFC;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
    }
    .timeline-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        text-align: center;
        position: relative;
        z-index: 2;
    }
    .node-circle {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .node-circle.pending { background-color: #E2E8F0; color: #64748B; border: 2px solid #CBD5E0; }
    .node-circle.passed { background-color: #DEF7EC; color: #03543F; border: 2px solid #31C48D; }
    .node-circle.failed { background-color: #FDE8E8; color: #9B1C1C; border: 2px solid #F05252; }
    .node-circle.warning { background-color: #FEF08A; color: #713F12; border: 2px solid #EAB308; }
    .node-circle.degraded { background-color: #FEF08A; color: #713F12; border: 2px solid #EAB308; }
    .node-label {
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .timeline-arrow {
        height: 4px;
        background-color: #E2E8F0;
        flex-grow: 1;
        margin: 0 -10px;
        margin-top: -24px;
        position: relative;
        z-index: 1;
    }
    .timeline-arrow.passed { background-color: #31C48D; }
    
    /* Financial Flow */
    .financial-flow-container {
        display: flex;
        align-items: center;
        justify-content: space-around;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .fin-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
        min-width: 140px;
    }
    .fin-box.gross { border-top: 4px solid #64748B; }
    .fin-box.discount { border-top: 4px solid #6366F1; }
    .fin-box.copay { border-top: 4px solid #F59E0B; }
    .fin-box.payout { border-top: 4px solid #10B981; background: #ECFDF5; }
    .fin-label { font-size: 10px; text-transform: uppercase; color: #64748B; font-weight: 600; letter-spacing: 0.5px; }
    .fin-value { font-size: 18px; font-weight: 700; color: #1E293B; margin-top: 6px; }
    .fin-arrow { font-size: 24px; color: #94A3B8; font-weight: 300; }
    
    /* Button Hover Effects */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — cached loaders
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

@st.cache_data
def load_policy_terms() -> Dict[str, Any]:
    path = Path(__file__).parent / "Docs" / "policy_terms.json"
    with open(path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


@st.cache_data
def load_test_cases() -> List[Dict[str, Any]]:
    path = Path(__file__).parent / "Docs" / "test_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        return cast(List[Dict[str, Any]], json.load(f)["test_cases"])


policy = load_policy_terms()
test_cases_data = load_test_cases()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🏥 Group Health Insurance OPD Claims Portal</h1>
        <p class="header-subtitle">
            Automated Health Insurance Claims Adjudication Portal &nbsp;·&nbsp; PLUM_GHI_2024
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — policy view
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Policy Summary")
    coverage = policy.get("coverage", {})
    st.markdown(f"**Plan:** `{policy.get('plan_id', 'N/A')}`")
    st.markdown(f"**Annual OPD Limit:** ₹{int(coverage.get('annual_opd_limit', 0)):,}")
    st.markdown(f"**Per-Claim Limit:** ₹{int(coverage.get('per_claim_limit', 0)):,}")
    st.markdown(f"**Status:** `{policy.get('policy_holder', {}).get('renewal_status', 'N/A')}`")
    with st.expander("View full policy_terms.json"):
        st.json(policy)
    st.divider()
    
    with st.expander("⚙️ System Diagnostics", expanded=False):
        google_key_set = bool(os.environ.get("GOOGLE_API_KEY"))
        groq_key_set = bool(os.environ.get("GROQ_API_KEY"))
        
        if google_key_set:
            st.success("✓ Google API Key loaded.")
        else:
            st.warning("⚠️ Google API Key missing in environment.")
            
        if groq_key_set:
            st.success("✓ Groq API Key loaded.")
        else:
            st.info("ℹ Groq API Key missing (optional).")

# ─────────────────────────────────────────────────────────────────────────────
# Custom Pipeline Stage Tracking Helper
# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline_stages_status(trace_steps: List[Any], outcome_type: str, early_stop: Any = None) -> List[Dict[str, Any]]:
    stages = [
        {"id": "classification", "label": "Document Audit", "status": "PENDING"},
        {"id": "quality", "label": "Clarity Verification", "status": "PENDING"},
        {"id": "extractor", "label": "Fact Extraction", "status": "PENDING"},
        {"id": "consistency", "label": "Name Matching", "status": "PENDING"},
        {"id": "policy", "label": "Policy Adjudication", "status": "PENDING"},
        {"id": "fraud", "label": "Risk Assessment", "status": "PENDING"},
    ]
    
    if not trace_steps:
        # Check early stop triggers if no trace steps exist
        if early_stop:
            if early_stop.stop_stage == "DOCUMENT_SET_VALIDATION":
                stages[0]["status"] = "FAILED"
            elif early_stop.stop_stage == "QUALITY_CHECK":
                stages[0]["status"] = "PASSED"
                stages[1]["status"] = "FAILED"
        return stages
        
    # Analyze trace steps for mapping
    # 1. Classification
    class_steps = [s for s in trace_steps if "doc_class_" in s.step_id or s.node == "document_classifier"]
    if class_steps:
        if any(s.result == "FAILED" for s in class_steps):
            stages[0]["status"] = "FAILED"
        else:
            stages[0]["status"] = "PASSED"
    else:
        if early_stop and early_stop.stop_stage == "DOCUMENT_SET_VALIDATION":
            stages[0]["status"] = "FAILED"
        elif len(trace_steps) > 0:
            stages[0]["status"] = "PASSED"
            
    # 2. Quality
    qual_steps = [s for s in trace_steps if "t_valset_" in s.step_id or s.node == "quality_verifier" or "quality" in s.node]
    if qual_steps:
        if any(s.result == "FAILED" for s in qual_steps):
            stages[1]["status"] = "FAILED"
        elif any(s.result == "DEGRADED" for s in qual_steps):
            stages[1]["status"] = "DEGRADED"
        else:
            stages[1]["status"] = "PASSED"
    else:
        if early_stop and early_stop.stop_stage == "QUALITY_CHECK":
            stages[1]["status"] = "FAILED"
        elif len(trace_steps) > 1:
            stages[1]["status"] = "PASSED"
            
    # 3. Extraction
    ext_steps = [s for s in trace_steps if "t_extract_" in s.step_id or s.node == "fact_extractor"]
    if ext_steps:
        if any(s.result == "FAILED" for s in ext_steps):
            stages[2]["status"] = "FAILED"
        elif any(s.result == "DEGRADED" for s in ext_steps):
            stages[2]["status"] = "DEGRADED"
        else:
            stages[2]["status"] = "PASSED"
    else:
        if len(trace_steps) > 2:
            stages[2]["status"] = "PASSED"
            
    # 4. Consistency
    const_steps = [s for s in trace_steps if "t_consistency_" in s.step_id or s.node == "consistency_checker"]
    if const_steps:
        if any(s.result == "FAILED" for s in const_steps):
            stages[3]["status"] = "FAILED"
        elif any(s.result == "DEGRADED" for s in const_steps):
            stages[3]["status"] = "DEGRADED"
        elif any(s.result == "FLAGGED" for s in const_steps):
            stages[3]["status"] = "WARNING"
        else:
            stages[3]["status"] = "PASSED"
    else:
        if early_stop and early_stop.stop_stage == "CONSISTENCY_CHECK":
            stages[3]["status"] = "FAILED"
        elif len(trace_steps) > 3:
            stages[3]["status"] = "PASSED"
            
    # 5. Policy Adjudication
    policy_steps = [s for s in trace_steps if "adj_" in s.step_id or s.node == "policy_engine"]
    if policy_steps:
        if any(s.result == "FAILED" for s in policy_steps):
            stages[4]["status"] = "FAILED"
        else:
            stages[4]["status"] = "PASSED"
    else:
        if len(trace_steps) > 4:
            stages[4]["status"] = "PASSED"
            
    # 6. Fraud & Risk
    fraud_steps = [s for s in trace_steps if "t_fraud_" in s.step_id or s.node == "fraud_agent"]
    if fraud_steps:
        if any(s.result == "FAILED" for s in fraud_steps):
            stages[5]["status"] = "FAILED"
        elif any(s.result == "FLAGGED" for s in fraud_steps):
            stages[5]["status"] = "WARNING"
        else:
            stages[5]["status"] = "PASSED"
            
    # Propagate failure halts
    halt_found = False
    for stage in stages:
        if halt_found:
            stage["status"] = "PENDING"
        if stage["status"] == "FAILED":
            halt_found = True
            
    return stages


def draw_timeline_html(stages: List[Dict[str, Any]]) -> None:
    html = '<div class="timeline-wrapper">'
    for i, stage in enumerate(stages):
        status = stage["status"].lower()
        circle_content = "✓" if status == "passed" else ("✗" if status == "failed" else ("!" if status in ("warning", "degraded") else str(i + 1)))
        
        html += f"""
        <div class="timeline-node">
            <div class="node-circle {status}">{circle_content}</div>
            <div class="node-label">{stage['label']}</div>
        </div>
        """
        if i < len(stages) - 1:
            arrow_status = "passed" if status == "passed" else ""
            html += f'<div class="timeline-arrow {arrow_status}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def draw_financial_flow_html(gross: Decimal, discount: Decimal, copay: Decimal, payout: Decimal) -> None:
    html = f"""
    <div class="financial-flow-container">
        <div class="fin-box gross">
            <div class="fin-label">Gross Claimed</div>
            <div class="fin-value">₹{gross:,.2f}</div>
        </div>
        <div class="fin-arrow">➔</div>
        <div class="fin-box discount">
            <div class="fin-label">Network Discount</div>
            <div class="fin-value">-₹{discount:,.2f}</div>
        </div>
        <div class="fin-arrow">➔</div>
        <div class="fin-box copay">
            <div class="fin-label">Co-payment</div>
            <div class="fin-value">-₹{copay:,.2f}</div>
        </div>
        <div class="fin-arrow">➔</div>
        <div class="fin-box payout">
            <div class="fin-label">Approved Payout</div>
            <div class="fin-value">₹{payout:,.2f}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_business_audit_checklist(trace_steps: List[Any]) -> None:
    """Render a professional, human-readable checklist of policy rules evaluated."""
    if not trace_steps:
        st.info("No policy adjudication steps recorded.")
        return
        
    rule_mapping = {
        "POLICY_ACTIVE_CHECK": ("Policy Status Verification", "Checks if the policy is active and in-force."),
        "MEMBER_ELIGIBILITY_CHECK": ("Member Eligibility & Roster Check", "Validates that the patient is covered under the policy roster."),
        "SUBMISSION_DEADLINE_CHECK": ("Claim Submission Window Check", "Verifies the claim was submitted within the allowed window from treatment date."),
        "MINIMUM_CLAIM_AMOUNT_CHECK": ("Minimum Claim Threshold Check", "Verifies the claimed amount meets the minimum policy threshold."),
        "INITIAL_WAITING_PERIOD_CHECK": ("General Waiting Period Check", "Checks if the policy is past its general waiting period for new members."),
        "WAITING_PERIOD": ("Pre-existing Condition Waiting Period Check", "Validates if any waiting periods apply to specific diagnoses (e.g. Diabetes)."),
        "EXCLUDED_CONDITION_CHECK": ("Excluded Medical Conditions Audit", "Audits the diagnosis against policy excluded conditions."),
        "PRE_AUTH_REQUIRED_CHECK": ("Pre-Authorization Requirement Check", "Checks if high-value diagnostic tests or procedures required prior approval."),
        "PER_CLAIM_LIMIT_CHECK": ("Per-Claim Limit Verification", "Ensures the claimed amount does not exceed the single claim ceiling."),
        "ANNUAL_LIMIT_EXHAUSTED": ("Annual OPD Limit Check", "Checks if the member has remaining annual OPD benefits limit."),
        "LINE_ITEM_EXCLUSION_CHECK": ("Excluded Services & Line-Items Check", "Scans individual bill line-items for cosmetic or excluded procedures."),
        "NETWORK_HOSPITAL_DISCOUNT": ("Network Provider Discount Verification", "Checks if hospital belongs to the network and applies contract discounts."),
        "COPAY_APPLICATION": ("Co-payment Deductible Calculation", "Applies the member's co-pay percentage on the eligible amount."),
        "NAME_CONSISTENCY_CHECK": ("Patient Name Alignment Check", "Cross-checks spelling similarity between documents and roster database."),
        "FRAUD_SAME_DAY_CLAIMS_EXCEEDED": ("Claims Frequency & Risk Assessment", "Scans history for potential high-frequency claims abuse on the same day."),
        "DOCUMENT_SET_VALIDATION": ("Required Documents Audit", "Validates that all required billing and prescription files were uploaded.")
    }
    
    st.markdown('<div style="margin-top: 10px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)
    
    for step in trace_steps:
        rule = step.rule_applied or step.step_id
        
        # Determine title and description
        title, desc = rule_mapping.get(rule, (step.rule_applied or step.node.replace("_", " ").title(), "Evaluates policy rule conditions."))
        
        # Fallback names mapping for structural steps
        if "t_valset" in step.step_id:
            title, desc = "Required Documents Audit", "Validates that all required billing and prescription files were uploaded."
        elif "t_consistency" in step.step_id:
            title, desc = "Patient Name Alignment Check", "Cross-checks spelling similarity between documents and roster database."
        elif "t_fraud" in step.step_id:
            title, desc = "Claims Frequency & Risk Assessment", "Scans history for potential high-frequency claims abuse on the same day."
        elif "adj_degraded" in step.step_id:
            title, desc = "System Degradation Event", "A system component degraded; proceeding with fallback data."
            
        result = step.result
        
        # Color badges
        badges = {
            "PASSED": '<span style="background-color:#DEF7EC;color:#03543F;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">PASSED</span>',
            "FAILED": '<span style="background-color:#FDE8E8;color:#9B1C1C;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">FAILED</span>',
            "FLAGGED": '<span style="background-color:#FEF08A;color:#713F12;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">FLAGGED</span>',
            "DEGRADED": '<span style="background-color:#FEF08A;color:#713F12;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">DEGRADED</span>',
            "SKIPPED": '<span style="background-color:#E2E8F0;color:#64748B;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">SKIPPED</span>',
        }
        badge_html = badges.get(result, f'<span style="background-color:#E2E8F0;color:#64748B;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">{result}</span>')
        
        notes_html = f'<div style="font-size:12px;color:#64748B;margin-top:4px;">↳ <i>{step.notes}</i></div>' if step.notes else ""
        
        st.markdown(
            f"""
            <div style="background-color: white; padding: 12px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; font-size: 13.5px; color: #1E293B;">{title}</span>
                    {badge_html}
                </div>
                <div style="font-size: 12px; color: #64748B; margin-top: 2px;">{desc}</div>
                {notes_html}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_explainability_trace(trace_steps: List[Any]) -> None:
    """Render audit trace as a color-coded expandable table for developer views."""
    if not trace_steps:
        st.info("No trace steps recorded.")
        return
    flat_steps = []
    for step in trace_steps:
        llm_info = ""
        if step.llm_trace:
            lt = step.llm_trace
            fallback_str = " ⬆ fallback" if lt.fallback_triggered else ""
            tool_str = f" 🔧{','.join(lt.tool_calls)}" if lt.tool_calls else ""
            llm_info = f"`{lt.model_used}`{fallback_str}{tool_str}"
        flat_steps.append({
            "Step ID": step.step_id,
            "Node": step.node,
            "Rule": step.rule_applied or "—",
            "Result": step.result,
            "Input": str(step.input_value)[:120] if step.input_value else "—",
            "Output": str(step.output_value)[:120] if step.output_value else "—",
            "Conf Δ": f"{step.confidence_delta:+.2f}" if step.confidence_delta != 0 else "0.00",
            "Model": llm_info,
            "Notes": (step.notes or "")[:160],
        })

    df = pd.DataFrame(flat_steps)

    def highlight_rows(row: pd.Series) -> List[str]:
        val = row["Result"]
        colors: Dict[str, str] = {
            "FAILED":  "background-color:#ffd6d6;color:#333",
            "DEGRADED":"background-color:#ffe8b3;color:#333",
            "FLAGGED": "background-color:#fffbe6;color:#333",
            "PASSED":  "background-color:#e8f8e8;color:#333",
        }
        style = colors.get(val, "")
        return [style] * len(row)

    st.dataframe(df.style.apply(highlight_rows, axis=1), use_container_width=True)


def render_ground_truth_comparison(result: Any) -> None:
    """Compare system result with ground-truth test case expectations."""
    if st.session_state.selected_case_str == "None (Manual Entry)":
        return
        
    try:
        case_id = st.session_state.selected_case_str.split(":")[0].strip()
        case_data = next(tc for tc in test_cases_data if tc["case_id"] == case_id)
        expected = case_data.get("expected", {})
        
        expected_decision = expected.get("decision")
        expected_amount = expected.get("approved_amount")
        
        actual_decision = None
        actual_amount = None
        
        if result.outcome_type == "EARLY_STOP":
            actual_decision = None
            actual_amount = None
        else:
            actual_decision = result.decision.decision
            actual_amount = float(result.decision.approved_amount)
            
        st.markdown('<div class="section-title">🎯 Ground-Truth Evaluation Comparison</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Expected Outcome (Ground Truth):**")
            if expected_decision:
                st.markdown(f"- Decision: `{expected_decision}`")
                if expected_amount is not None:
                    st.markdown(f"- Approved Payout: `₹{expected_amount:,.2f}`")
            else:
                st.markdown(f"- Decision: `EARLY_STOP` (Early document gating check)")
                st.markdown("- Rejection Reason: Policy Validation Halt")
                
        with col2:
            st.markdown(f"**Actual System Outcome:**")
            if actual_decision:
                st.markdown(f"- Decision: `{actual_decision}`")
                if actual_amount is not None:
                    st.markdown(f"- Approved Payout: `₹{actual_amount:,.2f}`")
            else:
                st.markdown(f"- Decision: `EARLY_STOP` (Adjudication halted)")
                if result.early_stop:
                    st.markdown(f"- Stage Halted: `{result.early_stop.stop_stage}`")
                
        # Comparison logic
        decisions_match = (expected_decision == actual_decision)
        amounts_match = True
        if expected_amount is not None:
            amounts_match = (actual_amount is not None and abs(float(expected_amount) - actual_amount) < 0.01)
            
        if decisions_match and amounts_match:
            st.success("✓ **Outcome Matches Expected Target:** Guaranteed ₹0 variance matching ground-truth.")
        else:
            st.warning("⚠️ **Outcome Variance Detected:** System outcome deviates from expected target.")
    except Exception as e:
        st.error(f"Could not load evaluation comparison: {e}")


def render_decision_result(result: Any) -> None:
    """Render pipeline decision and explainability details."""
    st.markdown("---")
    st.markdown('<div class="section-title">⚡ Pipeline Results</div>', unsafe_allow_html=True)
    
    # 1. Timeline progress tracker
    trace = result.early_stop.trace if result.outcome_type == "EARLY_STOP" else result.decision.audit_trace
    stages = get_pipeline_stages_status(trace, result.outcome_type, result.early_stop)
    draw_timeline_html(stages)

    if result.outcome_type == "EARLY_STOP":
        es = result.early_stop
        st.warning(f"⚠️ **Early Stop Halted Stage: `{es.stop_stage}`**")
        st.error(f"**Message to Member:**\n\n{es.user_message}")
        col1, col2 = st.columns(2)
        col1.markdown(f"**Halt Reason Code:** `{es.stop_reason.value}`")
        col2.markdown(f"**Claim ID:** `{result.claim_id}`")
        if es.unreadable_documents:
            st.markdown(f"**Unreadable Documents Detected:** {', '.join(es.unreadable_documents)}")
            
        # Refactored Audit checklist section
        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        st.write("#### 🔍 Adjudication Decision Audit Trail")
        tab_biz, tab_tech = st.tabs(["📋 Adjudication Checklist (Business View)", "🛠️ Technical Trace (Developer View)"])
        with tab_biz:
            render_business_audit_checklist(es.trace)
        with tab_tech:
            render_explainability_trace(es.trace)
        return

    dec = result.decision
    if not dec:
        st.error("Adjudication output is missing.")
        return

    # Decision banner styling
    _banner = {
        "APPROVED":      ("success", "✅ APPROVED"),
        "PARTIAL":       ("info",    "🔶 PARTIAL APPROVAL"),
        "REJECTED":      ("error",   "❌ REJECTED"),
        "MANUAL_REVIEW": ("warning", "🔍 MANUAL REVIEW"),
    }
    method, label = _banner.get(dec.decision, ("info", dec.decision))
    getattr(st, method)(f"### Decision Status: {label}")

    st.markdown(f"**Outcome Explanation:** {dec.reason}")
    if dec.rejection_reasons:
        st.markdown(f"**Rule Violations:** `{', '.join(r.value for r in dec.rejection_reasons)}`")

    # Financial Flow
    draw_financial_flow_html(dec.gross_claimed, dec.network_discount_applied, dec.copay_applied, dec.approved_amount)

    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Gross Claimed", f"₹{dec.gross_claimed:,.2f}")
    m2.metric("Network Discount", f"₹{dec.network_discount_applied:,.2f}")
    m3.metric("Co-pay", f"₹{dec.copay_applied:,.2f}")
    m4.metric("✅ Approved Payout", f"₹{dec.approved_amount:,.2f}",
              delta=f"-₹{dec.gross_claimed - dec.approved_amount:,.2f}" if dec.gross_claimed != dec.approved_amount else None,
              delta_color="inverse")
    m5.metric("Confidence Score", f"{dec.confidence_score:.2%}")

    # Line items
    if dec.line_item_decisions:
        st.write("#### Itemized Line-Item Decisions")
        rows = [
            {
                "Description": item.description,
                "Claimed (₹)": f"{item.amount:,.2f}",
                "Status": item.status,
                "Rejection Reason": item.rejection_reason or "—",
            }
            for item in dec.line_item_decisions
        ]
        st.table(pd.DataFrame(rows))

    # Pipeline warnings
    if dec.pipeline_warnings:
        st.write("#### ⚠️ Pipeline Warnings")
        for w in dec.pipeline_warnings:
            st.warning(w)

    # Split Adjudication Audit log tabs
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.write("#### 🔍 Adjudication Decision Audit Trail")
    tab_biz, tab_tech = st.tabs(["📋 Adjudication Checklist (Business View)", "🛠️ Technical Trace (Developer View)"])
    with tab_biz:
        render_business_audit_checklist(dec.audit_trace)
    with tab_tech:
        render_explainability_trace(dec.audit_trace)

# ─────────────────────────────────────────────────────────────────────────────
# Human-in-the-Loop Override Execution Logic
# ─────────────────────────────────────────────────────────────────────────────
def resume_claim_pipeline(thread_id: str, corrected_name: str) -> ClaimResponse:
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. Update the state with corrected name and clear human_override_fields
    compiled_graph.update_state(
        config,
        {"corrected_name": corrected_name, "human_override_fields": None},
        as_node="consistency_node"
    )
    
    # 2. Resume execution
    final_state = compiled_graph.invoke(None, config)
    
    # 3. Auto-resume loop
    while True:
        state_info = compiled_graph.get_state(config)
        if not state_info.next:
            break
        if final_state.get("human_override_fields") is not None:
            break
        final_state = compiled_graph.invoke(None, config)
        
    # 4. Construct final response
    claim_id = final_state.get("claim_id", "")
    if final_state.get("final_response") is not None:
        return cast(ClaimResponse, final_state["final_response"])
    elif final_state.get("early_stop") is not None:
        return ClaimResponse(
            claim_id=claim_id,
            outcome_type="EARLY_STOP",
            early_stop=final_state["early_stop"],
            decision=None
        )
    else:
        return ClaimResponse(
            claim_id=claim_id,
            outcome_type="DECISION",
            early_stop=None,
            decision=final_state.get("adjudication_output")
        )

# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "is_interrupted" not in st.session_state:
    st.session_state.is_interrupted = False
if "override_fields" not in st.session_state:
    st.session_state.override_fields = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "last_claim_input" not in st.session_state:
    st.session_state.last_claim_input = None

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_mock, tab_live = st.tabs(["🗂️ Mock Test Cases", "🤖 Live AI Upload"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Mock Test Cases
# ══════════════════════════════════════════════════════════════════════════════
with tab_mock:
    st.subheader("Pre-loaded Ground-Truth Test Cases")
    st.markdown(
        "Select a test case to pre-fill the form. All 12 cases run in **mock mode** "
        "(zero API calls, ₹0 financial variance guaranteed)."
    )

    # Case selector + callback
    case_options = ["None (Manual Entry)"] + [
        f"{tc['case_id']}: {tc['case_name']}" for tc in test_cases_data
    ]

    def on_case_change() -> None:
        sel = st.session_state.selected_case_str
        if sel == "None (Manual Entry)":
            return
        case_id = sel.split(":")[0].strip()
        case_data = next(tc for tc in test_cases_data if tc["case_id"] == case_id)
        inp = case_data["input"]
        st.session_state.m_member_id = inp.get("member_id", "")
        st.session_state.m_policy_id = inp.get("policy_id", "")
        st.session_state.m_claim_category = inp.get("claim_category", "CONSULTATION")
        t_date_str = inp.get("treatment_date", "2024-11-01")
        st.session_state.m_treatment_date = datetime.strptime(t_date_str, "%Y-%m-%d").date()
        st.session_state.m_claimed_amount = float(inp.get("claimed_amount", 0))
        st.session_state.m_hospital_name = inp.get("hospital_name", "")
        st.session_state.m_pre_auth_reference = inp.get("pre_auth_reference", "")
        st.session_state.m_ytd_claims_amount = float(inp.get("ytd_claims_amount", 0)) if "ytd_claims_amount" in inp else 0.0
        st.session_state.m_claims_history_str = json.dumps(inp.get("claims_history", []), indent=2)
        st.session_state.m_simulate = inp.get("simulate_component_failure", False)
        st.session_state.m_documents_str = json.dumps(inp.get("documents", []), indent=2)

    # Initialize session state
    for k, v in {
        "selected_case_str": "None (Manual Entry)",
        "m_member_id": "EMP001",
        "m_policy_id": "PLUM_GHI_2024",
        "m_claim_category": "CONSULTATION",
        "m_treatment_date": date(2024, 11, 1),
        "m_claimed_amount": 1500.0,
        "m_hospital_name": "",
        "m_pre_auth_reference": "",
        "m_ytd_claims_amount": 0.0,
        "m_claims_history_str": "[]",
        "m_simulate": False,
        "m_documents_str": "[]",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.selectbox(
        "Pre-load Test Case",
        case_options,
        key="selected_case_str",
        on_change=on_case_change,
    )

    if st.session_state.selected_case_str != "None (Manual Entry)":
        case_id = st.session_state.selected_case_str.split(":")[0].strip()
        case_data = next(tc for tc in test_cases_data if tc["case_id"] == case_id)
        st.info(f"💡 **Scenario Description:** {case_data['description']}")

    categories_list = ["CONSULTATION", "DIAGNOSTIC", "PHARMACY", "DENTAL", "VISION", "ALTERNATIVE_MEDICINE"]

    with st.form("mock_claim_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            member_id  = st.text_input("Member ID",  value=st.session_state.m_member_id)
            policy_id  = st.text_input("Policy ID",  value=st.session_state.m_policy_id)
            cat_idx    = categories_list.index(st.session_state.m_claim_category) if st.session_state.m_claim_category in categories_list else 0
            claim_cat  = st.selectbox("Claim Category", categories_list, index=cat_idx)
        with c2:
            treat_date = st.date_input("Treatment Date", value=st.session_state.m_treatment_date)
            claimed_amt= st.number_input("Claimed Amount (₹)", min_value=0.0, value=float(st.session_state.m_claimed_amount), step=100.0)
            hosp_name  = st.text_input("Hospital Name",  value=st.session_state.m_hospital_name)
        with c3:
            pre_auth   = st.text_input("Pre-auth Reference", value=st.session_state.m_pre_auth_reference)
            ytd_amt    = st.number_input("YTD Claims (₹)", min_value=0.0, value=float(st.session_state.m_ytd_claims_amount), step=100.0)

        # Visual Summaries of preloaded JSON metadata
        try:
            hist_list = json.loads(st.session_state.m_claims_history_str)
            docs_list = json.loads(st.session_state.m_documents_str)
        except Exception:
            hist_list = []
            docs_list = []

        vis_doc_col, vis_hist_col = st.columns(2)
        with vis_doc_col:
            st.markdown("**📂 Uploaded Documents Metadata:**")
            if docs_list:
                for doc in docs_list:
                    doc_type = doc.get("actual_type", "UNKNOWN")
                    doc_name = doc.get("file_name", f"Doc {doc.get('file_id')}")
                    p_name = f" | Patient: {doc.get('patient_name_on_doc')}" if doc.get("patient_name_on_doc") else ""
                    st.markdown(f"- 📄 `{doc_name}` ({doc_type}{p_name})")
            else:
                st.info("No documents uploaded.")
        with vis_hist_col:
            st.markdown("**📜 Member Claims History:**")
            if hist_list:
                for hist in hist_list:
                    h_date = hist.get("date", "N/A")
                    h_amt = hist.get("amount", 0)
                    h_prov = hist.get("provider", "Unknown")
                    st.markdown(f"- 🕒 `{h_date}`: ₹{h_amt:,} at *{h_prov}*")
            else:
                st.info("No claims history.")

        # Collapsible Developer Expanders (sim_fail moved here)
        with st.expander("🛠️ Developer Controls (Raw JSON Input & Simulation)", expanded=False):
            sim_fail = st.checkbox("Simulate Component Failure", value=st.session_state.m_simulate)
            st.divider()
            ch_col, doc_col = st.columns(2)
            with ch_col:
                hist_str   = st.text_area("Claims History (JSON)", value=st.session_state.m_claims_history_str, height=180)
            with doc_col:
                docs_str   = st.text_area("Documents & Extraction Metadata (JSON)", value=st.session_state.m_documents_str, height=180)

        mock_submit = st.form_submit_button("▶ Run Mock Pipeline", type="primary")

    if mock_submit:
        try:
            hist_data = json.loads(hist_str)
            history_entries = [
                ClaimsHistoryEntry(
                    claim_id=h.get("claim_id", ""),
                    date=datetime.strptime(h["date"], "%Y-%m-%d").date() if isinstance(h.get("date"), str) else h.get("date"),
                    amount=Decimal(str(h.get("amount", 0))),
                    provider=h.get("provider", ""),
                )
                for h in hist_data
            ]
        except Exception as e:
            st.error(f"Claims History JSON error: {e}")
            st.stop()

        try:
            docs_data = json.loads(docs_str)
            documents = [
                DocumentUpload(
                    file_id=d.get("file_id", ""),
                    file_name=d.get("file_name"),
                    actual_type=d.get("actual_type"),
                    quality=d.get("quality"),
                    content=d.get("content"),
                    patient_name_on_doc=d.get("patient_name_on_doc"),
                )
                for d in docs_data
            ]
        except Exception as e:
            st.error(f"Documents JSON error: {e}")
            st.stop()

        claim_input = ClaimInput(
            member_id=member_id,
            policy_id=policy_id,
            claim_category=claim_cat,
            treatment_date=treat_date,
            claimed_amount=Decimal(str(claimed_amt)),
            hospital_name=hosp_name or None,
            pre_auth_reference=pre_auth or None,
            ytd_claims_amount=Decimal(str(ytd_amt)),
            claims_history=history_entries,
            simulate_component_failure=sim_fail,
            documents=documents,
        )

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.live_mode = False
        st.session_state.last_claim_input = claim_input

        with st.spinner("Processing through LangGraph pipeline (mock mode)…"):
            result = run_claim_pipeline(claim_input, use_mock=True, thread_id=st.session_state.thread_id)
            
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state_info = compiled_graph.get_state(config)
        
        if state_info.next and state_info.values.get("human_override_fields"):
            st.session_state.is_interrupted = True
            st.session_state.override_fields = state_info.values.get("human_override_fields")
            st.session_state.current_result = None
        else:
            st.session_state.is_interrupted = False
            st.session_state.override_fields = None
            st.session_state.current_result = result
            
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live AI Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("Live Document Processing")
    st.markdown(
        """
        Upload real medical documents. The AI automatically extracts structured medical data, 
        matches member profiles, and adjudicates policy rules.

        > **Note:** Requires a valid `GOOGLE_API_KEY` configured in the system environment.
        """
    )

    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("⚠️ `GOOGLE_API_KEY` is not set. Enter it in the sidebar diagnostics before submitting.")

    # ── Claim metadata ──────────────────────────────────────────────────────
    with st.form("live_claim_form"):
        st.markdown("#### 1 · Claim Metadata")
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            l_member_id = st.text_input("Member ID", value="EMP001", key="l_member_id")
            l_policy_id = st.text_input("Policy ID", value="PLUM_GHI_2024", key="l_policy_id")
            l_cats = ["CONSULTATION", "DIAGNOSTIC", "PHARMACY", "DENTAL", "VISION", "ALTERNATIVE_MEDICINE"]
            l_claim_cat = st.selectbox("Claim Category", l_cats, key="l_claim_cat")
        with lc2:
            l_treat_date = st.date_input("Treatment Date", value=date.today(), key="l_treat_date")
            l_claimed_amt = st.number_input("Claimed Amount (₹)", min_value=0.0, value=1500.0, step=100.0, key="l_claimed_amt")
            l_hosp_name = st.text_input("Hospital/Clinic Name", value="", key="l_hosp_name")
        with lc3:
            l_pre_auth = st.text_input("Pre-auth Reference (if any)", value="", key="l_pre_auth")
            l_ytd_amt = st.number_input("YTD Claims Amount (₹)", min_value=0.0, value=0.0, step=100.0, key="l_ytd_amt")

        st.markdown("#### 2 · Upload Documents")
        st.markdown(
            "Upload one or more medical documents (JPEG, PNG, or PDF). "
            "The AI will classify and extract data from each file."
        )

        uploaded_files = st.file_uploader(
            "Drag & drop or browse",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            key="live_file_uploader",
            help="Accepted: JPEG, PNG, PDF. Max 200 MB per file.",
        )

        # Advanced Settings Expanders (l_sim_fail and l_hist_str moved here)
        with st.expander("🛠️ Advanced Settings (Claims History & Failure Simulation)"):
            l_sim_fail = st.checkbox("Simulate Component Failure", value=False, key="l_sim_fail")
            st.divider()
            l_hist_str = st.text_area(
                "Claims History (JSON)",
                value="[]",
                height=120,
                key="l_hist_str",
                help='e.g. [{"claim_id":"CLM001","date":"2024-10-30","amount":500,"provider":"Apollo"}]',
            )

        live_submit = st.form_submit_button("▶ Run Live AI Pipeline", type="primary")

    # ── Preview uploaded files ──────────────────────────────────────────────
    if uploaded_files:
        st.markdown("**Uploaded files preview:**")
        prev_cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files):
            col = prev_cols[i % 4]
            with col:
                if f.type in ("image/jpeg", "image/png"):
                    col.image(f.read(), caption=f.name, use_container_width=True)
                    f.seek(0)  # reset after read
                else:
                    col.markdown(f"📄 **{f.name}** ({f.size // 1024} KB)")

    # ── Submission handler ──────────────────────────────────────────────────
    if live_submit:
        if not os.environ.get("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is required for live mode. Please set it in your environment variables or .env file.")
            st.stop()

        if not uploaded_files:
            st.error("Please upload at least one document.")
            st.stop()

        if len(uploaded_files) > 5:
            st.error("Maximum of 5 documents can be uploaded per claim.")
            st.stop()

        allowed_mimes = ["image/jpeg", "image/png", "application/pdf"]
        for f in uploaded_files:
            if f.size > 10 * 1024 * 1024:
                st.error(f"File {f.name} exceeds the 10 MB size limit.")
                st.stop()
            if f.type not in allowed_mimes:
                st.error(f"File {f.name} has an unsupported format: {f.type}. Only JPEG, PNG, and PDF are allowed.")
                st.stop()

        # Session-based Rate Limiting
        import time
        if "submission_timestamps" not in st.session_state:
            st.session_state.submission_timestamps = []
        
        now = time.time()
        st.session_state.submission_timestamps = [t for t in st.session_state.submission_timestamps if now - t < 60]
        
        if len(st.session_state.submission_timestamps) >= 3:
            st.error("Rate limit exceeded. Please wait a minute before running another live AI extraction.")
            st.stop()
            
        st.session_state.submission_timestamps.append(now)

        live_documents: List[DocumentUpload] = []
        for i, f in enumerate(uploaded_files):
            raw_bytes = f.read()
            live_documents.append(
                DocumentUpload(
                    file_id=f"F{i+1:03d}",
                    file_name=f.name,
                    image_bytes=raw_bytes,
                )
            )

        try:
            l_hist_data = json.loads(l_hist_str)
            l_history = [
                ClaimsHistoryEntry(
                    claim_id=h.get("claim_id", ""),
                    date=datetime.strptime(h["date"], "%Y-%m-%d").date() if isinstance(h.get("date"), str) else h.get("date"),
                    amount=Decimal(str(h.get("amount", 0))),
                    provider=h.get("provider", ""),
                )
                for h in l_hist_data
            ]
        except Exception as e:
            st.error(f"Claims History JSON error: {e}")
            st.stop()

        live_claim_input = ClaimInput(
            member_id=l_member_id,
            policy_id=l_policy_id,
            claim_category=l_claim_cat,
            treatment_date=l_treat_date,
            claimed_amount=Decimal(str(l_claimed_amt)),
            hospital_name=l_hosp_name or None,
            pre_auth_reference=l_pre_auth or None,
            ytd_claims_amount=Decimal(str(l_ytd_amt)),
            claims_history=l_history,
            simulate_component_failure=l_sim_fail,
            documents=live_documents,
        )

        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.live_mode = True
        st.session_state.last_claim_input = live_claim_input

        status_placeholder = st.empty()
        progress = st.progress(0)

        def show_status(msg: str, pct: int) -> None:
            status_placeholder.info(f"⚙️ {msg}")
            progress.progress(pct)

        show_status("Starting LangGraph multi-agent pipeline…", 5)
        show_status(f"Classifying {len(live_documents)} document(s) with gemini-3.1-flash-lite…", 20)

        try:
            result = run_claim_pipeline(
                live_claim_input,
                use_mock=False,
                thread_id=st.session_state.thread_id,
            )
        except Exception as exc:
            progress.empty()
            status_placeholder.empty()
            st.error(f"Pipeline error: {exc}")
            st.stop()

        progress.progress(100)
        status_placeholder.success("✅ Pipeline complete.")

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state_info = compiled_graph.get_state(config)
        
        if state_info.next and state_info.values.get("human_override_fields"):
            st.session_state.is_interrupted = True
            st.session_state.override_fields = state_info.values.get("human_override_fields")
            st.session_state.current_result = None
        else:
            st.session_state.is_interrupted = False
            st.session_state.override_fields = None
            st.session_state.current_result = result
            
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT ADJUDICATION RESULTS VIEW
# ─────────────────────────────────────────────────────────────────────────────

# 1. Interrupt / Human-in-the-Loop Override Section
if st.session_state.is_interrupted:
    st.markdown("---")
    st.markdown('<div class="section-title">⚠️ Human-in-the-Loop Override Required</div>', unsafe_allow_html=True)
    
    fields = st.session_state.override_fields
    mismatch = fields["mismatched_names"][0]
    
    st.warning(
        f"""
        **Borderline Patient Name Similarity Detected:** Jaro-Winkler Similarity: **{mismatch['similarity']:.2%}**
        
        * **Document Patient Name:** `{mismatch['name1']}` (from `{mismatch['file1']}`)
        * **Policy Roster Name:** `{mismatch['name2']}` (from `{mismatch['file2']}`)
        
        The multi-agent system requires visual confirmation to verify if these names belong to the same person.
        """
    )
    
    with st.form("override_resubmission_form"):
        adj_corrected_name = st.text_input("Corrected Patient Name (matches member database)", value=fields["suggested_name"])
        
        col1, col2 = st.columns(2)
        with col1:
            approve_resume = st.form_submit_button("✅ Approve Name & Resume Adjudication", type="primary")
        with col2:
            reject_claim = st.form_submit_button("❌ Reject Claim (Verified Mismatch)", type="secondary")
            
        if approve_resume:
            with st.spinner("Injecting correction and resuming workflow..."):
                final_result = resume_claim_pipeline(st.session_state.thread_id, adj_corrected_name)
            st.session_state.current_result = final_result
            st.session_state.is_interrupted = False
            st.session_state.override_fields = None
            st.success("Claim resumed and adjudicated successfully!")
            st.rerun()
            
        if reject_claim:
            # Reject immediately
            claim_id = f"CLM_{st.session_state.last_claim_input.member_id}"
            st.session_state.current_result = ClaimResponse(
                claim_id=claim_id,
                outcome_type="EARLY_STOP",
                early_stop=EarlyStopResponse(
                    claim_id=claim_id,
                    status="EARLY_STOP",
                    stop_stage="CONSISTENCY_CHECK",
                    stop_reason=RejectionReason.PATIENT_NAME_MISMATCH,
                    user_message=f"Claim rejected: Patient name mismatch between document '{mismatch['name1']}' and policy holder '{mismatch['name2']}' verified by claims adjuster.",
                    documents_uploaded=[],
                    documents_required=[],
                    documents_missing=[],
                    trace=[]
                ),
                decision=None
            )
            st.session_state.is_interrupted = False
            st.session_state.override_fields = None
            st.info("Claim rejected by claims adjuster override.")
            st.rerun()

# 2. Render Final Adjudication Results
if st.session_state.current_result is not None:
    render_decision_result(st.session_state.current_result)
    render_ground_truth_comparison(st.session_state.current_result)