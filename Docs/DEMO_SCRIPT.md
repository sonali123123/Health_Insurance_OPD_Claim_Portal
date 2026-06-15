# UI Demo Script

This script outlines the segments designed to demonstrate the system's capabilities through the Streamlit portal UI.

---

## Segment 1: Document Upload Gating & Early Stop (TC001)
*Duration: 2–3 minutes*

### Objective
Show how the multi-agent pipeline immediately detects missing document types and stops processing early, providing a helpful, clear guidance message to the member.

### Steps
1. Open the Streamlit portal (`app.py`).
2. In the sidebar dropdown **"Pre-load Ground-Truth Test Case"**, select **"TC001: Wrong Document Uploaded"**.
3. Observe how the form automatically populates with member `EMP001` and category `CONSULTATION`.
4. In the document metadata section, note that two `PRESCRIPTION` files are loaded (with no hospital bill).
5. Click **"Run Claims Adjudication Pipeline"**.
6. **UI Output**:
   - The status bar stops at the "Document Set Validator" stage.
   - An error callout box displays the `EARLY_STOP` outcome.
   - A detailed member-facing message is rendered: 
     > *"You uploaded 2 PRESCRIPTION documents. This CONSULTATION claim requires: PRESCRIPTION (found ✓) and HOSPITAL_BILL (missing ✗). Please upload a hospital bill or clinic invoice and resubmit."*
   - Expand the **"Adjudication Audit Trail & Trace Steps"** to show that `t_valset_TC001` failed with `result = FAILED`.

---

## Segment 2: Graceful Degradation & Tool Use (TC011)
*Duration: 3–4 minutes*

### Objective
Show how the multi-agent pipeline handles a mid-processing extraction component failure gracefully, utilizing mock metadata fallback and logging the degraded status.

### Steps
1. In the sidebar dropdown, select **"TC011: Component Failure — Graceful Degradation"**.
2. Notice the **"Simulate Component Failure"** checkbox is automatically checked.
3. Click **"Run Claims Adjudication Pipeline"**.
4. **UI Output**:
   - The progress tracker indicates nodes completed successfully, but notes component failure.
   - An outcome banner displays **APPROVED** with approved amount `₹4,000.00`.
   - A warning card appears: *"Manual review recommended due to incomplete processing"*.
   - Check the **Metrics display**:
     - Gross Claimed: `₹4,000.00`
     - Approved Payout: `₹4,000.00`
   - Expand the **"Adjudication Audit Trail & Trace Steps"**:
     - Walk through the log showing the `request_reextraction` tool execution.
     - Note the degraded trace step `adj_degraded` showing `confidence_delta = 0.25` and the final confidence score reduced to `0.75`.

---

## Segment 3: Network Hospital Discount & Copay Adjudication (TC010)
*Duration: 4–5 minutes*

### Objective
Demonstrate an end-to-end claim approval at a network hospital, highlighting the sequential application of the network discount followed by the outpatient copay.

### Steps
1. In the sidebar dropdown, select **"TC010: Network Hospital — Discount Applied"**.
2. Notice the pre-loaded fields:
   - Member ID: `EMP010`
   - Claimed Amount: `₹4,500`
   - Hospital: `Apollo Hospitals` (a network hospital listed in `policy_terms.json`).
3. Click **"Run Claims Adjudication Pipeline"**.
4. **UI Output**:
   - The progress tracker indicates all nodes completed successfully (green).
   - An outcome banner displays **APPROVED**.
   - Explanation text shows:
     > *"Network discount (20%) applied first on ₹4,500 = ₹3,600. Co-pay (10%) applied on ₹3,600 = ₹360 deducted. Final: ₹3,240."*
   - Check the **Metrics display**:
     - Gross Claimed: `₹4,500.00`
     - Network Discount: `₹900.00` (20%)
     - Copay Applied: `₹360.00` (10% of ₹3600)
     - Approved Payout: `₹3,240.00` (matches ground-truth target exactly)
   - Expand the **"Adjudication Audit Trail & Trace Steps"** and walk through the chronological log entries showing `adj_012` (Network Discount applied) preceding `adj_013` (Copay applied).

---

## Segment 4: Technical Design & Deep-Dive
*Duration: 2–3 minutes*

### Objective
Summarize the key architectural patterns that prevent hallucinations, ensure type safety, and provide absolute audit transparency.

### Talking Points
1. **Durable Orchestration**: Show how LangGraph manages the multi-agent graph with parallel document processing (`Send` API) and Fan-in Aggregation.
2. **Human-in-the-Loop Override**: Point out that the graph compiles with `interrupt_after=["consistency_node"]`, enabling the adjuster override form to resolve borderline patient name spelling variances.
3. **Mathematical Precision**: Show that all calculations are handled strictly via Python `Decimal` data types, ensuring 100% agreement with expected financial amounts and avoiding float inaccuracies.
4. **Separation of Concerns**: Point out that LLMs are not used to make financial determinations or apply rules; they are confined strictly to document extraction, while the policy engine evaluates logic deterministically.
5. **Traceability**: Emphasize that every rule application generates a `TraceStep` object capturing input parameters, computed values, rule descriptions, and execution latency.
6. **Static Analysis**: Demonstrate that type checks (`mypy --strict --explicit-package-bases src/`) run successfully across the entire codebase to maintain software quality.
