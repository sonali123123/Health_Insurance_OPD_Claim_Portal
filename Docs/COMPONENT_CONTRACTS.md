# Component Contracts

This document outlines the interfaces, function signatures, data models, error handling, and exact contract terms for each core component in the Claims Processing System.

---

## 1. Document Subgraph Node (Classifier, Quality, Extractor)

- **Function**: `process_document_node` (invoked in parallel via LangGraph `Send` API)
- **Signature**: `process_document_node(state: DocSubgraphState) -> Dict[str, Any]`
- **Agents Involved**:
  - **Agent 0 (Classifier)**: Categorizes files into `PRESCRIPTION`, `HOSPITAL_BILL`, `PHARMACY_BILL`, `LAB_REPORT`, `DISCHARGE_SUMMARY`, `DENTAL_REPORT`, or `UNKNOWN`.
  - **Agent 1 (Quality Verifier)**: Computes local OpenCV Laplacian variance check. variance < 75 is marked unreadable.
  - **Agent 2 (Fact Extractor)**: Extracts patient name, doctor registration, diagnosis, bill items, and total amount.
- **Genuine Tool-Use**:
  - **`request_reextraction(file_id, reason)`**: Invoked by Fact Extractor if confidence is < 0.50 or during simulated component failure.
- **Contracts**:
  - In mock mode (or when `simulate_component_failure` is enabled), parses the mock data from `doc.content` (to allow graceful degradation downstream) while applying appropriate confidence penalties.

---

## 2. Fan-in Aggregator

- **Function**: `aggregate_subgraph_results` (invoked after parallel execution completes)
- **Signature**: `aggregate_subgraph_results(state: ClaimsState) -> Dict[str, Any]`
- **Contracts**:
  - Gathers outputs from each document processing subgraph.
  - Detects timeouts or crashes in subgraphs. Applies a confidence deduction of `-0.25` for each failed/timed-out subgraph and logs them to `failed_subgraphs` and `pipeline_warnings`.

---

## 3. Document Set Validator (Exit 1)

- **Function**: `validate_set_node`
- **Signature**: `validate_set_node(state: ClaimsState) -> Dict[str, Any]`
- **Error Conditions**: If the required documents for the claim category (defined in `policy_terms.json`) are not present in the classified types, triggers `EARLY_STOP` with `RejectionReason.DOCUMENT_TYPE_MISMATCH`.
- **Contracts**:
  - The `user_message` in the early stop response must explicitly state the uploaded document counts and list the missing document types with checked status (e.g. `PRESCRIPTION (found ✓) and HOSPITAL_BILL (missing ✗)`).

---

## 4. Quality Gate Validator (Exit 2)

- **Function**: `quality_gate_node`
- **Signature**: `quality_gate_node(state: ClaimsState) -> Dict[str, Any]`
- **Error Conditions**: If any of the required documents are marked unreadable (Laplacian variance < 75), triggers `EARLY_STOP` with `RejectionReason.DOCUMENT_UNREADABLE`.
- **Contracts**:
  - Emits user message detailing which document filename is unreadable.

---

## 5. Consistency Checker Agent (Exit 3 / HITL)

- **Function**: `consistency_node`
- **Signature**: `consistency_node(state: ClaimsState) -> Dict[str, Any]`
- **Genuine Tool-Use**:
  - **`resolve_name_ambiguity(name_a, name_b)`**: Invoked for borderline Jaro-Winkler name similarity scores.
- **Contracts**:
  - **Phase 1**: Checks cross-document patient name Jaro-Winkler similarity.
  - **Phase 2**: Checks member-to-document patient name Jaro-Winkler similarity.
  - Mismatch similarity < 0.75 triggers `EARLY_STOP` with `RejectionReason.PATIENT_NAME_MISMATCH`.
  - Mismatch similarity between 0.75 and 0.85 (borderline) triggers name ambiguity check, applies a confidence penalty of `-0.05`, and sets `human_override_fields` to prompt adjuster override.
  - Graph compiles with `interrupt_after=["consistency_node"]` to enable this HITL step.

---

## 6. Deterministic Policy Engine Node

- **Function**: `policy_node` (LangGraph wrapper node) and `adjudicate_claim` (Python engine)
- **Signature**:
  ```python
  def adjudicate_claim(
      claim_input: ClaimInput,
      policy: Dict[str, Any],
      extractions: List[FactExtractionPayload],
      pipeline_confidence: Optional[float] = None
  ) -> AdjudicationOutput:
  ```
- **Contracts**:
  - Executes 14 sequential rules using Decimal currency math and dynamically loaded `policy_terms.json` values.
  - Throws `AdjudicationValidationError` if final approved amount is negative.
  - Step 6 (exclusions) MUST precede Step 7 (waiting periods) to ensure permanent exclusions take precedence (TC012).
  - Step 12 (network discount) MUST precede Step 13 (copays).

---

## 7. Fraud & Risk Agent Node

- **Function**: `fraud_node`
- **Signature**: `fraud_node(state: ClaimsState) -> Dict[str, Any]`
- **Genuine Tool-Use**:
  - **`query_claims_history(member_id, date)`**: Queries claims history for same-day submissions.
- **Contracts**:
  - Counts claims history entries matching current day's submission date.
  - If total same-day claims count exceeds `same_day_claims_limit` (typically 2), sets decision to `MANUAL_REVIEW`, adds `FRAUD_SUSPECTED` rejection reason, and applies a confidence penalty of `-0.10`.
  - Reconciles final confidence and compiles the final `ClaimResponse`.
