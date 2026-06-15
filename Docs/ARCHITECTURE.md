# Architecture Overview

This document describes the architecture and design of the **Trace-First Health Insurance Claims Processing System** for Group Health Insurance OPD plans.

The system is split into two distinct, coupled halves:
1. **Probabilistic Multi-Agent Pipeline (LangGraph)**: Responsible for parallel document processing (classification, quality verifier, fact extraction), aggregation, document set validation, quality gate checks, name consistency checks (with human override support), and same-day fraud counting.
2. **Deterministic Policy Engine (Python)**: Executes the 14-step adjudication sequence using strict `Decimal` math. It loads constraints dynamically from `policy_terms.json` and evaluates membership, waiting periods, pre-auth requirements, exclusions, network discounts, and copays.

---

## System Architecture Diagram

```mermaid
graph TD
    START([Submit ClaimInput]) --> fan_out{Parallel Fan-out}
    fan_out -- "For each doc" --> process_doc["process_document_node<br>(Classifier -> Quality -> Extractor)"]
    process_doc --> fan_in[Fan-in Aggregator Node]
    
    fan_in --> validate_set{validate_set_node<br>(Exit 1: Document Set Validator)}
    
    validate_set -- "Missing required docs" --> early_stop[Early Stop Outcome]
    validate_set -- "Docs Present" --> quality_gate{quality_gate_node<br>(Exit 2: Quality Gate)}
    
    quality_gate -- "Any unreadable doc" --> early_stop
    quality_gate -- "All readable" --> consistency{consistency_node<br>(Exit 3: Consistency Checker)}
    
    consistency -- "Name mismatch similarity < 0.75" --> early_stop
    consistency -- "Name similarity 0.75 - 0.85" --> hitl[HUMAN_OVERRIDE / Adjuster Input]
    consistency -- "Name similarity >= 0.85" --> policy_engine[policy_node]
    
    hitl -- "Resume with corrected name" --> policy_engine
    
    policy_engine --> fraud_agent[fraud_node<br>(Fraud Same-Day Claims Check)]
    fraud_agent --> END([ClaimResponse Output])
    
    early_stop --> END
```

---

## Probabilistic vs. Deterministic Separation

To guarantee mathematical correctness and zero AI hallucinations for financial and regulatory compliance, the system separates cognitive LLM-based tasks from financial computations:

| Pipeline Half | Responsibility | Technology Stack |
|---|---|---|
| **Probabilistic** | Document classification, quality checks, fact/line item extraction, patient name Jaro-Winkler similarity, same-day fraud counting. | LangGraph, `google-genai` SDK, `groq` SDK, OpenCV, Jellyfish (Jaro-Winkler). |
| **Deterministic** | Exclusions check (Step 6), waiting periods (Step 7), per-claim limits (Step 9), network discounts (Step 12), copays (Step 13). | Python `Decimal` math, standard policy JSON config parsing. |

---

## Component Summary Table

| Component | Class / Function | Location | Role / Responsibility |
|---|---|---|---|
| **Document Subgraph** | `process_document_node` | `src/pipeline.py` | Runs Classifier, Quality Verifier, and Fact Extractor in parallel for each document. |
| **Fan-in Aggregator** | `aggregate_subgraph_results` | `src/pipeline.py` | Merges per-document subgraph results and reduces confidence on timeouts. |
| **Document Set Validator** | `validate_set_node` | `src/pipeline.py` | Asserts uploaded files match category requirements (Exit 1). |
| **Quality Gate Validator** | `quality_gate_node` | `src/pipeline.py` | Standalone check for document readability scores (Exit 2). |
| **Consistency Checker** | `consistency_node` | `src/pipeline.py` | Cross-document and member-to-document patient name checking (Exit 3 / HITL). |
| **Policy Engine Node** | `policy_node` | `src/pipeline.py` | Invokes the deterministic policy engine. |
| **Policy Engine** | `adjudicate_claim` | `src/policy_engine.py` | Evaluates 14 sequential rules using Decimal currency math. |
| **Fraud Agent Node** | `fraud_node` | `src/pipeline.py` | Runs volume checks (same-day limits) and reconciles final confidence. |
| **UI Portal** | Streamlit Dashboard | `app.py` | Selector for test cases, manual form entry, visual progress and explainability trace. |

---

## Data Flow Diagram

1. **Submission**: User uploads files and claims metadata via the Streamlit UI or `pytest` harness.
2. **Preprocessing**: PDFs are rasterized to image bytes (using PyMuPDF).
3. **Graph Execution**:
   - Files are sent to Classifier/Quality/Extractor nodes in parallel using LangGraph's `Send` API.
   - Outputs are merged in the Fan-in Aggregator node.
   - Gating validations are checked at the Document Set, Quality Gate, and Consistency Checker exits.
   - Names are cross-compared. Borderline matches suspend the graph for human intervention (HITL) using checkpointers.
4. **Policy Engine Adjudication**: 14 steps executed sequentially. Returns approved line items, copay details, discounts, and log traces.
5. **Output**: System serializes the results to `ClaimResponse` and logs execution metadata to `Docs/EVAL_REPORT.md` (for pytest) or displays them in Streamlit.

---

## Tech Stack Rationale

- **LangGraph**: Organizes multi-agent state machines, handles loops, and facilitates checkpointer persistence for human-in-the-loop overrides.
- **OpenCV**: Computes Laplacian variance for fast, cost-effective document readability validation before calling Gemini API.
- **Python Decimal**: Prevents floating-point precision errors (e.g. ₹3240.00 vs ₹3240.0000000004) when processing currency transactions.
- **Jellyfish**: Computes Jaro-Winkler name similarity scores to match Indian names that are spelled differently across various bills.

---

## Deployment & Running Instructions

### Local Execution (Streamlit App)
Start the portal using the Streamlit command line:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Test Harness
Run the 12 parametrized test cases in mock mode:
```bash
pytest
```
Test results will automatically write/update `Docs/EVAL_REPORT.md`.

### Strict Type Validation
Verify type safety using mypy:
```bash
mypy --strict --explicit-package-bases src/
```
