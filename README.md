---
title: Health Insurance Claims Processing System
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.46.1
app_file: app.py
pinned: false
---

# Trace-First Health Insurance Claims Processing System

> **Group Health Insurance OPD Claims Adjudication**  
> Policy: PLUM_GHI_2024 (ICICI Lombard) · Plan: OPD-only

A production-grade, auditable insurance claims processing engine built on a **LangGraph multi-agent pipeline** and a **deterministic Python policy engine**. Every single decision - from document classification to copay calculation - is backed by a structured `TraceStep` audit trail embedded in the response payload.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Key Features](#2-key-features)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [Configuration (.env)](#6-configuration-env)
7. [Running the Portal](#7-running-the-portal)
8. [Running Tests](#8-running-tests)
9. [Using the Streamlit Portal](#9-using-the-streamlit-portal)
10. [Live AI Mode — How It Works](#10-live-ai-mode--how-it-works)
11. [Policy Engine Rules](#11-policy-engine-rules)
12. [API Reference (Programmatic Use)](#12-api-reference-programmatic-use)
13. [Test Case Coverage](#13-test-case-coverage)
14. [Known Limitations](#14-known-limitations)

---

## 1. Architecture Overview

```
[ Streamlit Portal — app.py ]
           │
           ▼
[ LangGraph Orchestrator — src/pipeline.py ]
           │
     Fan-out (Send API — parallel per document)
     ├── [ Doc Subgraph: File 1 ]
     │     ├─ Agent 0: Document Classifier   (gemini-3.1-flash-lite vision)
     │     ├─ Agent 1: Quality Verifier      (OpenCV blur + LLM flags)
     │     └─ Agent 2: Fact Extractor        (LLM + request_reextraction tool)
     └── [ Doc Subgraph: File N ]  (same pipeline, parallel)
           │
     Fan-in Aggregator
           │
     ┌─────┴─────────────────────────┐
     │ Document Set Validator         │ ← EARLY_STOP Exit 1 (TC001)
     │ Quality Gate Validator         │ ← EARLY_STOP Exit 2 (TC002)
     │ Consistency Checker            │ ← EARLY_STOP Exit 3 (TC003)
     │   (Jaro-Winkler + tool)        │
     │ ── HITL interrupt_after ──     │ (human override for 0.75–0.85 names)
     │ Deterministic Policy Engine    │ ← 14-step sequential rules
     │ Fraud & Risk Agent             │ ← query_claims_history tool
     └───────────────────────────────┘
           │
     [ ClaimResponse — AdjudicationOutput + List[TraceStep] ]



```

**Two clearly separated halves:**
- **Probabilistic half** — LLMs for document classification, quality assessment, and fact extraction.
- **Deterministic half** — Pure Python `Decimal` math for all policy rules. No LLM touches a calculation.

---

## 2. Key Features

| Feature | Detail |
|---|---|
| **Multi-agent LangGraph graph** | Parallel document processing via `Send` API, HITL `interrupt_after`, `MemorySaver` checkpointing |
| **LLM fallback chain** | `gemini-3.1-flash-lite` → `gemini-2.5-flash-lite` → `Groq Llama-4-Scout` |
| **Confidence penalties** | −0.05 on Groq escalation; −0.10 on Groq + bill document |
| **PII redaction** | Aadhaar numbers and Indian phone numbers stripped before any external API call |
| **PDF rasterization** | PyMuPDF converts multi-page PDFs → per-page PNG before LLM |
| **OpenCV blur detection** | Laplacian variance < 75 triggers fast-fail before any LLM call |
| **14-step policy engine** | All rules read from `policy_terms.json`; `Decimal` math throughout |
| **Full audit trail** | Every decision emits a `TraceStep` embedded in `ClaimResponse` |
| **12-case test suite** | ₹0 financial variance across all 12 test cases; zero API calls in mock mode |
| **Streamlit portal** | Dual-tab UI: mock test cases + live file upload |

---

## 3. Project Structure

```
Health_Insurance_Claims_Processing_System/
│
├── app.py                        # Streamlit portal entry point
├── conftest.py                   # pytest fixtures
├── .env.example                  # Environment variable template
│
├── src/
│   ├── models.py                 # Pydantic V2 schemas (ClaimInput, ClaimsState, etc.)
│   ├── pipeline.py               # LangGraph multi-agent graph & run_claim_pipeline()
│   ├── llm_extraction.py         # Live LLM extraction (Gemini + Groq, PII, PDF)
│   ├── policy_engine.py          # Deterministic 14-step adjudication engine
│   └── utils.py                  # assess_blur(), strip_pii(), normalize_diagnosis()
│
├── tests/
│   ├── test_pipeline.py          # 12-case parametrized end-to-end suite
│   ├── test_policy_engine.py     # Unit tests for policy rules
│   └── test_utils.py             # Unit tests for blur/PII/normalization utilities
│
└── Docs/
    ├── PRD_v2.3.md               # Full product requirements document
    ├── policy_terms.json         # All policy rules (OPD, limits, exclusions, members)
    ├── test_cases.json           # 12 ground-truth test cases
    ├── EVAL_REPORT.md            # Auto-generated on every pytest run
    ├── ARCHITECTURE.md           # System architecture deep-dive
    ├── COMPONENT_CONTRACTS.md    # Per-component input/output contracts
    └── DEMO_SCRIPT.md            # Step-by-step demo walkthrough
```

---

## 4. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| `google-genai` SDK | 1.26.0 | For Gemini vision calls |
| `groq` SDK | 1.4.0 | For Llama-4-Scout fallback |
| `PyMuPDF` | 1.26.0 | PDF → image rasterization |
| `opencv-python-headless` | 4.11+ | Blur detection |
| `langgraph` | 0.5.0 | Graph orchestration |
| `streamlit` | 1.46.1 | Web portal |
| `pydantic` | 2.12.5 | Schema validation |
| `jellyfish` | 1.2.1 | Jaro-Winkler name matching |

**API keys (for Live AI mode only):**
- `GOOGLE_API_KEY` — from [Google AI Studio](https://aistudio.google.com/app/apikey) (free tier: 15 RPM / 500 RPD)
- `GROQ_API_KEY` — from [Groq Console](https://console.groq.com/keys) (free tier: 14,400 RPD)

> **Mock mode works without any API keys.** All 12 test cases and the "Mock Test Cases" tab in the portal run completely offline.

---

## 5. Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Health_Insurance_Claims_Processing_System

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables (only needed for Live AI mode)
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and GROQ_API_KEY
```

> If you don't have a `requirements.txt` yet, install core packages directly:
> ```bash
> pip install google-genai==1.26.0 groq langgraph streamlit pydantic[email] \
>             PyMuPDF opencv-python-headless jellyfish pandas python-dotenv \
>             pytest mypy
> ```

---

## 6. Configuration (.env)

Copy `.env.example` to `.env` and fill in your keys:

```env
# Required for Live AI Upload tab
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional — Langfuse observability (set false to disable)
LANGFUSE_ENABLED=false
```

Load the `.env` file before running:

```bash
# Option A — let python-dotenv auto-load (already in conftest.py)
# Nothing extra needed.

# Option B — export manually (PowerShell)
$env:GOOGLE_API_KEY = "AIza..."
$env:GROQ_API_KEY   = "gsk_..."

# Option B — export manually (bash)
export GOOGLE_API_KEY="AIza..."
export GROQ_API_KEY="gsk_..."
```

Alternatively, enter keys directly in the **sidebar** of the Streamlit portal without touching any files.

---

## 7. Running the Portal

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 8. Running Tests

```bash
# Full test suite (12 cases, mock mode — no API keys needed)
pytest

# Verbose output with case names
pytest -v

# Single test case
pytest tests/test_pipeline.py -k "TC006"

# Type checking
mypy --strict src/
```

Expected output:
```
12 passed in ~1s
```

`Docs/EVAL_REPORT.md` is auto-generated on every `pytest` run with decision and financial variance for all 12 cases.

---

## 9. Using the Streamlit Portal

### Tab 1 — 🗂️ Mock Test Cases

1. Use the **"Pre-load Test Case"** dropdown to select any of the 12 ground-truth scenarios (TC001–TC012).
2. The form auto-fills with member ID, treatment date, claimed amount, document metadata, etc.
3. Click **"▶ Run Mock Pipeline"**.
4. Results appear immediately — no API calls, no waiting.

**What you'll see:**
- For TC001–TC003: an **Early Stop** banner with the specific actionable error message.
- For TC004–TC012: the **Decision** (APPROVED / PARTIAL / REJECTED / MANUAL_REVIEW), financial breakdown, and the full **Audit Trail** table.

### Tab 2 — 🤖 Live AI Upload

1. Enter your `GOOGLE_API_KEY` (and optionally `GROQ_API_KEY`) in the **sidebar**.
2. Fill in the claim metadata (member ID, treatment date, amount, etc.).
3. Use the **file uploader** to drag-and-drop your real medical documents (JPEG, PNG, or PDF).
4. Click **"▶ Run Live AI Pipeline"**.

**What happens behind the scenes:**
```
Your file upload
      │
      ▼
OpenCV blur check (< 500 ms)
      │
      ▼  [if readable]
gemini-3.1-flash-lite (classify + quality flags + extract)
      │
      ├─ 429 or conf < 0.70? → gemini-2.5-flash-lite
      │                              │
      │              429 again? → Groq Llama-4-Scout (−0.05 confidence penalty)
      ▼
Pydantic V2 validation → Deterministic Policy Engine → Fraud Agent
      │
      ▼
ClaimResponse + full TraceStep audit trail
```

---

## 10. Live AI Mode — How It Works

### Model Fallback Chain

| Priority | Model | Rate Limit (Free) | Triggered When |
|---|---|---|---|
| Primary | `gemini-3.1-flash-lite` | 15 RPM / 500 RPD | Always first |
| Secondary | `gemini-2.5-flash-lite-preview` | 10 RPM / 20 RPD | 429 error OR confidence < 0.70 |
| Tertiary | `groq/llama-4-scout` | 14,400 RPD | Secondary also fails |

### Confidence Penalties Applied

| Event | Pipeline Confidence Delta |
|---|---|
| Component failure (any node) | −0.25 |
| Extraction confidence < 0.70 | −0.15 |
| Groq fallback invoked | −0.05 |
| Groq fallback + bill document | −0.10 |
| Fraud flag triggered | −0.10 |
| Name similarity 0.75–0.85 (borderline) | −0.05 |

### PII Redaction (automatic)

Before any text or extracted content reaches an external API, the following are redacted:
- **Aadhaar numbers** — 12-digit or space-separated `XXXX XXXX XXXX` format
- **Indian phone numbers** — `+91`, `91`, and local 10-digit formats

### PDF Support

Multi-page PDFs are rasterized with **PyMuPDF** at 150 DPI before being sent to Gemini. Each page becomes a separate PNG; the first page is used for single-document classification.

---

## 11. Policy Engine Rules

All rules execute in this exact sequence. First failing rule terminates the sequence.

| Step | Rule | Test Case |
|---|---|---|
| 1 | Policy Active Check | — |
| 2 | Member Eligibility | — |
| 3 | Submission Deadline (30 days) | — |
| 4 | Minimum Claim Amount (₹500) | — |
| 5 | Initial Waiting Period (30 days) | — |
| 6 | Global Exclusion Check (obesity, bariatric, cosmetic) | TC012 |
| 7 | Specific Condition Waiting Periods (diabetes: 90d, etc.) | TC005 |
| 8 | Pre-Authorization Check (MRI > ₹10,000) | TC007 |
| 9 | Per-Claim Limit (₹5,000, for applicable categories) | TC008 |
| 10 | Annual OPD Limit (₹50,000) | — |
| 11 | Category-Specific Rules (dental exclusions, AYUSH validation) | TC006, TC011 |
| 12 | Network Hospital Discount (applied before co-pay) | TC010 |
| 13 | Co-pay Deduction | TC004, TC010 |
| 14 | High-Value Auto-MANUAL_REVIEW (> ₹25,000) | TC009 |

All limits and rates are read from [`Docs/policy_terms.json`](Docs/policy_terms.json) — nothing is hardcoded.

---

## 12. API Reference (Programmatic Use)

```python
from src.pipeline import run_claim_pipeline
from src.models import ClaimInput, DocumentUpload

# ── Mock mode (zero API calls) ─────────────────────────────────────────────
claim_input = ClaimInput(
    member_id="EMP001",
    policy_id="PLUM_GHI_2024",
    claim_category="CONSULTATION",
    treatment_date=date(2024, 11, 1),
    claimed_amount=Decimal("1500"),
    documents=[
        DocumentUpload(
            file_id="F001",
            file_name="prescription.jpg",
            actual_type="PRESCRIPTION",   # mock bypass
            quality="GOOD",               # mock bypass
            content={
                "doctor_name": "Dr. Sharma",
                "patient_name": "Arjun Mehta",
                "diagnosis": "Hypertension",
            },
        ),
    ],
)

result = run_claim_pipeline(claim_input, use_mock=True)
print(result.outcome_type)       # "DECISION"
print(result.decision.decision)  # "APPROVED"
print(result.decision.approved_amount)  # Decimal("1350.00")

# ── Live AI mode (requires API keys) ──────────────────────────────────────
with open("prescription.jpg", "rb") as f:
    raw_bytes = f.read()

live_doc = DocumentUpload(
    file_id="F001",
    file_name="prescription.jpg",
    image_bytes=raw_bytes,          # real file bytes — LLM classifies & extracts
)

result = run_claim_pipeline(
    ClaimInput(..., documents=[live_doc]),
    use_mock=False,
)
```

### Response Shape

```python
# Early Stop (TC001–TC003)
result.outcome_type          # "EARLY_STOP"
result.early_stop.stop_stage # "DOCUMENT_SET_VALIDATION" | "QUALITY_CHECK" | "CONSISTENCY_CHECK"
result.early_stop.user_message  # human-readable explanation

# Decision (TC004–TC012)
result.outcome_type              # "DECISION"
result.decision.decision         # "APPROVED" | "PARTIAL" | "REJECTED" | "MANUAL_REVIEW"
result.decision.approved_amount  # Decimal — final payout
result.decision.confidence_score # float 0.0–1.0
result.decision.audit_trace      # List[TraceStep] — full audit trail
```

---

## 13. Test Case Coverage

| Case | Scenario | Expected | Amount |
|---|---|---|---|
| TC001 | Wrong document type submitted | EARLY_STOP | — |
| TC002 | Unreadable / blurry document | EARLY_STOP | — |
| TC003 | Documents belong to different patients | EARLY_STOP | — |
| TC004 | Clean consultation claim | APPROVED | ₹1,350 |
| TC005 | Diabetes within 90-day waiting period | REJECTED | — |
| TC006 | Dental: root canal (covered) + whitening (excluded) | PARTIAL | ₹8,000 |
| TC007 | MRI without pre-authorization | REJECTED | — |
| TC008 | Consultation exceeds per-claim limit | REJECTED | — |
| TC009 | 4th same-day claim — fraud signal | MANUAL_REVIEW | — |
| TC010 | Apollo Hospitals — network discount applied | APPROVED | ₹3,240 |
| TC011 | Component failure — graceful degradation | APPROVED (degraded) | ₹4,000 |
| TC012 | Bariatric / obesity treatment — excluded | REJECTED | — |

All 12 pass with **₹0 financial variance** in mock mode.

---

## 14. Known Limitations

| Limitation | Detail |
|---|---|
| **Sub-limit annual enforcement** | Per-category YTD tracking is deferred (requires claims history service) |
| **Family floater combined limit** | Cross-member state not tracked; deferred |
| **Alternative medicine session limit** | `max_sessions_per_year: 20` — not enforced without session history |
| **Branded vs. generic drug detection** | Cannot be determined from bill text; 0% copay applied conservatively |
| **Gemini free-tier RPD** | `gemini-2.5-flash-lite` caps at 20 RPD; Groq (14,400 RPD) is the practical high-volume fallback |
| **MemorySaver checkpointer** | In-process only; not durable across restarts (production: use PostgresSaver) |

See [`Docs/PRD_v2.3.md § 15`](Docs/PRD_v2.3.md) for full trade-offs and assumptions.

---

## Contributing

1. Never hardcode policy limits, exclusions, or copay rates — always read from `Docs/policy_terms.json`.
2. Never use an LLM for financial calculations.
3. All monetary arithmetic must use Python `Decimal`.
4. All new nodes must emit `TraceStep` entries.
5. Run `pytest` and `mypy --strict src/` before every commit.

---

*Built with LangGraph · Gemini · Groq · Streamlit · Pydantic V2*
