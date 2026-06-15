"""
Live Multimodal LLM Extraction Pipeline
========================================
Implements Agents 0–2 for real (non-mock) document processing:

  Agent 0 — Document Classifier    : Gemini vision → classify document type
  Agent 1 — Quality Verifier       : OpenCV blur check; LLM for richer flags
  Agent 2 — Fact Extractor         : Gemini/Groq extraction with tool-use logging

Fallback chain (PRD §11.1):
  Primary   : gemini-3.1-flash-lite     (15 RPM / 500 RPD)
  Secondary : gemini-2.5-flash-lite     (10 RPM / 20 RPD)  – on 429 or conf < 0.70
  Tertiary  : groq / llama-4-scout      (14 400 RPD)        – confidence penalty −0.05 base
                                                              – extra −0.05 for bill docs → −0.10

PII is stripped before any bytes/text reach external APIs (Aadhaar, phone).

PDF rasterization uses PyMuPDF (`fitz`): one image per page, first page used for
single-page docs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Literal, cast

# ── Third-party ──────────────────────────────────────────────────────────────
import cv2
import fitz  # type: ignore[import-untyped]  # PyMuPDF
import numpy as np
from pydantic import ValidationError

# ── Google GenAI SDK ──────────────────────────────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ── Groq SDK ─────────────────────────────────────────────────────────────────
try:
    from groq import Groq, RateLimitError as GroqRateLimitError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ── Local models ─────────────────────────────────────────────────────────────
from src.models import (
    DocumentClassification,
    DocumentType,
    FactExtractionPayload,
    ItemizedLine,
    LLMCallTrace,
    QualityResult,
    TraceStep,
)
from src.utils import DIAGNOSIS_NORMALIZATION, assess_blur, normalize_diagnosis, strip_pii

# ─────────────────────────────────────────────────────────────────────────────
# Model identifiers
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_PRIMARY = "gemini-3.1-flash-lite"                       # Primary — 15 RPM / 500 RPD
_MODEL_SECONDARY = "gemini-2.5-flash-lite-preview-06-17"       # Secondary — escalated on 429 or conf < 0.70
_MODEL_GROQ = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_MODEL_LABEL = "groq-llama4-scout"

# ─────────────────────────────────────────────────────────────────────────────
# PII Patterns
# ─────────────────────────────────────────────────────────────────────────────

_AADHAAR_PATTERN = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b")
_PHONE_PATTERN = re.compile(
    r"\+91[\s-]?[6-9]\d{9}\b|(?:0{0,2}91[\s-]?)?[6-9]\d{9}\b|0?[6-9]\d{9}\b"
)


def _redact_pii_text(text: str) -> str:
    """Strip Aadhaar and Indian phone numbers from extracted text."""
    text = _PHONE_PATTERN.sub("[REDACTED]", text)
    text = _AADHAAR_PATTERN.sub("[REDACTED]", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# PDF → Image rasterisation
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[bytes]:
    """
    Convert a PDF (bytes) to a list of PNG image bytes (one per page).
    Uses PyMuPDF (fitz). Returns [] on failure.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images: List[bytes] = []
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))
        doc.close()
        return images
    except Exception:
        return []


def _bytes_to_base64_png(image_bytes: bytes) -> str:
    """Return base64-encoded PNG string (no data-URI prefix)."""
    return base64.b64encode(image_bytes).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_gemini_client() -> Any:
    """Instantiate a Google GenAI client using GOOGLE_API_KEY env var."""
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai SDK not installed.")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.")
    return google_genai.Client(api_key=api_key)


def _call_gemini_vision(
    client: Any,
    model_id: str,
    system_prompt: str,
    image_bytes: bytes,
) -> Tuple[str, bool]:
    """
    Call a Gemini vision model with a single image.
    Returns (raw_text, rate_limited: bool).
    Raises other exceptions to the caller.
    """
    try:
        # Build content with inline image
        img_part = genai_types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )
        text_part = genai_types.Part.from_text(text=system_prompt)
        response = client.models.generate_content(
            model=model_id,
            contents=[text_part, img_part],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        raw = response.text or ""
        raw = strip_pii(raw)
        return raw, False
    except Exception as exc:
        exc_str = str(exc)
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
            return "", True
        raise


def _call_groq_vision(
    image_bytes: bytes,
    prompt: str,
) -> Tuple[str, bool]:
    """
    Call Groq Llama-4-Scout with base64 image.
    Returns (raw_text, rate_limited: bool).
    """
    if not _GROQ_AVAILABLE:
        raise RuntimeError("groq SDK not installed.")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    try:
        client = Groq(api_key=api_key)
        b64 = _bytes_to_base64_png(image_bytes)
        response = client.chat.completions.create(
            model=_MODEL_GROQ,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        raw = strip_pii(raw)
        return raw, False
    except Exception as exc:
        exc_str = str(exc)
        if "429" in exc_str or "rate_limit" in exc_str.lower():
            return "", True
        raise


def _parse_json_from_llm(raw: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output.
    Strips markdown fences if present.
    """
    # Strip markdown code fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    return dict(json.loads(raw))


# ─────────────────────────────────────────────────────────────────────────────
# Image loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_png_bytes(image_bytes: bytes, filename: str = "") -> bytes:
    """
    Accepts raw image bytes (JPEG, PNG) or PDF bytes.
    For PDFs: rasterises the first page.
    For images: re-encodes as PNG via OpenCV.
    Returns PNG bytes or empty bytes on failure.
    """
    fname_lower = filename.lower()
    if fname_lower.endswith(".pdf") or image_bytes[:4] == b"%PDF":
        pages = pdf_to_images(image_bytes)
        return pages[0] if pages else b""

    # Try OpenCV decode
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return b""
        ok, buf = cv2.imencode(".png", img)
        return bytes(buf.tobytes()) if ok else b""
    except Exception:
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# Agent 0 — Document Classifier (live)
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """\
You are a medical document classifier for Indian health insurance claims.
Examine this document image and determine its type.

Indian medical documents include: doctor prescriptions (with Rx symbol, letterhead,
registration number), hospital bills (itemized charges, GST, bill number),
pharmacy bills (drug license number, medicine list with batch numbers),
lab reports (test results, normal ranges, NABL logo), dental reports, and discharge summaries.

Classify as exactly one of:
PRESCRIPTION | HOSPITAL_BILL | PHARMACY_BILL | LAB_REPORT | DISCHARGE_SUMMARY | DENTAL_REPORT | UNKNOWN

Output ONLY valid JSON — no preamble, no backticks:
{
  "classified_type": "<TYPE>",
  "confidence": <0.0–1.0>,
  "signals": ["<observed signal 1>", "<observed signal 2>"],
  "patient_name_visible": "<name if legible, else null>"
}"""


def classify_document_live(
    file_id: str,
    image_bytes: bytes,
    filename: str = "",
) -> Tuple[DocumentClassification, LLMCallTrace, List[str]]:
    """
    Classify a document image using the live LLM fallback chain.

    Returns:
        (DocumentClassification, LLMCallTrace, warnings)
    """
    png_bytes = _ensure_png_bytes(image_bytes, filename)
    if not png_bytes:
        return (
            DocumentClassification(
                file_id=file_id,
                classified_type=DocumentType.UNKNOWN,
                confidence=0.0,
                signals=["Failed to decode image"],
            ),
            LLMCallTrace(
                model_used="none",
                prompt_summary=_CLASSIFIER_PROMPT[:200],
                raw_response_preview="Image decode failed",
                parse_success=False,
                fallback_triggered=False,
                tool_calls=[],
            ),
            ["Image could not be decoded — returned UNKNOWN"],
        )

    warnings: List[str] = []
    client = _get_gemini_client()

    # ── Try primary model ──────────────────────────────────────────────────
    raw, rate_limited = _call_gemini_vision(client, _MODEL_PRIMARY, _CLASSIFIER_PROMPT, png_bytes)
    model_used = _MODEL_PRIMARY
    fallback_triggered = False

    if rate_limited or not raw:
        # ── Try secondary model ────────────────────────────────────────────
        raw, rate_limited2 = _call_gemini_vision(client, _MODEL_SECONDARY, _CLASSIFIER_PROMPT, png_bytes)
        model_used = _MODEL_SECONDARY
        fallback_triggered = True
        warnings.append(f"Primary model rate-limited; escalated to {_MODEL_SECONDARY}")

        if rate_limited2 or not raw:
            # ── Try Groq ───────────────────────────────────────────────────
            raw, _ = _call_groq_vision(png_bytes, _CLASSIFIER_PROMPT)
            model_used = _GROQ_MODEL_LABEL
            warnings.append(f"Secondary model rate-limited; escalated to {_GROQ_MODEL_LABEL}")

    # Parse JSON
    try:
        data = _parse_json_from_llm(raw)
        classified_type_str = str(data.get("classified_type", "UNKNOWN")).upper()
        try:
            classified_type = DocumentType(classified_type_str)
        except ValueError:
            classified_type = DocumentType.UNKNOWN

        confidence = float(data.get("confidence", 0.5))
        signals = [str(s) for s in data.get("signals", [])]
        patient_name = data.get("patient_name_visible") or None
        parse_success = True
    except Exception:
        classified_type = DocumentType.UNKNOWN
        confidence = 0.0
        signals = ["JSON parse error"]
        patient_name = None
        parse_success = False
        warnings.append("Classifier JSON parse failed — returned UNKNOWN")

    classification = DocumentClassification(
        file_id=file_id,
        classified_type=classified_type,
        confidence=confidence,
        signals=signals,
        patient_name_visible=patient_name,
    )
    llm_trace = LLMCallTrace(
        model_used=model_used,
        prompt_summary=_CLASSIFIER_PROMPT[:200],
        raw_response_preview=raw[:500],
        parse_success=parse_success,
        fallback_triggered=fallback_triggered,
        tool_calls=[],
    )
    return classification, llm_trace, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Quality Verifier (live, OpenCV first)
# ─────────────────────────────────────────────────────────────────────────────

_QUALITY_PROMPT = """\
You are a document quality inspector for insurance claims processing.
Examine this medical document image.

Check for:
1. Text legibility — can critical fields be read?
2. Official markers — stamp, signature, letterhead
3. Completeness — is the document cut off or partially visible?
4. Distortion — blur, glare, skew, low contrast

Output ONLY valid JSON:
{
  "readable": true | false,
  "readability_score": <0.0–1.0>,
  "quality_flags": ["RUBBER_STAMP_DETECTED" | "PARTIAL_PAGE" | "HEAVY_BLUR" | "LOW_CONTRAST" | "HANDWRITTEN"],
  "unreadable_fields": ["<field name>"],
  "recommendation": "PROCEED" | "REQUEST_REUPLOAD" | "PROCEED_WITH_WARNING"
}"""


def verify_quality_live(
    file_id: str,
    image_bytes: bytes,
    filename: str = "",
) -> Tuple[QualityResult, LLMCallTrace, List[str]]:
    """
    Run quality check using OpenCV blur detection first, then LLM for richer flags.

    Returns:
        (QualityResult, LLMCallTrace, warnings)
    """
    png_bytes = _ensure_png_bytes(image_bytes, filename)

    # ── OpenCV blur check ──────────────────────────────────────────────────
    blur_score = assess_blur(png_bytes) if png_bytes else 0.0

    if blur_score < 0.15:
        # Fail fast — skip LLM
        return (
            QualityResult(
                file_id=file_id,
                readable=False,
                readability_score=blur_score,
                quality_flags=["HEAVY_BLUR"],
                unreadable_fields=["all"],
                recommendation="REQUEST_REUPLOAD",
            ),
            LLMCallTrace(
                model_used="opencv-laplacian",
                prompt_summary="OpenCV blur check (Laplacian variance < 75)",
                raw_response_preview=f"blur_score={blur_score:.4f}",
                parse_success=True,
                fallback_triggered=False,
                tool_calls=[],
            ),
            [],
        )

    warnings: List[str] = []

    if not png_bytes:
        return (
            QualityResult(
                file_id=file_id,
                readable=True,
                readability_score=blur_score,
                quality_flags=[],
                unreadable_fields=[],
                recommendation="PROCEED",
            ),
            LLMCallTrace(
                model_used="opencv-only",
                prompt_summary="OpenCV blur check only",
                raw_response_preview=f"blur_score={blur_score:.4f}",
                parse_success=True,
                fallback_triggered=False,
                tool_calls=[],
            ),
            ["Could not produce PNG — skipping LLM quality check"],
        )

    client = _get_gemini_client()
    raw, rate_limited = _call_gemini_vision(client, _MODEL_PRIMARY, _QUALITY_PROMPT, png_bytes)
    model_used = _MODEL_PRIMARY
    fallback_triggered = False

    if rate_limited or not raw:
        raw, rate_limited2 = _call_gemini_vision(client, _MODEL_SECONDARY, _QUALITY_PROMPT, png_bytes)
        model_used = _MODEL_SECONDARY
        fallback_triggered = True
        warnings.append(f"Quality verifier escalated to {_MODEL_SECONDARY}")

        if rate_limited2 or not raw:
            raw, _ = _call_groq_vision(png_bytes, _QUALITY_PROMPT)
            model_used = _GROQ_MODEL_LABEL
            warnings.append(f"Quality verifier escalated to {_GROQ_MODEL_LABEL}")

    try:
        data = _parse_json_from_llm(raw)
        readable = bool(data.get("readable", True))
        readability_score = float(data.get("readability_score", blur_score))
        quality_flags = [str(f) for f in data.get("quality_flags", [])]
        unreadable_fields = [str(f) for f in data.get("unreadable_fields", [])]
        recommendation_raw = str(data.get("recommendation", "PROCEED"))
        if recommendation_raw not in ("PROCEED", "REQUEST_REUPLOAD", "PROCEED_WITH_WARNING"):
            recommendation_raw = "PROCEED"
        recommendation = cast(Literal["PROCEED", "REQUEST_REUPLOAD", "PROCEED_WITH_WARNING"], recommendation_raw)
        parse_success = True
    except Exception:
        readable = True
        readability_score = blur_score
        quality_flags = []
        unreadable_fields = []
        recommendation = "PROCEED_WITH_WARNING"
        parse_success = False
        warnings.append("Quality verifier JSON parse failed — defaulting to PROCEED_WITH_WARNING")

    return (
        QualityResult(
            file_id=file_id,
            readable=readable,
            readability_score=readability_score,
            quality_flags=quality_flags,
            unreadable_fields=unreadable_fields,
            recommendation=recommendation,
        ),
        LLMCallTrace(
            model_used=model_used,
            prompt_summary=_QUALITY_PROMPT[:200],
            raw_response_preview=raw[:500],
            parse_success=parse_success,
            fallback_triggered=fallback_triggered,
            tool_calls=[],
        ),
        warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Fact Extractor (live)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTOR_PROMPT_TEMPLATE = """\
You are a high-precision medical billing extraction engine for Indian health insurance.
Document type: {document_type}

Extract the following fields:

For PRESCRIPTION:
- doctor_name, doctor_registration (format: STATE/NNNNN/YYYY e.g. KA/45678/2015)
- patient_name, diagnosis (exact text as written — do NOT normalize or interpret)
- medicines (list), tests_ordered (list)

For HOSPITAL_BILL / PHARMACY_BILL:
- hospital_name (or pharmacy_name), patient_name, bill_date (YYYY-MM-DD)
- line_items: [{{"description": str, "amount": float}}]
- bill_total

Rules:
- Do NOT infer or hallucinate missing values. Use null for missing fields.
- Do NOT perform calculations.
- Extract diagnosis EXACTLY as written.
- If a field is partially obscured by a stamp, note it in quality_flags.

Output ONLY valid JSON conforming to this schema:
{{
  "doctor_name": null,
  "doctor_registration": null,
  "diagnosis": null,
  "medicines": [],
  "tests_ordered": [],
  "hospital_name": null,
  "patient_name": null,
  "bill_date": null,
  "line_items": [],
  "bill_total": null,
  "extraction_confidence": <0.0-1.0>,
  "quality_flags": []
}}"""


def _is_bill_document(doc_type: DocumentType) -> bool:
    return doc_type in (DocumentType.HOSPITAL_BILL, DocumentType.PHARMACY_BILL)


def extract_facts_live(
    file_id: str,
    document_type: DocumentType,
    image_bytes: bytes,
    filename: str = "",
    quality_result: Optional[QualityResult] = None,
) -> Tuple[FactExtractionPayload, LLMCallTrace, List[str], float]:
    """
    Extract structured facts from a document image.

    Returns:
        (FactExtractionPayload, LLMCallTrace, warnings, confidence_penalty)
        confidence_penalty is 0.0 normally, −0.05 for Groq, −0.10 for Groq + bill doc.
    """
    prompt = _EXTRACTOR_PROMPT_TEMPLATE.format(document_type=document_type.value)
    png_bytes = _ensure_png_bytes(image_bytes, filename)
    warnings: List[str] = []
    confidence_penalty = 0.0

    if not png_bytes:
        return (
            FactExtractionPayload(
                file_id=file_id,
                document_type=document_type,
                readability_score=0.0,
                extraction_confidence=0.0,
                quality_flags=["IMAGE_DECODE_FAILED"],
            ),
            LLMCallTrace(
                model_used="none",
                prompt_summary=prompt[:200],
                raw_response_preview="Image decode failed",
                parse_success=False,
                fallback_triggered=False,
                tool_calls=[],
            ),
            ["Image could not be decoded for extraction"],
            0.0,
        )

    client = _get_gemini_client()
    raw, rate_limited = _call_gemini_vision(client, _MODEL_PRIMARY, prompt, png_bytes)
    model_used = _MODEL_PRIMARY
    fallback_triggered = False
    tool_calls: List[str] = []

    if rate_limited or not raw:
        raw, rate_limited2 = _call_gemini_vision(client, _MODEL_SECONDARY, prompt, png_bytes)
        model_used = _MODEL_SECONDARY
        fallback_triggered = True
        warnings.append(f"Extraction escalated to {_MODEL_SECONDARY}")

        if rate_limited2 or not raw:
            raw, _ = _call_groq_vision(png_bytes, prompt)
            model_used = _GROQ_MODEL_LABEL
            warnings.append(f"Extraction escalated to {_GROQ_MODEL_LABEL}")
            # Apply confidence penalty per PRD §11.1
            confidence_penalty = -0.10 if _is_bill_document(document_type) else -0.05

    # Parse extracted fields
    try:
        data = _parse_json_from_llm(raw)
        extraction_confidence = float(data.get("extraction_confidence", 0.5))
        parse_success = True
    except Exception:
        data = {}
        extraction_confidence = 0.0
        parse_success = False
        warnings.append("Extraction JSON parse failed — low confidence result")

    # If confidence is low after primary/secondary, call re-extraction tool
    if extraction_confidence < 0.50 and model_used != _GROQ_MODEL_LABEL:
        tool_calls.append("request_reextraction")
        warnings.append(
            f"Low extraction confidence ({extraction_confidence:.2f}) — "
            f"request_reextraction tool invoked for {file_id}"
        )
        # Re-try with Groq as escalation
        raw2, _ = _call_groq_vision(png_bytes, prompt)
        if raw2:
            try:
                data2 = _parse_json_from_llm(raw2)
                extraction_confidence2 = float(data2.get("extraction_confidence", 0.0))
                if extraction_confidence2 > extraction_confidence:
                    data = data2
                    extraction_confidence = extraction_confidence2
                    model_used = _GROQ_MODEL_LABEL
                    fallback_triggered = True
                    confidence_penalty = -0.10 if _is_bill_document(document_type) else -0.05
            except Exception:
                pass

    # Build line items
    line_items: Optional[List[ItemizedLine]] = None
    raw_items = data.get("line_items") or []
    if raw_items and isinstance(raw_items, list):
        built: List[ItemizedLine] = []
        for i, item in enumerate(raw_items):
            if isinstance(item, dict):
                try:
                    built.append(
                        ItemizedLine(
                            description=_redact_pii_text(str(item.get("description", ""))),
                            amount=Decimal(str(item.get("amount", 0))),
                            status="APPROVED",
                        )
                    )
                except Exception:
                    pass
        line_items = built if built else None

    # Parse bill_date
    bill_date: Optional[date] = None
    raw_date = data.get("bill_date")
    if raw_date:
        try:
            bill_date = datetime.strptime(str(raw_date), "%Y-%m-%d").date()
        except ValueError:
            pass

    # Parse bill_total
    bill_total: Optional[Decimal] = None
    raw_total = data.get("bill_total")
    if raw_total is not None:
        try:
            bill_total = Decimal(str(raw_total))
        except Exception:
            pass

    readability_score = float(quality_result.readability_score if quality_result else 0.95)

    diagnosis_raw = data.get("diagnosis")
    diagnosis_normalized = normalize_diagnosis(diagnosis_raw) if diagnosis_raw else None

    quality_flags = [str(f) for f in (data.get("quality_flags") or [])]

    payload = FactExtractionPayload(
        file_id=file_id,
        document_type=document_type,
        doctor_name=_redact_pii_text(str(data["doctor_name"])) if data.get("doctor_name") else None,
        doctor_registration=str(data["doctor_registration"]) if data.get("doctor_registration") else None,
        diagnosis=diagnosis_raw,
        diagnosis_normalized=diagnosis_normalized,
        medicines=([str(m) for m in data["medicines"]] if data.get("medicines") else None),
        tests_ordered=([str(t) for t in data["tests_ordered"]] if data.get("tests_ordered") else None),
        hospital_name=_redact_pii_text(str(data["hospital_name"])) if data.get("hospital_name") else None,
        patient_name=_redact_pii_text(str(data["patient_name"])) if data.get("patient_name") else None,
        bill_date=bill_date,
        line_items=line_items,
        bill_total=bill_total,
        readability_score=readability_score,
        extraction_confidence=extraction_confidence,
        quality_flags=quality_flags,
    )

    llm_trace = LLMCallTrace(
        model_used=model_used,
        prompt_summary=prompt[:200],
        raw_response_preview=raw[:500],
        parse_success=parse_success,
        fallback_triggered=fallback_triggered,
        tool_calls=tool_calls,
    )
    return payload, llm_trace, warnings, confidence_penalty