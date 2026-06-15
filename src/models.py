from datetime import date
from decimal import Decimal
from enum import Enum
import operator
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field

class DocumentType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    PHARMACY_BILL = "PHARMACY_BILL"
    LAB_REPORT = "LAB_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    DENTAL_REPORT = "DENTAL_REPORT"
    UNKNOWN = "UNKNOWN"

class RejectionReason(str, Enum):
    POLICY_INACTIVE = "POLICY_INACTIVE"
    MEMBER_NOT_FOUND = "MEMBER_NOT_FOUND"
    MEMBER_NOT_COVERED = "MEMBER_NOT_COVERED"
    SUBMISSION_DEADLINE_EXCEEDED = "SUBMISSION_DEADLINE_EXCEEDED"
    MINIMUM_AMOUNT_NOT_MET = "MINIMUM_AMOUNT_NOT_MET"
    INITIAL_WAITING_PERIOD = "INITIAL_WAITING_PERIOD"
    WAITING_PERIOD = "WAITING_PERIOD"
    EXCLUDED_CONDITION = "EXCLUDED_CONDITION"
    PRE_AUTH_MISSING = "PRE_AUTH_MISSING"
    PER_CLAIM_EXCEEDED = "PER_CLAIM_EXCEEDED"
    ANNUAL_LIMIT_EXHAUSTED = "ANNUAL_LIMIT_EXHAUSTED"
    LINE_ITEM_EXCLUDED = "LINE_ITEM_EXCLUDED"
    DOCUMENT_TYPE_MISMATCH = "DOCUMENT_TYPE_MISMATCH"
    DOCUMENT_UNREADABLE = "DOCUMENT_UNREADABLE"
    PATIENT_NAME_MISMATCH = "PATIENT_NAME_MISMATCH"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"

ClaimCategory = Literal[
    "CONSULTATION", "DIAGNOSTIC", "PHARMACY",
    "DENTAL", "VISION", "ALTERNATIVE_MEDICINE"
]

class LLMCallTrace(BaseModel):
    """Embedded in TraceStep for LLM nodes only."""
    model_used: str             # "gemini-3.1-flash-lite" | "gemini-2.5-flash-lite" | "groq-llama4-scout"
    prompt_summary: str         # First 200 chars of system prompt
    raw_response_preview: str   # First 500 chars of verbatim LLM output before parsing
    parse_success: bool         # Did Pydantic validation pass on first attempt?
    fallback_triggered: bool    # Was a secondary/tertiary model used?
    tool_calls: List[str]       # Names of tools called during this LLM turn

class TraceStep(BaseModel):
    step_id: str
    node: str
    rule_applied: Optional[str] = None
    input_value: Optional[Any] = None
    output_value: Optional[Any] = None
    line_item_ref: Optional[str] = None
    result: Literal["PASSED", "FAILED", "FLAGGED", "SKIPPED", "DEGRADED"]
    confidence_delta: float = 0.0
    llm_trace: Optional[LLMCallTrace] = None
    latency_ms: int = 0
    timestamp: str = ""
    notes: Optional[str] = None

class ItemizedLine(BaseModel):
    description: str
    amount: Decimal
    status: Literal["APPROVED", "EXCLUDED", "CAPPED"] = "APPROVED"
    rejection_reason: Optional[str] = None

class DocumentClassification(BaseModel):
    file_id: str
    classified_type: DocumentType
    confidence: float
    signals: List[str] = Field(default_factory=list)
    patient_name_visible: Optional[str] = None

class DocumentSetValidation(BaseModel):
    valid: bool
    uploaded_types: List[str] = Field(default_factory=list)
    required_types: List[str] = Field(default_factory=list)
    missing_types: List[str] = Field(default_factory=list)
    unreadable_documents: List[str] = Field(default_factory=list)
    patient_names_found: Dict[str, Optional[str]] = Field(default_factory=dict)

class QualityResult(BaseModel):
    file_id: str
    readable: bool
    readability_score: float
    quality_flags: List[str]
    unreadable_fields: List[str]
    recommendation: Literal["PROCEED", "REQUEST_REUPLOAD", "PROCEED_WITH_WARNING"]

class QualityGateResult(BaseModel):
    passed: bool
    unreadable_documents: List[str]
    unreadable_document_types: List[str]

class AggregatedDocumentResults(BaseModel):
    classifications: List[DocumentClassification]
    quality_results: List[QualityResult]
    extractions: List["FactExtractionPayload"]
    failed_subgraphs: List[str]
    merge_warnings: List[str]

class FactExtractionPayload(BaseModel):
    file_id: str
    document_type: DocumentType

    # Prescription fields (populated when document_type == PRESCRIPTION)
    doctor_name: Optional[str] = None
    doctor_registration: Optional[str] = None
    diagnosis: Optional[str] = None
    diagnosis_normalized: Optional[str] = None
    medicines: Optional[List[str]] = None
    tests_ordered: Optional[List[str]] = None

    # Bill fields
    hospital_name: Optional[str] = None
    patient_name: Optional[str] = None
    bill_date: Optional[date] = None
    line_items: Optional[List[ItemizedLine]] = None
    bill_total: Optional[Decimal] = None

    # Quality metadata
    readability_score: float = 0.0
    extraction_confidence: float = 0.0
    quality_flags: List[str] = Field(default_factory=list)

class ConsistencyReport(BaseModel):
    consistent: bool
    name_matches: Dict[str, float] = Field(default_factory=dict)
    mismatched_names: List[Dict[str, Any]] = Field(default_factory=list)
    date_alignment_valid: bool
    cross_document_patient_consistent: bool
    stop_reason: Optional[str] = None

class FraudAssessment(BaseModel):
    fraud_score: float
    flags: List[str] = Field(default_factory=list)
    recommendation: Literal["CLEAR", "MANUAL_REVIEW"]
    triggers: List[str] = Field(default_factory=list)

class ClaimsHistoryEntry(BaseModel):
    claim_id: str
    date: date
    amount: Decimal
    provider: str

class DocumentUpload(BaseModel):
    file_id: str
    file_name: Optional[str] = None
    actual_type: Optional[str] = None
    quality: Optional[str] = None                # "GOOD" | "UNREADABLE" — mock quality signal
    content: Optional[Dict[str, Any]] = None     # pre-extracted mock content (test mode)
    patient_name_on_doc: Optional[str] = None
    image_bytes: Optional[bytes] = Field(default=None, max_length=10 * 1024 * 1024)  # raw file bytes for live LLM extraction

class ClaimInput(BaseModel):
    member_id: str
    policy_id: str
    claim_category: ClaimCategory
    treatment_date: date
    claimed_amount: Decimal
    hospital_name: Optional[str] = None
    pre_auth_reference: Optional[str] = None
    ytd_claims_amount: Optional[Decimal] = None
    claims_history: Optional[List[ClaimsHistoryEntry]] = None
    simulate_component_failure: bool = False
    documents: List[DocumentUpload] = Field(default_factory=list)

class AdjudicationOutput(BaseModel):
    claim_id: str
    decision: Literal["APPROVED", "PARTIAL", "REJECTED", "MANUAL_REVIEW"]
    reason: str
    rejection_reasons: List[RejectionReason] = Field(default_factory=list)
    confidence_score: float

    gross_claimed: Decimal
    network_discount_applied: Decimal = Decimal("0")
    copay_applied: Decimal = Decimal("0")
    approved_amount: Decimal

    line_item_decisions: List[ItemizedLine] = Field(default_factory=list)
    pipeline_warnings: List[str] = Field(default_factory=list)
    audit_trace: List[TraceStep] = Field(default_factory=list)

class EarlyStopResponse(BaseModel):
    claim_id: str
    status: Literal["EARLY_STOP"] = "EARLY_STOP"
    stop_stage: Literal["DOCUMENT_SET_VALIDATION", "QUALITY_CHECK", "CONSISTENCY_CHECK"]
    stop_reason: RejectionReason
    user_message: str
    documents_uploaded: List[DocumentClassification] = Field(default_factory=list)
    documents_required: List[str] = Field(default_factory=list)
    documents_missing: List[str] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    unreadable_documents: List[str] = Field(default_factory=list)

class ClaimResponse(BaseModel):
    claim_id: str
    outcome_type: Literal["EARLY_STOP", "DECISION"]
    early_stop: Optional[EarlyStopResponse] = None
    decision: Optional[AdjudicationOutput] = None

class ClaimsState(TypedDict):
    claim_id: str
    claim_input: ClaimInput
    policy: Dict[str, Any]

    # Document processing
    document_classifications: Annotated[List[DocumentClassification], operator.add]
    quality_results: Annotated[List[QualityResult], operator.add]
    aggregated_results: Optional[AggregatedDocumentResults]
    document_set_validation: Optional[DocumentSetValidation]
    quality_gate_result: Optional[QualityGateResult]
    extractions: Annotated[List[FactExtractionPayload], operator.add]

    # Consistency
    consistency_report: Optional[ConsistencyReport]

    # Adjudication
    adjudication_output: Optional[AdjudicationOutput]
    fraud_assessment: Optional[FraudAssessment]

    # Pipeline state
    pipeline_confidence: float           # starts at 1.0, decremented on failures
    confidence_delta_log: Annotated[List[Dict[str, Any]], operator.add]     # [{event: str, delta: float}] for reconciliation
    failed_components: Annotated[List[str], operator.add]
    simulate_component_failure: bool

    # HITL
    human_override_fields: Optional[Dict[str, Any]]
    corrected_name: Optional[str]

    # Response
    early_stop: Optional[EarlyStopResponse]
    final_response: Optional[ClaimResponse]

    # Trace
    trace: Annotated[List[TraceStep], operator.add]
    errors: Annotated[List[str], operator.add]

    # Bypass
    mock_metadata: Optional[Dict[str, Any]]

    # Execution mode: True = call real LLM APIs; False (default) = mock bypass
    use_live: bool