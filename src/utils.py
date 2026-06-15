import re
from typing import Optional, Dict, List
import cv2
import numpy as np

DIAGNOSIS_NORMALIZATION: Dict[str, List[str]] = {
    # Waiting period keys
    "diabetes": [
        "diabetes", "t2dm", "type 2 diabetes", "type ii diabetes",
        "diabetes mellitus", "type 2 diabetes mellitus", "type-2 diabetes"
    ],
    "hypertension": ["hypertension", "htn", "high blood pressure", "elevated bp"],
    "thyroid_disorders": ["hypothyroidism", "hyperthyroidism", "thyroid", "hashimoto"],
    "joint_replacement": ["knee replacement", "hip replacement", "joint replacement"],
    "maternity": ["pregnancy", "antenatal", "postnatal", "delivery", "maternity", "obstetric"],
    "mental_health": ["depression", "anxiety", "bipolar", "schizophrenia", "ocd", "ptsd"],
    "obesity_treatment": [
        "obesity", "morbid obesity", "bariatric", "weight loss program",
        "bmi > 30", "bmi > 35", "overweight", "bariatric consultation"
    ],
    "hernia": ["hernia", "inguinal hernia", "umbilical hernia"],
    "cataract": ["cataract"],
    # Exclusion triggers
    "cosmetic_dental": ["teeth whitening", "veneers", "bleaching", "orthodontic", "braces", "implants"],
    "cosmetic_vision": ["lasik", "refractive surgery", "laser eye"],
    "cosmetic_general": ["cosmetic surgery", "aesthetic procedure", "plastic surgery"],
    "bariatric_surgery": ["bariatric surgery", "sleeve gastrectomy", "gastric bypass", "lap band"],
}

def assess_blur(image_bytes: bytes) -> float:
    """
    Decodes an image from memory and computes its Laplacian variance.
    Normalizes the variance (variance / 500) and caps at 1.0.
    Returns 0.0 if decoding fails or an error occurs.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var: float = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return min(laplacian_var / 500.0, 1.0)
    except Exception:
        return 0.0

def strip_pii(text: str) -> str:
    """
    Redacts Aadhaar and Indian Phone Numbers in the given text with [REDACTED].
    """
    # Redact Aadhaar: \b\d{4}\s\d{4}\s\d{4}\b or \b\d{12}\b
    aadhaar_pattern = r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"
    # Redact Indian Phone Numbers
    phone_pattern = r"\+91[\s-]?[6-9]\d{9}\b|\b(?:0{0,2}91[\s-]?)?[6-9]\d{9}\b|\b0?[6-9]\d{9}\b"
    
    text = re.sub(phone_pattern, "[REDACTED]", text)
    text = re.sub(aadhaar_pattern, "[REDACTED]", text)
    return text

def normalize_diagnosis(diagnosis: Optional[str]) -> Optional[str]:
    """
    Normalizes the raw diagnosis string by lowercasing it and matching it
    against the values in the DIAGNOSIS_NORMALIZATION table.
    Returns the mapped key, or None if no match is found.
    """
    if not diagnosis:
        return None
    
    diag_lower = diagnosis.strip().lower()
    
    # 1. First check for exact matches
    for key, values in DIAGNOSIS_NORMALIZATION.items():
        if diag_lower in values:
            return key
            
    # 2. Otherwise, check for full-word substring matches, matching longer keywords first
    all_pairs = []
    for key, values in DIAGNOSIS_NORMALIZATION.items():
        for val in values:
            all_pairs.append((val, key))
            
    # Sort by length of keyword in descending order
    all_pairs.sort(key=lambda x: len(x[0]), reverse=True)
    
    for val, key in all_pairs:
        pattern = rf"\b{re.escape(val)}\b"
        if re.search(pattern, diag_lower):
            return key
            
    return None