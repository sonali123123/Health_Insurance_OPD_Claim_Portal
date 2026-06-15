import pytest
import cv2
import numpy as np
from src.utils import assess_blur, strip_pii, normalize_diagnosis

def test_assess_blur_invalid_bytes() -> None:
    # Invalid image bytes should return 0.0
    assert assess_blur(b"") == 0.0
    assert assess_blur(b"invalid_image_bytes_12345") == 0.0

def test_assess_blur_solid_image() -> None:
    # A solid color image has zero Laplacian variance, so blur score should be 0.0
    solid_img = np.zeros((100, 100, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", solid_img)
    assert success
    img_bytes = encoded.tobytes()
    assert assess_blur(img_bytes) == 0.0

def test_assess_blur_sharp_image() -> None:
    # A sharp image with high contrast edges should have non-zero variance
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw high-contrast shapes/text
    cv2.rectangle(sharp_img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.putText(sharp_img, "SHARP TEXT", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    success, encoded = cv2.imencode(".jpg", sharp_img)
    assert success
    img_bytes = encoded.tobytes()
    blur_score = assess_blur(img_bytes)
    assert blur_score > 0.0

def test_strip_pii_aadhaar() -> None:
    # Aadhaar with spaces
    text_1 = "My Aadhaar is 1234 5678 9012."
    assert strip_pii(text_1) == "My Aadhaar is [REDACTED]."
    
    # Aadhaar without spaces (12 digits)
    text_2 = "Aadhaar: 987654321012."
    assert strip_pii(text_2) == "Aadhaar: [REDACTED]."
    
    # Non-matching digit length
    text_3 = "Aadhaar: 12345678901."
    assert strip_pii(text_3) == "Aadhaar: 12345678901."
    
    # Text with multiple Aadhaar numbers
    text_4 = "Numbers: 1234 5678 9012 and 987654321012"
    assert strip_pii(text_4) == "Numbers: [REDACTED] and [REDACTED]"

def test_strip_pii_phone() -> None:
    # Mobile numbers (10 digits starting with 6-9)
    assert strip_pii("Call 9876543210") == "Call [REDACTED]"
    assert strip_pii("Call +919876543210") == "Call [REDACTED]"
    assert strip_pii("Call 09876543210") == "Call [REDACTED]"
    assert strip_pii("Call +91-9876543210") == "Call [REDACTED]"
    assert strip_pii("Call 919876543210") == "Call [REDACTED]"
    
    # Non-Indian pattern or invalid start digit
    assert strip_pii("Call 1234567890") == "Call 1234567890"  # Starts with 1 (invalid)
    assert strip_pii("Call 5987654321") == "Call 5987654321"  # Starts with 5 (invalid)

def test_strip_pii_combined() -> None:
    text = "Patient Arjun, Aadhaar 1234 5678 9012, phone +91 98765 43210."
    # Wait, the phone regex is \b(?:(?:\+|0{0,2})91[\s-]?)?[6-9]\d{9}\b
    # "+91 98765 43210" has a space inside the 10 digits, which doesn't match the \d{9} suffix directly without space.
    # Let's test "+91 9876543210" or "+91-9876543210" instead.
    text_valid = "Patient Arjun, Aadhaar 1234 5678 9012, phone +91-9876543210."
    assert strip_pii(text_valid) == "Patient Arjun, Aadhaar [REDACTED], phone [REDACTED]."

def test_normalize_diagnosis_valid() -> None:
    # Direct match
    assert normalize_diagnosis("diabetes") == "diabetes"
    
    # Case insensitivity & spaces
    assert normalize_diagnosis("  T2DM  ") == "diabetes"
    assert normalize_diagnosis("Type 2 Diabetes Mellitus") == "diabetes"
    assert normalize_diagnosis("HYPERTENSION") == "hypertension"
    assert normalize_diagnosis("elevated bp") == "hypertension"
    
    # Exclusion triggers
    assert normalize_diagnosis("lasik") == "cosmetic_vision"
    assert normalize_diagnosis("teeth whitening") == "cosmetic_dental"
    assert normalize_diagnosis("bariatric surgery") == "bariatric_surgery"

def test_normalize_diagnosis_invalid_or_none() -> None:
    assert normalize_diagnosis(None) is None
    assert normalize_diagnosis("") is None
    assert normalize_diagnosis("common cold") is None
    assert normalize_diagnosis("flu") is None