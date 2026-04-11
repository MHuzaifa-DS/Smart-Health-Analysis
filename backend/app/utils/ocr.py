"""
utils/ocr.py — Extract lab values from uploaded PDF or image files.
Uses Tesseract OCR with OpenCV preprocessing for better accuracy.
"""
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import structlog

log = structlog.get_logger()

# Maps common OCR'd test names → our canonical key names
TEST_NAME_ALIASES: Dict[str, str] = {
    # Glucose
    "fasting blood glucose":      "fasting_glucose",
    "fasting glucose":            "fasting_glucose",
    "fbg":                        "fasting_glucose",
    "blood sugar fasting":        "fasting_glucose",
    "random blood glucose":       "random_glucose",
    "random glucose":             "random_glucose",
    "rbg":                        "random_glucose",
    "hba1c":                      "hba1c",
    "glycated hemoglobin":        "hba1c",
    "a1c":                        "hba1c",
    # Blood pressure
    "systolic":                   "systolic_bp",
    "systolic bp":                "systolic_bp",
    "systolic blood pressure":    "systolic_bp",
    "diastolic":                  "diastolic_bp",
    "diastolic bp":               "diastolic_bp",
    "diastolic blood pressure":   "diastolic_bp",
    # Hemoglobin / CBC
    "hemoglobin":                 "hemoglobin",
    "hgb":                        "hemoglobin",
    "hb":                         "hemoglobin",
    "haemoglobin":                "hemoglobin",
    "mcv":                        "mcv",
    "mean corpuscular volume":    "mcv",
    "mch":                        "mch",
    "mean corpuscular hemoglobin":"mch",
    "mchc":                       "mchc",
    "wbc":                        "wbc",
    "white blood cells":          "wbc",
    "white blood count":          "wbc",
    "platelets":                  "platelets",
    "platelet count":             "platelets",
    # Lipids
    "total cholesterol":          "total_cholesterol",
    "cholesterol":                "total_cholesterol",
    "ldl":                        "ldl",
    "ldl cholesterol":            "ldl",
    "hdl":                        "hdl",
    "hdl cholesterol":            "hdl",
    "triglycerides":              "triglycerides",
    # Kidney / Liver
    "creatinine":                 "creatinine",
    "serum creatinine":           "creatinine",
    "bun":                        "bun",
    "blood urea nitrogen":        "bun",
    "alt":                        "alt",
    "sgpt":                       "alt",
    "ast":                        "ast",
    "sgot":                       "ast",
    # Thyroid
    "tsh":                        "tsh",
    "thyroid stimulating hormone":"tsh",
    # Iron
    "serum iron":                 "serum_iron",
    "ferritin":                   "ferritin",
    "serum ferritin":             "ferritin",
}

# Regex to extract numeric value after test name
VALUE_PATTERN = re.compile(
    r"([\d]+\.?\d*)\s*(?:mg/dl|mg/l|g/dl|mmhg|%|iu/l|u/l|pg|fl|ng/ml|µg/dl|miu/l|/µl|/ul)?",
    re.IGNORECASE,
)


def extract_lab_values_from_text(text: str) -> Dict[str, float]:
    """
    Parse OCR'd text and extract lab test name → numeric value pairs.
    """
    results: Dict[str, float] = {}
    lines = text.lower().splitlines()

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Try to match a known test name
        matched_key = None
        for alias, canonical in sorted(TEST_NAME_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in line:
                matched_key = canonical
                # Extract the number after the alias
                remainder = line[line.index(alias) + len(alias):]
                val_match = VALUE_PATTERN.search(remainder)
                if val_match:
                    try:
                        value = float(val_match.group(1))
                        results[canonical] = value
                    except ValueError:
                        pass
                break

    log.info("ocr.values_extracted", count=len(results), keys=list(results.keys()))
    return results


def extract_from_image_bytes(image_bytes: bytes) -> Tuple[str, Dict[str, float]]:
    """
    Run OCR on raw image bytes (PNG, JPEG).
    Returns (raw_text, extracted_values).
    """
    try:
        from PIL import Image
        import pytesseract
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes))

        # Preprocess: convert to grayscale for better OCR
        img_gray = img.convert("L")

        # Run Tesseract
        raw_text = pytesseract.image_to_string(img_gray, config="--psm 6")
        values = extract_lab_values_from_text(raw_text)
        return raw_text, values

    except ImportError:
        log.error("ocr.tesseract_not_available")
        return "", {}
    except Exception as e:
        log.error("ocr.image_extraction_failed", error=str(e))
        return "", {}


def extract_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, Dict[str, float]]:
    """
    Extract text from PDF bytes (first tries native text extraction, then OCR).
    Returns (raw_text, extracted_values).
    """
    raw_text = ""

    # Try native text extraction first (much faster than OCR)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:5]:  # read max 5 pages
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"

        if len(raw_text.strip()) > 50:
            values = extract_lab_values_from_text(raw_text)
            log.info("ocr.pdf_native_extraction_success", chars=len(raw_text))
            return raw_text, values
    except Exception as e:
        log.warning("ocr.pdf_native_failed", error=str(e))

    # Fallback to OCR via pdf2image
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)
        all_text = ""
        for img in images:
            import pytesseract
            img_gray = img.convert("L")
            page_text = pytesseract.image_to_string(img_gray, config="--psm 6")
            all_text += page_text + "\n"
        values = extract_lab_values_from_text(all_text)
        log.info("ocr.pdf_ocr_success", chars=len(all_text))
        return all_text, values
    except Exception as e:
        log.error("ocr.pdf_ocr_failed", error=str(e))
        return "", {}
