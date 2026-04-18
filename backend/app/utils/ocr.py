"""
utils/ocr.py — Extract lab values from uploaded PDF or image files.
Uses Tesseract OCR with OpenCV preprocessing for better accuracy.

FIXES vs previous version:
  FIX 1 — VALUE_PATTERN now strips commas from numbers before parsing,
           so "9,200" → 9200.0 and "310,000" → 310000.0 instead of 9.0 / 310.0
  FIX 2 — Added more aliases for triglycerides, platelet count, WBC,
           and other test names that appear in different formats on lab reports
  FIX 3 — Added RBC, hematocrit, TIBC aliases
  FIX 4 — Value extraction now also handles values that appear BEFORE
           the unit on the same line (e.g. "Hemoglobin  10.8  g/dL")
"""
import re
import io
from typing import Dict, Tuple
import structlog

log = structlog.get_logger()

# ── Test name aliases ──────────────────────────────────────────────────────────
# Maps every variation OCR might produce → canonical key used by lab_rules.py
# Sorted by length (longest first) so more specific matches win.

TEST_NAME_ALIASES: Dict[str, str] = {
    # ── Glucose ────────────────────────────────────────────────────────────────
    "fasting blood glucose":          "fasting_glucose",
    "fasting plasma glucose":         "fasting_glucose",
    "fasting blood sugar":            "fasting_glucose",
    "fasting glucose":                "fasting_glucose",
    "blood sugar fasting":            "fasting_glucose",
    "fbg":                            "fasting_glucose",
    "fpg":                            "fasting_glucose",
    "random blood glucose":           "random_glucose",
    "random blood sugar":             "random_glucose",
    "random glucose":                 "random_glucose",
    "rbg":                            "random_glucose",
    "hba1c (glycated hemoglobin)":    "hba1c",
    "glycated hemoglobin":            "hba1c",
    "glycosylated hemoglobin":        "hba1c",
    "hemoglobin a1c":                 "hba1c",
    "hba1c":                          "hba1c",
    "a1c":                            "hba1c",

    # ── Blood Pressure ─────────────────────────────────────────────────────────
    "systolic blood pressure":        "systolic_bp",
    "systolic bp":                    "systolic_bp",
    "systolic":                       "systolic_bp",
    "diastolic blood pressure":       "diastolic_bp",
    "diastolic bp":                   "diastolic_bp",
    "diastolic":                      "diastolic_bp",
    "heart rate":                     "heart_rate",
    "pulse rate":                     "heart_rate",
    "pulse":                          "heart_rate",

    # ── CBC — Hemoglobin ───────────────────────────────────────────────────────
    "hemoglobin (hb)":                "hemoglobin",
    "haemoglobin (hb)":               "hemoglobin",
    "hemoglobin":                     "hemoglobin",
    "haemoglobin":                    "hemoglobin",
    "hgb":                            "hemoglobin",
    "hb":                             "hemoglobin",

    # ── CBC — Indices ──────────────────────────────────────────────────────────
    "mean corpuscular volume":        "mcv",
    "mean cell volume":               "mcv",
    "mcv":                            "mcv",
    "mean corpuscular hemoglobin concentration": "mchc",
    "mean corpuscular hemoglobin":    "mch",
    "mean cell hemoglobin":           "mch",
    "mchc":                           "mchc",
    "mch":                            "mch",

    # ── CBC — Counts ───────────────────────────────────────────────────────────
    "white blood cell count":         "wbc",
    "white blood cells":              "wbc",
    "white blood count":              "wbc",
    "wbc count":                      "wbc",
    "wbc":                            "wbc",
    "total leukocyte count":          "wbc",
    "tlc":                            "wbc",
    "rbc count":                      "rbc",
    "red blood cell count":           "rbc",
    "red blood cells":                "rbc",
    "rbc":                            "rbc",
    "platelet count":                 "platelets",
    "platelets":                      "platelets",
    "plt":                            "platelets",
    "thrombocytes":                   "platelets",
    "hematocrit":                     "hematocrit",
    "haematocrit":                    "hematocrit",
    "packed cell volume":             "hematocrit",
    "pcv":                            "hematocrit",

    # ── CBC — Differential ─────────────────────────────────────────────────────
    "neutrophils":                    "neutrophils",
    "lymphocytes":                    "lymphocytes",
    "monocytes":                      "monocytes",
    "eosinophils":                    "eosinophils",
    "basophils":                      "basophils",

    # ── Lipids ─────────────────────────────────────────────────────────────────
    "total cholesterol":              "total_cholesterol",
    "serum cholesterol":              "total_cholesterol",
    "cholesterol":                    "total_cholesterol",
    "ldl cholesterol":                "ldl",
    "ldl-c":                          "ldl",
    "ldl":                            "ldl",
    "hdl cholesterol":                "hdl",
    "hdl-c":                          "hdl",
    "hdl":                            "hdl",
    "triglycerides":                  "triglycerides",
    "triglyceride":                   "triglycerides",
    "tg":                             "triglycerides",
    "serum triglycerides":            "triglycerides",
    "vldl":                           "vldl",

    # ── Kidney / Liver ─────────────────────────────────────────────────────────
    "serum creatinine":               "creatinine",
    "creatinine":                     "creatinine",
    "blood urea nitrogen":            "bun",
    "blood urea":                     "bun",
    "urea nitrogen":                  "bun",
    "bun":                            "bun",
    "uric acid":                      "uric_acid",
    "alanine aminotransferase":       "alt",
    "alt (sgpt)":                     "alt",
    "sgpt":                           "alt",
    "alt":                            "alt",
    "aspartate aminotransferase":     "ast",
    "ast (sgot)":                     "ast",
    "sgot":                           "ast",
    "ast":                            "ast",
    "alkaline phosphatase":           "alp",
    "alp":                            "alp",
    "total bilirubin":                "total_bilirubin",
    "bilirubin":                      "total_bilirubin",
    "albumin":                        "albumin",
    "total protein":                  "total_protein",

    # ── Thyroid ────────────────────────────────────────────────────────────────
    "thyroid stimulating hormone":    "tsh",
    "tsh":                            "tsh",
    "free t4":                        "free_t4",
    "free t3":                        "free_t3",
    "ft4":                            "free_t4",
    "ft3":                            "free_t3",

    # ── Iron Studies ───────────────────────────────────────────────────────────
    "serum iron":                     "serum_iron",
    "iron":                           "serum_iron",
    "serum ferritin":                 "ferritin",
    "ferritin":                       "ferritin",
    "total iron binding capacity":    "tibc",
    "tibc":                           "tibc",
    "transferrin saturation":         "transferrin_saturation",

    # ── Vitamins / Others ──────────────────────────────────────────────────────
    "vitamin b12":                    "vitamin_b12",
    "vitamin d":                      "vitamin_d",
    "25-hydroxyvitamin d":            "vitamin_d",
    "folate":                         "folate",
    "folic acid":                     "folate",
    "calcium":                        "calcium",
    "magnesium":                      "magnesium",
    "sodium":                         "sodium",
    "potassium":                      "potassium",
    "chloride":                       "chloride",
}

# ── Value extraction regex ─────────────────────────────────────────────────────
# FIX 1: Now matches numbers WITH commas (e.g. 9,200  310,000  1,50,000)
# The comma-containing digits are captured in group 1 and cleaned before float()

VALUE_PATTERN = re.compile(
    r"([\d]{1,3}(?:,[\d]{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)"   # number with optional commas
    r"\s*"
    r"(?:mg/dl|mg/l|g/dl|mmhg|%|iu/l|u/l|pg|fl|ng/ml|"
    r"µg/dl|ug/dl|miu/l|/µl|/ul|meq/l|mmol/l|nmol/l|pmol/l|"
    r"ng/dl|pg/ml|iu/ml|10\^3|10\^6|cells/µl|cells/ul)?",
    re.IGNORECASE,
)


def _parse_number(raw: str) -> float:
    """
    Convert a raw matched number string to float.
    Handles comma-formatted numbers: '9,200' → 9200.0, '3,10,000' → 310000.0
    """
    return float(raw.replace(",", ""))


def extract_lab_values_from_text(text: str) -> Dict[str, float]:
    """
    Parse OCR'd or natively-extracted PDF text and return
    a dict of canonical_test_name → float_value.

    Strategy:
    1. For each line, try to match the longest alias first.
    2. Once a test name is found, look for a number in the remainder of the line.
    3. Strip commas from numbers before converting to float (FIX 1).
    4. Skip implausible values (e.g. 0.0, negative numbers).
    """
    results: Dict[str, float] = {}
    lines = text.lower().splitlines()

    # Sort aliases longest-first so "fasting blood glucose" matches before "glucose"
    sorted_aliases = sorted(TEST_NAME_ALIASES.items(), key=lambda x: -len(x[0]))

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        for alias, canonical in sorted_aliases:
            if alias not in line:
                continue

            # Found a test name — extract value from rest of line
            idx = line.index(alias)
            remainder = line[idx + len(alias):]

            val_match = VALUE_PATTERN.search(remainder)
            if val_match:
                raw = val_match.group(1)
                try:
                    value = _parse_number(raw)
                    # Sanity check — skip clearly implausible values
                    if value <= 0:
                        break
                    # Skip percentage differentials that are already captured elsewhere
                    # (e.g. neutrophils 62% on a CBC — skip if we already have this key)
                    if canonical in results:
                        break
                    results[canonical] = value
                except ValueError:
                    pass
            break  # stop checking aliases once one matched on this line

    log.info(
        "ocr.values_extracted",
        count=len(results),
        keys=list(results.keys()),
    )
    return results


def extract_from_image_bytes(image_bytes: bytes) -> Tuple[str, Dict[str, float]]:
    """
    Run OCR on raw image bytes (PNG, JPEG).
    Returns (raw_text, extracted_values).
    """
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        img_gray = img.convert("L")

        # PSM 6 = assume uniform block of text (good for lab reports)
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
    Extract text from PDF bytes.
    Tries native text extraction first (fast, accurate for typed PDFs),
    falls back to Tesseract OCR for scanned/image PDFs.
    Returns (raw_text, extracted_values).
    """
    raw_text = ""

    # ── Attempt 1: native text extraction via pdfplumber ──────────────────────
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:5]:          # max 5 pages
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"

        if len(raw_text.strip()) > 50:
            values = extract_lab_values_from_text(raw_text)
            log.info(
                "ocr.pdf_native_extraction_success",
                chars=len(raw_text),
                values_found=len(values),
            )
            return raw_text, values

    except Exception as e:
        log.warning("ocr.pdf_native_failed", error=str(e))

    # ── Attempt 2: OCR via pdf2image + Tesseract ──────────────────────────────
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)
        all_text = ""
        for img in images:
            img_gray = img.convert("L")
            page_text = pytesseract.image_to_string(img_gray, config="--psm 6")
            all_text += page_text + "\n"

        values = extract_lab_values_from_text(all_text)
        log.info(
            "ocr.pdf_ocr_success",
            chars=len(all_text),
            values_found=len(values),
        )
        return all_text, values

    except Exception as e:
        log.error("ocr.pdf_ocr_failed", error=str(e))
        return "", {}