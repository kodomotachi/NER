"""Streamlit receipt OCR + NER extraction app.

Run:
    streamlit run app.py

Notes:
    - PaddleOCR downloads its OCR model weights on first use.
    - Put your fine-tuned HuggingFace token-classification model in the path
      configured by DEFAULT_MODEL_PATH, or change it in the sidebar.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "receipt_scans.sqlite3"
DEFAULT_MODEL_PATH = APP_DIR / "models" / "deep_ner" / "roberta_token_cls" / "best"
REQUIRED_MODEL_FILES = ("config.json",)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the SQLite table if it does not already exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipt_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                raw_ocr_text TEXT NOT NULL,
                entities_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_scan(raw_text: str, entities: dict[str, Any], db_path: Path = DB_PATH) -> int:
    """Persist one receipt scan and return the inserted row id."""
    scanned_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO receipt_scans (scanned_at, raw_ocr_text, entities_json)
            VALUES (?, ?, ?)
            """,
            (scanned_at, raw_text, json.dumps(entities, ensure_ascii=False)),
        )
        conn.commit()
        return int(cursor.lastrowid)


def load_recent_scans(limit: int = 10, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load recent database rows for display in the app."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT id, scanned_at, entities_json
            FROM receipt_scans
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def clear_scans(db_path: Path = DB_PATH) -> None:
    """Delete all saved scans from the local SQLite database."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM receipt_scans")
        conn.commit()


def image_from_streamlit_upload(uploaded_file: Any) -> Image.Image:
    """Convert Streamlit camera/file input into a PIL image."""
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Apply light preprocessing that usually helps receipt OCR."""
    grayscale = ImageOps.grayscale(image)
    # Upscale small webcam captures. Receipt OCR usually benefits from larger text.
    width, height = grayscale.size
    if max(width, height) < 1600:
        scale = 1600 / max(width, height)
        grayscale = grayscale.resize((int(width * scale), int(height * scale)))
    return grayscale


def extract_text_with_ocr(image: Image.Image) -> str:
    """Extract raw English text from a receipt image with PaddleOCR."""
    import numpy as np

    processed = preprocess_for_ocr(image)
    ocr = load_paddleocr_reader()
    result = run_paddleocr(ocr, np.array(processed.convert("RGB")))
    return "\n".join(extract_paddleocr_lines(result)).strip()


@st.cache_resource(show_spinner="Loading PaddleOCR reader...")
def load_paddleocr_reader():
    """Load PaddleOCR once; the first run may download OCR weights."""
    from paddleocr import PaddleOCR

    init_attempts = [
        {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "engine": "paddle",
        },
        {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": "en",
            "use_angle_cls": True,
            "show_log": False,
        },
        {"lang": "en"},
    ]
    last_error: Exception | None = None
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return PaddleOCR(lang="en")


def run_paddleocr(ocr: Any, image_array: Any) -> Any:
    """Run PaddleOCR across both newer 3.x and older 2.x style APIs."""
    if hasattr(ocr, "predict"):
        return ocr.predict(image_array)
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(image_array, cls=True)
        except TypeError:
            return ocr.ocr(image_array)
    raise RuntimeError("Installed paddleocr package does not expose a supported OCR API.")


def extract_paddleocr_lines(result: Any) -> list[str]:
    """Normalize PaddleOCR 3.x and 2.x outputs into ordered text lines."""
    lines: list[str] = []

    def add_text(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text:
            lines.append(text)

    for page in result or []:
        page_data = page
        if hasattr(page, "res"):
            try:
                page_data = page.res
            except Exception:
                page_data = page
        if hasattr(page, "json"):
            try:
                page_json = page.json
                page_data = page_json() if callable(page_json) else page_json
            except Exception:
                page_data = page
        if isinstance(page_data, dict) and "res" in page_data:
            page_data = page_data["res"]

        # PaddleOCR 3.x returns result objects/dicts with rec_texts.
        if isinstance(page_data, dict) and page_data.get("rec_texts") is not None:
            for text in page_data.get("rec_texts") or []:
                add_text(text)
            continue

        # PaddleOCR 2.x commonly returns [box, (text, confidence)] rows.
        if isinstance(page_data, list):
            for item in page_data:
                if not item:
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    rec = item[1]
                    if isinstance(rec, (list, tuple)) and rec:
                        add_text(rec[0])
                    else:
                        add_text(rec)
                elif isinstance(item, dict):
                    add_text(item.get("text") or item.get("rec_text"))
            continue

        if isinstance(page_data, dict):
            add_text(page_data.get("text") or page_data.get("rec_text"))

    return lines


@st.cache_resource(show_spinner="Loading NER model...")
def load_ner_pipeline(model_path: str):
    """Load a local HuggingFace token-classification model once per session."""
    model_path = str(Path(model_path).expanduser().resolve())
    return pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",
    )


def validate_model_path(model_path: str) -> tuple[bool, str]:
    """Return whether a local HuggingFace model folder looks loadable."""
    path = Path(model_path).expanduser()
    if not path.exists():
        return False, f"Model folder does not exist: {path}"
    if not path.is_dir():
        return False, f"Model path is not a folder: {path}"
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).exists()]
    has_weight = any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "vocab.json", "vocab.txt"))
    if missing:
        return False, f"Model folder is missing required file(s): {', '.join(missing)}"
    if not has_weight:
        return False, "Model folder is missing weights: model.safetensors or pytorch_model.bin"
    if not has_tokenizer:
        return False, "Model folder is missing tokenizer files."
    return True, ""


def normalize_label(label: str) -> str:
    """Normalize common label names from token-classification outputs."""
    label = label.upper().replace("B-", "").replace("I-", "")
    aliases = {
        "COMPANY": "vendor",
        "VENDOR": "vendor",
        "MERCHANT": "vendor",
        "STORE": "vendor",
        "TOTAL": "total",
        "AMOUNT": "total",
        "DATE": "date",
        "INVOICE_DATE": "date",
        "ADDRESS": "address",
    }
    return aliases.get(label, label.lower())


def clean_entity_text(text: str) -> str:
    """Clean HuggingFace subword spacing artifacts."""
    return (
        text.replace(" ##", "")
        .replace("Ġ", " ")
        .replace("▁", " ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" :", ":")
        .strip()
    )


def parse_ner_entities(ner_output: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert pipeline output into a compact receipt entity dictionary."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in ner_output:
        label = normalize_label(str(item.get("entity_group") or item.get("entity") or "unknown"))
        value = clean_entity_text(str(item.get("word", "")))
        if not value:
            continue
        grouped.setdefault(label, []).append(
            {
                "text": value,
                "confidence": round(float(item.get("score", 0.0)), 4),
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )

    structured: dict[str, Any] = {}
    for label, candidates in grouped.items():
        best = max(candidates, key=lambda candidate: candidate["confidence"])
        structured[label] = best["text"]
        structured[f"{label}_confidence"] = best["confidence"]
        structured[f"{label}_candidates"] = candidates

    return structured


def normalize_amount_text(value: str) -> str:
    """Normalize OCR amount strings such as RM32,69 or 32 69 into 32.69."""
    value = value.upper().replace("RM", "").replace("$", "").strip()
    value = re.sub(r"[^\d.,]", "", value)
    if "," in value and "." in value:
        value = value.replace(",", "")
    if "," in value and "." not in value:
        value = value.replace(",", ".")
    return value


def normalize_receipt_text(value: str) -> str:
    """Normalize receipt text for lightweight rule matching."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_receipt_field(value: str) -> str:
    """Remove noisy spacing while preserving receipt punctuation."""
    value = normalize_receipt_text(value)
    value = re.sub(r"\s+([,.:;/])", r"\1", value)
    value = re.sub(r"([,.:;/])(?=\S)", r"\1 ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;:")


def extract_date_from_ocr(raw_text: str) -> str | None:
    """Extract a complete receipt date, preferring dates near a DATE label."""
    patterns = [
        re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b"),
        re.compile(r"\b(\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"),
        re.compile(r"\b(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[A-Z]*\s+\d{2,4})\b", re.IGNORECASE),
    ]
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    candidates: list[tuple[int, int, str]] = []
    for line_index, line in enumerate(lines):
        lower = line.lower()
        score = 4 if "date" in lower or "invoice date" in lower else 0
        for pattern in patterns:
            for match in pattern.finditer(line):
                candidates.append((score, -line_index, match.group(1)))
        if ("date" in lower or "invoice date" in lower) and not any(pattern.search(line) for pattern in patterns):
            for offset, next_line in enumerate(lines[line_index + 1 : line_index + 3], start=1):
                for pattern in patterns:
                    match = pattern.search(next_line)
                    if match:
                        candidates.append((score + 3 - offset, -(line_index + offset), match.group(1)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:2])[2]


def is_address_noise_line(line: str) -> bool:
    """Return True for header/contact/product lines that should not be merchant address."""
    lower = line.lower().strip()
    if not lower:
        return True
    noise_markers = [
        "roc",
        "reg no",
        "company no",
        "gst",
        "sst",
        "tax invoice",
        "invoice",
        "receipt",
        "cash bill",
        "tel",
        "fax",
        "email",
        "date",
        "time",
        "cashier",
        "sales",
        "description",
        "item",
        "qty",
        "price",
        "amount",
        "bill to",
        "customer",
        "approval",
    ]
    return any(marker in lower for marker in noise_markers)


def looks_like_address_start(line: str) -> bool:
    """Detect a line that likely starts a merchant address."""
    upper = line.upper()
    start_patterns = [
        r"^\s*(NO\.?|LOT|LEVEL|L\d+|UNIT|SUITE|KM|JALAN|JLN|PLOT|BLOCK|BLOK)\b",
        r"^\s*\d+[A-Z]?\s*,?\s*(JALAN|JLN|LORONG|PERSIARAN|ROAD|STREET|TAMAN)\b",
    ]
    if any(re.search(pattern, upper) for pattern in start_patterns):
        return True
    address_keywords = [
        "JALAN",
        "JLN",
        "LORONG",
        "PERSIARAN",
        "TAMAN",
        "BANDAR",
        "KAMPUNG",
        "KG ",
        "SELANGOR",
        "JOHOR",
        "KUALA",
        "PETALING",
        "SUNGAI",
        "MALAYSIA",
        "RAYA",
        "UTARA",
        "SELATAN",
    ]
    return bool(re.search(r"\b\d{5}\b", upper)) or any(keyword in upper for keyword in address_keywords)


def looks_like_address_continuation(line: str) -> bool:
    """Detect continuation lines after an address has started."""
    if is_address_noise_line(line):
        return False
    if looks_like_address_start(line):
        return True
    upper = line.upper()
    if re.search(r"\b\d{5}\b", upper):
        return True
    if any(state in upper for state in ("SELANGOR", "JOHOR", "KEDAH", "PENANG", "PERAK", "PAHANG", "MELAKA", "SABAH", "SARAWAK")):
        return True
    continuation_keywords = (
        "BANDAR",
        "TAMAN",
        "KAMPUNG",
        "KG ",
        "JAYA",
        "BAHRU",
        "UTARA",
        "SELATAN",
        "LAMA",
        "KALI",
        "KLANG",
        "MASAI",
        "RAJA",
    )
    return any(keyword in upper for keyword in continuation_keywords)


def address_quality_score(value: str) -> int:
    """Score whether an extracted address looks complete and low-noise."""
    text = normalize_receipt_text(value).upper()
    score = 0
    score += 3 if re.search(r"\b\d{5}\b", text) else 0
    score += 2 if re.search(r"\b(NO\.?|LOT|LEVEL|KM|JALAN|JLN)\b", text) else 0
    score += sum(1 for keyword in ("JALAN", "BANDAR", "TAMAN", "SELANGOR", "JOHOR", "KUALA", "PETALING") if keyword in text)
    score -= 3 * sum(1 for keyword in ("FORMERLY", "LICENSEE", "RESTAURANTS SDN BHD", "ROC", "GST", "TEL", "FAX", "EMAIL") if keyword in text)
    return score


def extract_merchant_address_from_ocr(raw_text: str) -> str | None:
    """Extract likely merchant address from the receipt header area."""
    lines = [clean_receipt_field(line) for line in raw_text.splitlines() if clean_receipt_field(line)]
    if not lines:
        return None

    stop_index = len(lines)
    for index, line in enumerate(lines):
        lower = line.lower()
        if index > 1 and any(marker in lower for marker in ("tax invoice", "invoice no", "receipt", "cash bill", "date:", "cashier", "description")):
            stop_index = index
            break

    header_lines = lines[:stop_index]
    best: list[str] = []
    for index, line in enumerate(header_lines):
        if is_address_noise_line(line) or not looks_like_address_start(line):
            continue
        candidate = [line]
        for next_line in header_lines[index + 1 : index + 6]:
            if looks_like_address_continuation(next_line):
                candidate.append(next_line)
            else:
                break
        if len(" ".join(candidate)) > len(" ".join(best)):
            best = candidate

    if not best:
        return None
    return clean_receipt_field(", ".join(best))


def amount_to_float(value: str) -> float | None:
    """Convert a normalized amount string to float for ranking/filtering."""
    try:
        return float(normalize_amount_text(value))
    except ValueError:
        return None


def amount_score(line: str) -> int:
    """Rank receipt lines that are likely to contain the final payable total."""
    lower = line.lower()
    score = 0
    if "grand total" in lower or "net total" in lower or "nett total" in lower:
        score += 8
    if "amount due" in lower or "balance due" in lower or "total due" in lower:
        score += 7
    if "payable" in lower:
        score += 6
    if "grand" in lower:
        score += 5
    if "total" in lower:
        score += 4
    if "amount" in lower:
        score += 2
    if "incl" in lower or "including" in lower:
        score += 1
    if "excluding" in lower:
        score -= 5
    if "subtotal" in lower or "sub total" in lower:
        score -= 4
    if "sales" in lower:
        score -= 3
    if "cash" in lower or "change" in lower or "tender" in lower or "paid" in lower:
        score -= 4
    if "tax" in lower or "gst" in lower or "vat" in lower:
        score -= 2
    if "discount" in lower or "round" in lower:
        score -= 3
    if "price" in lower or "s/price" in lower:
        score -= 3
    if "qty" in lower or "quantity" in lower:
        score -= 1 if "total" in lower else 3
    return score


def extract_total_candidates_from_ocr(raw_text: str) -> list[dict[str, Any]]:
    """Find and score likely receipt total amounts from OCR text."""
    decimal_pattern = re.compile(
        r"(?<![\d/])(?:RM|MYR|\$)?\s*((?:\d{1,3}(?:,\d{3})+|\d{1,6})[.,]\d{2})(?![\d/])",
        re.IGNORECASE,
    )
    candidates: list[dict[str, Any]] = []

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line_index, line in enumerate(lines):
        base_score = amount_score(line)
        for match in decimal_pattern.finditer(line):
            amount = normalize_amount_text(match.group(1))
            amount_value = amount_to_float(amount)
            if amount_value is None or amount_value <= 0:
                continue
            score = base_score + 3
            # Final totals are often near the lower half of receipts.
            score += int(3 * (line_index / max(len(lines), 1)))
            candidates.append(
                {
                    "amount": amount,
                    "score": score,
                    "line_number": line_index + 1,
                    "line": line,
                    "source": "same_line_decimal",
                }
            )
        # Fallback for OCR that separates cents with a space: "32 69".
        spaced = None if decimal_pattern.search(line) else re.search(r"(?<!\d)(\d{1,6})\s+(\d{2})(?!\d)", line)
        if spaced:
            amount = f"{spaced.group(1)}.{spaced.group(2)}"
            amount_value = amount_to_float(amount)
            if amount_value and amount_value > 0:
                candidates.append(
                    {
                        "amount": amount,
                        "score": base_score + 2,
                        "line_number": line_index + 1,
                        "line": line,
                        "source": "same_line_spaced_decimal",
                    }
                )

        # Some OCR outputs put the label and amount on adjacent lines.
        if base_score > 2 and not decimal_pattern.search(line):
            for offset, next_line in enumerate(lines[line_index + 1 : line_index + 3], start=1):
                for match in decimal_pattern.finditer(next_line):
                    amount = normalize_amount_text(match.group(1))
                    amount_value = amount_to_float(amount)
                    if amount_value is None or amount_value <= 0:
                        continue
                    candidates.append(
                        {
                            "amount": amount,
                            "score": base_score + 2 - offset,
                            "line_number": line_index + 1 + offset,
                            "line": next_line,
                            "source": f"near_total_label_plus_{offset}",
                        }
                    )

    if candidates:
        best_by_amount: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            current = best_by_amount.get(candidate["amount"])
            if current is None or (candidate["score"], candidate["line_number"]) > (current["score"], current["line_number"]):
                best_by_amount[candidate["amount"]] = candidate
        return sorted(best_by_amount.values(), key=lambda item: (item["score"], item["line_number"]), reverse=True)

    # Last resort: choose the last decimal-looking amount in the whole OCR text.
    all_decimals = []
    for line_index, line in enumerate(lines):
        for match in decimal_pattern.finditer(line):
            amount = normalize_amount_text(match.group(1))
            if amount_to_float(amount):
                all_decimals.append(
                    {
                        "amount": amount,
                        "score": 0,
                        "line_number": line_index + 1,
                        "line": line,
                        "source": "last_decimal_fallback",
                    }
                )
    return all_decimals[-5:][::-1]


def extract_decimal_total_from_ocr(raw_text: str) -> str | None:
    """Find the best decimal total from OCR text."""
    candidates = extract_total_candidates_from_ocr(raw_text)
    return candidates[0]["amount"] if candidates else None


def enhance_receipt_entities(raw_text: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Patch common receipt extraction issues after NER, especially decimal totals."""
    enhanced = dict(entities)

    date_from_ocr = extract_date_from_ocr(raw_text)
    if date_from_ocr:
        current_date = normalize_receipt_text(str(enhanced.get("date", "")))
        if not current_date or len(date_from_ocr) > len(current_date):
            enhanced["date"] = date_from_ocr
            enhanced["date_source"] = "ocr_regex"
            enhanced["date_confidence"] = max(float(enhanced.get("date_confidence", 0.0) or 0.0), 0.99)

    address_from_ocr = extract_merchant_address_from_ocr(raw_text)
    if address_from_ocr:
        current_address = normalize_receipt_text(str(enhanced.get("address", "")))
        current_score = address_quality_score(current_address)
        ocr_score = address_quality_score(address_from_ocr)
        if not current_address or ocr_score >= current_score or len(address_from_ocr) >= max(20, int(len(current_address) * 0.85)):
            enhanced["address"] = address_from_ocr
            enhanced["address_source"] = "ocr_header_rule"
            enhanced["address_confidence"] = max(float(enhanced.get("address_confidence", 0.0) or 0.0), 0.95)

    total_candidates = extract_total_candidates_from_ocr(raw_text)
    decimal_total = total_candidates[0]["amount"] if total_candidates else None
    current_total = str(enhanced.get("total", "")).strip()

    if decimal_total and (not current_total or "." not in current_total or len(decimal_total) > len(current_total)):
        enhanced["total"] = decimal_total
        enhanced["total_source"] = "ocr_regex"
        enhanced["total_confidence"] = max(float(enhanced.get("total_confidence", 0.0) or 0.0), 0.99)
    elif current_total:
        enhanced["total"] = normalize_amount_text(current_total) or current_total
    if total_candidates:
        enhanced["total_ocr_candidates"] = total_candidates

    return enhanced


def entity_summary_rows(entities: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert structured entity dict into display rows."""
    rows = []
    for key, value in entities.items():
        if key.endswith("_candidates") or key.endswith("_confidence") or key.endswith("_source"):
            continue
        confidence = entities.get(f"{key}_confidence")
        rows.append({"entity": key, "value": value, "confidence": confidence})
    return rows


def make_result_payload(raw_text: str, entities: dict[str, Any]) -> str:
    """Build a downloadable JSON result."""
    return json.dumps(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_ocr_text": raw_text,
            "entities": entities,
        },
        ensure_ascii=False,
        indent=2,
    )


def main() -> None:
    st.set_page_config(page_title="Receipt OCR + NER", page_icon="🧾", layout="wide")
    init_db()

    st.title("Receipt OCR + NER Extractor")
    st.caption("Capture an English receipt, extract OCR text, run your RoBERTa NER model, and save entities to SQLite.")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("HuggingFace model path", value=str(DEFAULT_MODEL_PATH))
        st.caption("OCR engine: PaddleOCR for English receipt text detection and recognition.")
        save_to_db = st.toggle("Save processed receipts", value=True)
        show_raw_ner = st.toggle("Show raw NER output", value=False)

        is_valid_model, model_error = validate_model_path(model_path)
        if is_valid_model:
            st.success("NER model folder looks valid.")
        else:
            st.warning(model_error)
            st.caption("Expected default:")
            st.code(str(DEFAULT_MODEL_PATH), language="text")

        st.divider()
        st.markdown("**Tips**")
        st.caption("Use a flat, bright, uncropped receipt image. SROIE-like scanned receipts work best with this model.")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("1. Choose Receipt Image")
        input_mode = st.radio("Image source", ["Upload image", "Camera"], horizontal=True)
        captured_image = None
        if input_mode == "Upload image":
            captured_image = st.file_uploader(
                "Upload receipt image",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=False,
            )
        else:
            captured_image = st.camera_input("Capture receipt photo")

        process_clicked = False
        if captured_image is not None:
            caption = "Uploaded receipt" if input_mode == "Upload image" else "Captured receipt"
            st.image(captured_image, caption=caption, use_container_width=True)
            process_clicked = st.button("Process Receipt", type="primary")
        else:
            st.info("Upload a receipt image or switch to Camera to take a photo.")

    with right:
        st.subheader("2. Review Extraction")
        if captured_image is None:
            st.write("Waiting for receipt image.")

        if captured_image is not None and process_clicked:
            image = image_from_streamlit_upload(captured_image)

            with st.spinner("Extracting text with OCR..."):
                try:
                    raw_text = extract_text_with_ocr(image)
                except Exception as exc:
                    st.error("PaddleOCR failed. Check that paddleocr/paddlepaddle are installed, or try a clearer image.")
                    st.exception(exc)
                    return

            if not raw_text:
                st.error("OCR did not detect any text. Try a clearer, brighter photo.")
                return

            with st.spinner("Running receipt NER model..."):
                if not is_valid_model:
                    st.error(model_error)
                    st.code(str(DEFAULT_MODEL_PATH), language="text")
                    return
                try:
                    ner = load_ner_pipeline(model_path)
                    ner_output = ner(raw_text)
                    entities = enhance_receipt_entities(raw_text, parse_ner_entities(ner_output))
                except Exception as exc:
                    st.error("Could not load or run the NER model. Check that the model path points to a HuggingFace token-classification export.")
                    st.exception(exc)
                    return

            st.session_state["last_receipt_result"] = {
                "raw_text": raw_text,
                "entities": entities,
                "ner_output": ner_output,
            }
            st.success("Receipt processed. Review the extracted fields before saving.")

        if "last_receipt_result" in st.session_state:
            result = st.session_state["last_receipt_result"]
            raw_text = result["raw_text"]
            entities = dict(result["entities"])
            ner_output = result["ner_output"]

            st.metric("OCR characters", len(raw_text))
            with st.expander("Raw OCR text", expanded=False):
                st.text_area("OCR output", raw_text, height=220, label_visibility="collapsed", key="review_raw_ocr_text")

            total_candidates = entities.get("total_ocr_candidates", [])
            if total_candidates:
                candidate_options = [
                    f"{item['amount']} | score={item['score']} | line {item['line_number']}: {item['line']}"
                    for item in total_candidates[:8]
                ]
                with st.expander("Total amount candidates from OCR", expanded=True):
                    selected_candidate = st.selectbox("Choose the correct total candidate", candidate_options)
                    selected_amount = selected_candidate.split("|", 1)[0].strip()
                    if selected_amount:
                        entities["total"] = selected_amount
                        entities["total_source"] = "user_selected_ocr_candidate"

            reviewed_total = st.text_input("Review total before saving", value=str(entities.get("total", "")))
            if reviewed_total:
                normalized_reviewed_total = normalize_amount_text(reviewed_total) or reviewed_total
                if normalized_reviewed_total != entities.get("total"):
                    entities["total"] = normalized_reviewed_total
                    entities["total_source"] = "user_review"
                    entities["total_confidence"] = 1.0

            rows = entity_summary_rows(entities)
            if rows:
                st.write("Extracted entities")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No entities were extracted. OCR text may be noisy or the receipt layout may differ from training data.")

            with st.expander("Structured JSON"):
                st.json(entities)

            if show_raw_ner:
                with st.expander("Raw HuggingFace NER output", expanded=False):
                    st.json(ner_output)

            if save_to_db:
                if st.button("Save reviewed result", type="primary"):
                    scan_id = save_scan(raw_text, entities)
                    st.success(f"Reviewed receipt saved. Scan ID: {scan_id}")
                    st.session_state["last_receipt_result"]["entities"] = entities
            else:
                st.info("Saving is disabled in the sidebar.")

            st.download_button(
                "Download extraction JSON",
                data=make_result_payload(raw_text, entities),
                file_name="receipt_extraction.json",
                mime="application/json",
            )

    st.divider()
    st.subheader("Recent scans")
    history_limit = st.slider("History rows", min_value=5, max_value=50, value=10, step=5)
    clear_clicked = st.button("Clear saved scans")
    if clear_clicked:
        clear_scans()
        st.success("Saved scans cleared.")

    recent = load_recent_scans(limit=history_limit)
    if recent.empty:
        st.write("No scans saved yet.")
    else:
        recent["entities"] = recent["entities_json"].map(json.loads)
        recent["summary"] = recent["entities"].map(lambda item: {key: value for key, value in item.items() if not key.endswith("_candidates") and not key.endswith("_confidence")})
        st.dataframe(recent[["id", "scanned_at", "summary"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
