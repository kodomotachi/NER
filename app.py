"""Streamlit receipt OCR + NER extraction app.

Run:
    streamlit run app.py

Notes:
    - PaddleOCR downloads its OCR model weights on first use.
    - Put the full-trained RoBERTa-CRF checkpoint in the path configured by
      DEFAULT_MODEL_PATH, or change it in the sidebar.
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
import torch
from PIL import Image, ImageOps

from src.deep_ner_models import TransformerCRF, load_fast_tokenizer


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "receipt_scans.sqlite3"
DEFAULT_MODEL_PATH = APP_DIR / "models" / "deep_ner" / "roberta_crf"
MODEL_METADATA_FILE = "ner_config.json"
REQUIRED_MODEL_FILES = ("pytorch_model.bin",)
NER_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%+\-/@'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)


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


def inference_device() -> torch.device:
    """Use CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_metadata(model_path: Path, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Load CRF architecture metadata and verify its label order."""
    metadata_path = model_path / MODEL_METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_METADATA_FILE}. Re-run the CRF training notebook so the checkpoint includes its label map."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label2id = {str(label): int(index) for label, index in metadata.get("label2id", {}).items()}
    if not label2id:
        raise ValueError(f"{metadata_path} does not contain label2id.")

    classifier_labels = int(state_dict["classifier.weight"].shape[0])
    if len(label2id) != classifier_labels:
        raise ValueError(
            f"Checkpoint classifier has {classifier_labels} labels but {metadata_path} defines {len(label2id)}."
        )
    return metadata


def adapt_crf_state_dict(state_dict: dict[str, torch.Tensor], model: TransformerCRF) -> dict[str, torch.Tensor]:
    """Translate CRF parameter names between TorchCRF and pytorch-crf APIs."""
    adapted = dict(state_dict)
    target_keys = set(model.state_dict())
    torchcrf_to_pytorch_crf = {
        "crf.trans_matrix": "crf.transitions",
        "crf.start_trans": "crf.start_transitions",
        "crf.end_trans": "crf.end_transitions",
    }
    pytorch_crf_to_torchcrf = {target: source for source, target in torchcrf_to_pytorch_crf.items()}

    mappings = (torchcrf_to_pytorch_crf, pytorch_crf_to_torchcrf)
    for mapping in mappings:
        for source, target in mapping.items():
            if source in adapted and target in target_keys and target not in adapted:
                adapted[target] = adapted.pop(source)
    return adapted


@st.cache_resource(show_spinner="Loading RoBERTa-CRF model...")
def load_ner_model(model_path: str) -> dict[str, Any]:
    """Load the custom RoBERTa-CRF checkpoint used by the best experiment."""
    path = Path(model_path).expanduser().resolve()
    try:
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu")

    metadata = load_checkpoint_metadata(path, state_dict)
    label2id = {str(label): int(index) for label, index in metadata["label2id"].items()}
    id2label = {index: label for label, index in label2id.items()}
    base_model = str(metadata.get("base_model", "roberta-base"))

    tokenizer = load_fast_tokenizer(str(path))
    model = TransformerCRF(base_model, len(label2id))
    state_dict = adapt_crf_state_dict(state_dict, model)
    model.load_state_dict(state_dict)
    device = inference_device()
    model.to(device)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "id2label": id2label,
        "device": device,
        "max_length": int(metadata.get("max_length", 384)),
    }


def validate_model_path(model_path: str) -> tuple[bool, str]:
    """Return whether a local custom RoBERTa-CRF folder looks loadable."""
    path = Path(model_path).expanduser()
    if not path.exists():
        return False, f"Model folder does not exist: {path}"
    if not path.is_dir():
        return False, f"Model path is not a folder: {path}"
    required = (*REQUIRED_MODEL_FILES, MODEL_METADATA_FILE)
    missing = [name for name in required if not (path / name).exists()]
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "vocab.json", "vocab.txt"))
    if missing:
        return False, f"Model folder is missing required file(s): {', '.join(missing)}"
    if not has_tokenizer:
        return False, "Model folder is missing tokenizer files."
    try:
        metadata = json.loads((path / MODEL_METADATA_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Cannot read {MODEL_METADATA_FILE}: {exc}"
    train_rows = metadata.get("train_rows")
    if isinstance(train_rows, int) and train_rows < 100:
        return True, f"This checkpoint was trained on only {train_rows} rows. Replace it with the full-trained Colab checkpoint."
    return True, ""


def normalize_label(label: str) -> str:
    """Normalize receipt label names from the decoded CRF sequence."""
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


def predict_crf_entities(raw_text: str, bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Decode OCR text with RoBERTa-CRF and aggregate its BIO entity spans."""
    token_matches = list(NER_TOKEN_RE.finditer(raw_text))
    tokens = [match.group() for match in token_matches]
    if not tokens:
        return {}, []

    tokenizer = bundle["tokenizer"]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=bundle["max_length"],
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_inputs = {
        key: value.to(bundle["device"])
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask"}
    }
    model = bundle["model"]
    with torch.no_grad():
        hidden = model.encoder(**model_inputs).last_hidden_state
        emissions = model.classifier(model.dropout(hidden))
        mask = model_inputs["attention_mask"].bool()
        if model.crf_style == "torchcrf":
            decoded = model.crf.decode(emissions, mask=mask)[0]
        else:
            decoded = model.crf.viterbi_decode(emissions, mask)[0]
        emission_probabilities = emissions.softmax(dim=-1)[0].cpu()

    tags_by_word: dict[int, str] = {}
    confidence_by_word: dict[int, float] = {}
    for token_position, (prediction, word_id) in enumerate(zip(decoded, word_ids)):
        if word_id is not None and word_id not in tags_by_word:
            tags_by_word[word_id] = bundle["id2label"][int(prediction)]
            confidence_by_word[word_id] = float(emission_probabilities[token_position, int(prediction)])

    observed_words = min(len(tokens), max(tags_by_word, default=-1) + 1)
    tags = [tags_by_word.get(index, "O") for index in range(observed_words)]
    spans: list[tuple[int, int, str]] = []
    active_start: int | None = None
    active_label: str | None = None

    for index, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if active_start is not None and active_label is not None:
                spans.append((active_start, index, active_label))
            active_start = None
            active_label = None
            continue

        prefix, label = tag.split("-", 1)
        if active_start is None or prefix == "B" or label != active_label:
            if active_start is not None and active_label is not None:
                spans.append((active_start, index, active_label))
            active_start = index
            active_label = label

    raw_predictions: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for start_word, end_word, model_label in spans:
        start_char = token_matches[start_word].start()
        end_char = token_matches[end_word - 1].end()
        value = normalize_receipt_text(raw_text[start_char:end_char])
        normalized_label = normalize_label(model_label)
        span_confidences = [confidence_by_word[index] for index in range(start_word, end_word) if index in confidence_by_word]
        confidence = sum(span_confidences) / max(len(span_confidences), 1)
        prediction = {
            "entity_group": model_label,
            "normalized_label": normalized_label,
            "word": value,
            "start": start_char,
            "end": end_char,
            "confidence": round(confidence, 4),
            "source": "roberta_crf",
        }
        raw_predictions.append(prediction)
        grouped.setdefault(normalized_label, []).append(prediction)

    structured: dict[str, Any] = {}
    for label, predictions in grouped.items():
        # SROIE has one gold value per field. If CRF emits multiple spans with the
        # same label, use the span supported by the strongest model emissions.
        best = max(predictions, key=lambda item: (float(item["confidence"]), len(item["word"]), -int(item["start"])))
        structured[label] = best["word"]
        structured[f"{label}_confidence"] = best["confidence"]
        structured[f"{label}_source"] = "roberta_crf"
        structured[f"{label}_candidates"] = predictions
    return structured, raw_predictions


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


def enhance_receipt_entities(raw_text: str, entities: dict[str, Any]) -> dict[str, Any]:
    """Improve date/address fields without overriding model-predicted totals."""
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
    st.caption("Capture an English receipt, extract OCR text, run the selected RoBERTa-CRF model, and save entities to SQLite.")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("RoBERTa-CRF model path", value=str(DEFAULT_MODEL_PATH))
        st.caption("OCR engine: PaddleOCR for English receipt text detection and recognition.")
        save_to_db = st.toggle("Save processed receipts", value=True)
        show_raw_ner = st.toggle("Show raw NER output", value=False)

        is_valid_model, model_error = validate_model_path(model_path)
        if is_valid_model:
            st.success("NER model folder looks valid.")
            if model_error:
                st.warning(model_error)
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
                    ner_bundle = load_ner_model(model_path)
                    entities, ner_output = predict_crf_entities(raw_text, ner_bundle)
                    entities = enhance_receipt_entities(raw_text, entities)
                except Exception as exc:
                    st.error("Could not load or run the RoBERTa-CRF model. Check the checkpoint files and label metadata.")
                    st.exception(exc)
                    return

            st.session_state["last_receipt_result"] = {
                "raw_text": raw_text,
                "entities": entities,
                "ner_output": ner_output,
            }
            st.success("Receipt processed with RoBERTa-CRF.")

        if "last_receipt_result" in st.session_state:
            result = st.session_state["last_receipt_result"]
            raw_text = result["raw_text"]
            entities = dict(result["entities"])
            ner_output = result["ner_output"]

            st.metric("OCR characters", len(raw_text))
            with st.expander("Raw OCR text", expanded=False):
                st.text_area("OCR output", raw_text, height=220, label_visibility="collapsed", key="review_raw_ocr_text")

            rows = entity_summary_rows(entities)
            if rows:
                st.write("Extracted entities")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No entities were extracted. OCR text may be noisy or the receipt layout may differ from training data.")

            with st.expander("Structured JSON"):
                st.json(entities)

            if show_raw_ner:
                with st.expander("Raw RoBERTa-CRF output", expanded=False):
                    st.json(ner_output)

            if save_to_db:
                if st.button("Save extraction", type="primary"):
                    scan_id = save_scan(raw_text, entities)
                    st.success(f"Receipt extraction saved. Scan ID: {scan_id}")
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
