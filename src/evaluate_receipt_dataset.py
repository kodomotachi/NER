#!/usr/bin/env python3
"""Evaluate the receipt OCR + NER pipeline on SROIE English receipts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import enhance_receipt_entities, extract_text_with_ocr, parse_ner_entities  # noqa: E402


FIELD_MAP = {
    "company": "vendor",
    "address": "address",
    "date": "date",
    "total": "total",
}
DEFAULT_DATASET = ROOT / "data" / "raw" / "SROIE_datasetv2.zip"
DEFAULT_MODEL = ROOT / "models" / "deep_ner" / "khai_roberta_token_cls" / "best"
DEFAULT_REPORT = ROOT / "reports" / "sroie_receipt_evaluation.md"
DEFAULT_JSON = ROOT / "reports" / "sroie_receipt_evaluation.json"
DEFAULT_CSV = ROOT / "reports" / "sroie_receipt_predictions.csv"


@dataclass
class SroieRecord:
    doc_id: str
    split: str
    text: str
    gt: dict[str, str]
    image_name: str | None


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("\\n", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_total(value: Any) -> str:
    text = "" if value is None else str(value)
    match = re.search(r"\d+(?:[,.]\d{2})?", text.replace(" ", ""))
    if not match:
        return normalize_text(text)
    try:
        return f"{float(match.group(0).replace(',', '.')):.2f}"
    except ValueError:
        return normalize_text(text)


def normalize_date(value: Any) -> str:
    text = "" if value is None else str(value)
    match = re.search(r"(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})", text)
    if not match:
        return normalize_text(text)
    parts = [part.zfill(2) if len(part) <= 2 else part for part in match.groups()]
    return "/".join(parts)


def normalize_field(field: str, value: Any) -> str:
    if field == "total":
        return normalize_total(value)
    if field == "date":
        return normalize_date(value)
    return normalize_text(value)


def similarity(field: str, pred: Any, gold: Any) -> float:
    pred_norm = normalize_field(field, pred)
    gold_norm = normalize_field(field, gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    return SequenceMatcher(None, pred_norm, gold_norm).ratio()


def exact_match(field: str, pred: Any, gold: Any) -> bool:
    return normalize_field(field, pred) == normalize_field(field, gold)


def parse_box_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        parts = line.rstrip("\n").split(",", 8)
        if len(parts) >= 9 and parts[8].strip():
            lines.append(parts[8].strip())
    return "\n".join(lines)


def load_sroie_records(dataset_zip: Path, split: str) -> list[SroieRecord]:
    records: list[SroieRecord] = []
    with zipfile.ZipFile(dataset_zip) as zf:
        names = set(zf.namelist())
        entity_names = sorted(name for name in names if f"SROIE2019/{split}/entities/" in name and name.endswith(".txt"))
        for entity_name in entity_names:
            doc_id = Path(entity_name).stem
            box_name = f"SROIE2019/{split}/box/{doc_id}.txt"
            image_candidates = [
                f"SROIE2019/{split}/img/{doc_id}.jpg",
                f"SROIE2019/{split}/img/{doc_id}.jpeg",
                f"SROIE2019/{split}/img/{doc_id}.png",
            ]
            image_name = next((name for name in image_candidates if name in names), None)
            if box_name not in names:
                continue
            gt = json.loads(zf.read(entity_name).decode("utf-8-sig"))
            text = parse_box_text(zf.read(box_name).decode("utf-8-sig", errors="replace"))
            records.append(SroieRecord(doc_id=doc_id, split=split, text=text, gt=gt, image_name=image_name))
    return records


def extract_from_text(ner: Any, text: str) -> dict[str, Any]:
    ner_output = ner(text)
    return enhance_receipt_entities(text, parse_ner_entities(ner_output))


def predicted_fields(entities: dict[str, Any]) -> dict[str, str]:
    return {field: str(entities.get(field, "") or "") for field in FIELD_MAP.values()}


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = list(FIELD_MAP.values())
    per_field: dict[str, dict[str, Any]] = {}
    for field in fields:
        exact = sum(1 for row in rows if row[f"{field}_exact"])
        present = sum(1 for row in rows if normalize_field(field, row[f"pred_{field}"]))
        avg_sim = sum(float(row[f"{field}_similarity"]) for row in rows) / max(len(rows), 1)
        per_field[field] = {
            "exact": exact,
            "total": len(rows),
            "exact_accuracy": exact / max(len(rows), 1),
            "prediction_present_rate": present / max(len(rows), 1),
            "avg_similarity": avg_sim,
        }
    all_fields_exact = sum(1 for row in rows if all(row[f"{field}_exact"] for field in fields))
    return {
        "documents": len(rows),
        "all_fields_exact": all_fields_exact,
        "all_fields_exact_accuracy": all_fields_exact / max(len(rows), 1),
        "per_field": per_field,
    }


def build_eval_row(record: SroieRecord, mode: str, raw_text: str, entities: dict[str, Any], seconds: float) -> dict[str, Any]:
    pred = predicted_fields(entities)
    row: dict[str, Any] = {
        "doc_id": record.doc_id,
        "split": record.split,
        "mode": mode,
        "seconds": round(seconds, 3),
        "ocr_chars": len(raw_text),
    }
    for source_field, pred_field in FIELD_MAP.items():
        gold = record.gt.get(source_field, "")
        value = pred.get(pred_field, "")
        row[f"gold_{pred_field}"] = gold
        row[f"pred_{pred_field}"] = value
        row[f"{pred_field}_exact"] = exact_match(pred_field, value, gold)
        row[f"{pred_field}_similarity"] = round(similarity(pred_field, value, gold), 4)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Field | Exact accuracy | Avg similarity | Prediction present |",
        "|---|---:|---:|---:|",
    ]
    for field, item in metrics["per_field"].items():
        lines.append(
            f"| `{field}` | {item['exact_accuracy']:.1%} ({item['exact']}/{item['total']}) "
            f"| {item['avg_similarity']:.3f} | {item['prediction_present_rate']:.1%} |"
        )
    return "\n".join(lines)


def sample_errors(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row in rows:
        bad_fields = [field for field in FIELD_MAP.values() if not row[f"{field}_exact"]]
        if not bad_fields:
            continue
        field = min(bad_fields, key=lambda name: row[f"{name}_similarity"])
        examples.append(
            {
                "doc_id": row["doc_id"],
                "field": field,
                "gold": row[f"gold_{field}"],
                "pred": row[f"pred_{field}"],
                "similarity": str(row[f"{field}_similarity"]),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def write_report(summary: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text_metrics = summary["metrics"].get("dataset_text", {})
    image_metrics = summary["metrics"].get("image_paddleocr", {})
    text_total = text_metrics.get("per_field", {}).get("total", {}).get("exact_accuracy", 0.0)
    image_total = image_metrics.get("per_field", {}).get("total", {}).get("exact_accuracy", 0.0)
    image_vendor = image_metrics.get("per_field", {}).get("vendor", {}).get("exact_accuracy", 0.0)
    image_address = image_metrics.get("per_field", {}).get("address", {}).get("exact_accuracy", 0.0)
    image_date = image_metrics.get("per_field", {}).get("date", {}).get("exact_accuracy", 0.0)
    text_address = text_metrics.get("per_field", {}).get("address", {}).get("exact_accuracy", 0.0)
    text_date = text_metrics.get("per_field", {}).get("date", {}).get("exact_accuracy", 0.0)
    verdict = (
        "The current project is useful as an assisted review tool, but it is not reliable enough for fully automatic receipt extraction yet."
        if max(text_total, image_total) < 0.8
        else "The current project is promising for the tested fields, but should still keep human review before saving."
    )
    date_note = (
        f"`date` is now strong after regex fallback ({text_date:.1%} exact on dataset text, {image_date:.1%} exact on PaddleOCR image mode)."
        if max(text_date, image_date) >= 0.9
        else "`date` still needs work; inspect cases where the model returns only a day token or misses text-month dates."
    )
    address_note = (
        f"`address` improved with header-line extraction ({text_address:.1%} exact on dataset text, {image_address:.1%} exact on PaddleOCR image mode), though minor OCR typos can still fail strict exact match."
        if max(text_address, image_address) >= 0.7
        else "`address` remains noisy; layout-aware extraction or bounding-box features would help."
    )
    lines = [
        "# SROIE Receipt Evaluation",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Split: `{summary['split']}`",
        f"- Model: `{summary['model']}`",
        f"- Generated at: `{summary['generated_at']}`",
        "",
        "## Dataset",
        "",
        "SROIE is an English scanned-receipt benchmark for OCR and key information extraction. "
        "This evaluation uses its four key fields: `company`, `address`, `date`, and `total`, mapped to this app as `vendor`, `address`, `date`, and `total`.",
        "",
        "Dataset references: [ICDAR 2019 SROIE Challenge](https://rrc.cvc.uab.es/?ch=13), "
        "[Kaggle SROIE datasetv2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2), "
        "[Hugging Face scanned_receipts](https://huggingface.co/datasets/Voxel51/scanned_receipts).",
        "",
        "## Assessment",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"- Best signal: `vendor` works reasonably on image mode in this sample ({image_vendor:.1%} exact).",
        f"- Main risk: `total` is still unstable ({text_total:.1%} exact on dataset text, {image_total:.1%} exact on PaddleOCR image mode).",
        f"- {date_note}",
        f"- {address_note}",
        "- Keep the review/correction UI enabled. Automatic database saving without human review is not recommended yet.",
        "",
        "## Results",
        "",
    ]
    for mode, metrics in summary["metrics"].items():
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- Documents evaluated: **{metrics['documents']}**",
                f"- All 4 fields exact: **{metrics['all_fields_exact_accuracy']:.1%}** ({metrics['all_fields_exact']}/{metrics['documents']})",
                "",
                markdown_table(metrics),
                "",
            ]
        )

    lines.extend(
        [
            "## Example Errors",
            "",
            "| Mode | Doc ID | Field | Gold | Prediction | Similarity |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for mode, examples in summary["examples"].items():
        for item in examples:
            lines.append(
                f"| {mode} | `{item['doc_id']}` | `{item['field']}` | {item['gold']} | {item['pred']} | {item['similarity']} |"
            )

    lines.extend(
        [
            "",
            "## Reading The Numbers",
            "",
            "- `dataset_text` evaluates extraction when the input text is the dataset OCR transcript. This mostly tests the NER/extraction layer.",
            "- `image_paddleocr` evaluates the current app-style image pipeline: receipt image -> PaddleOCR -> NER -> post-processing.",
            "- Exact match is intentionally strict. For `address` and `vendor`, a small OCR spelling difference can fail exact match even when similarity is high.",
            "- `total` should be treated as the most important production field; it is also the easiest field to validate with amount-specific rules.",
            "",
            "## Recommendation",
            "",
            "Use this report as a regression check after changing OCR/model logic. For a reliable product decision, rerun with a larger `--text-limit` and `--image-limit`, then inspect the CSV examples where exact match fails.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--text-limit", type=int, default=100)
    parser.add_argument("--image-limit", type=int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--skip-image-ocr", action="store_true")
    args = parser.parse_args()

    records = load_sroie_records(args.dataset, args.split)
    if args.text_limit:
        text_records = records[: args.text_limit]
    else:
        text_records = records
    image_records = [] if args.skip_image_ocr else records[: args.image_limit]

    ner = pipeline("ner", model=str(args.model), tokenizer=str(args.model), aggregation_strategy="simple")

    rows_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, record in enumerate(text_records, start=1):
        started = time.perf_counter()
        entities = extract_from_text(ner, record.text)
        rows_by_mode["dataset_text"].append(build_eval_row(record, "dataset_text", record.text, entities, time.perf_counter() - started))
        print(f"dataset_text {idx}/{len(text_records)} {record.doc_id}")

    if image_records:
        with zipfile.ZipFile(args.dataset) as zf:
            for idx, record in enumerate(image_records, start=1):
                if not record.image_name:
                    continue
                started = time.perf_counter()
                image = Image.open(BytesIO(zf.read(record.image_name))).convert("RGB")
                raw_text = extract_text_with_ocr(image)
                entities = extract_from_text(ner, raw_text)
                rows_by_mode["image_paddleocr"].append(
                    build_eval_row(record, "image_paddleocr", raw_text, entities, time.perf_counter() - started)
                )
                print(f"image_paddleocr {idx}/{len(image_records)} {record.doc_id}")

    all_rows = [row for rows in rows_by_mode.values() for row in rows]
    metrics = {mode: evaluate_predictions(rows) for mode, rows in rows_by_mode.items()}
    summary = {
        "dataset": str(args.dataset),
        "split": args.split,
        "model": str(args.model),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "examples": {mode: sample_errors(rows) for mode, rows in rows_by_mode.items()},
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"summary": summary, "rows": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(all_rows, args.csv_out)
    write_report(summary, args.report)
    print(f"Wrote report: {args.report}")
    print(f"Wrote JSON: {args.json_out}")
    print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
