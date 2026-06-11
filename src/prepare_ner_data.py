#!/usr/bin/env python3
"""Prepare CSV invoice and SROIE receipt data for a NER workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A = ROOT / "data" / "interim" / "invoices.csv"
DEFAULT_B = ROOT / "data" / "raw" / "SROIE_datasetv2.zip"
OUT_DIR = ROOT / "data" / "processed"

ENTITY_COLUMNS = {
    "first_name": "FIRST_NAME",
    "last_name": "LAST_NAME",
    "email": "EMAIL",
    "product_id": "PRODUCT_ID",
    "qty": "QUANTITY",
    "amount": "AMOUNT",
    "invoice_date": "INVOICE_DATE",
    "address": "ADDRESS",
    "city": "CITY",
    "stock_code": "STOCK_CODE",
    "job": "JOB",
}

TEXT_COLUMNS = list(ENTITY_COLUMNS)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._%+\-/@'][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)
SROIE_ENTITY_LABELS = {
    "company": "COMPANY",
    "address": "ADDRESS",
    "date": "DATE",
    "total": "TOTAL",
}


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


def read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one CSV in {path}, found {csv_names}")
            with zf.open(csv_names[0]) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(path)


def detect_dataset_type(path: Path) -> str:
    if path.suffix.lower() != ".zip":
        return "invoice_csv"
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if any("/entities/" in name and "/box/" not in name for name in names):
            return "sroie"
        if len([name for name in names if name.lower().endswith(".csv")]) == 1:
            return "invoice_csv"
    raise ValueError(f"Cannot infer dataset type for {path}")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in TEXT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized = df.copy()
    for col in TEXT_COLUMNS:
        normalized[col] = normalized[col].map(normalize_value)
    return normalized[TEXT_COLUMNS]


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def row_signature(row: pd.Series) -> str:
    payload = "\u241f".join(row[col] for col in TEXT_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ner_signature(row: dict) -> str:
    payload = json.dumps(
        {"text": row["text"], "entities": sorted(row["entities"], key=lambda item: (item["start"], item["end"], item["label"]))},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_piece(parts: list[str], spans: list[Span], value: str, label: str | None = None) -> None:
    start = sum(len(part) for part in parts)
    parts.append(value)
    if label and value:
        spans.append(Span(start=start, end=start + len(value), label=label))


def row_to_ner(row: pd.Series, source: str, row_id: str) -> dict:
    parts: list[str] = []
    spans: list[Span] = []

    append_piece(parts, spans, "Invoice for ")
    append_piece(parts, spans, row["first_name"], "FIRST_NAME")
    append_piece(parts, spans, " ")
    append_piece(parts, spans, row["last_name"], "LAST_NAME")
    append_piece(parts, spans, "; email ")
    append_piece(parts, spans, row["email"], "EMAIL")
    append_piece(parts, spans, "; product ")
    append_piece(parts, spans, row["product_id"], "PRODUCT_ID")
    append_piece(parts, spans, "; quantity ")
    append_piece(parts, spans, row["qty"], "QUANTITY")
    append_piece(parts, spans, "; amount ")
    append_piece(parts, spans, row["amount"], "AMOUNT")
    append_piece(parts, spans, "; date ")
    append_piece(parts, spans, row["invoice_date"], "INVOICE_DATE")
    append_piece(parts, spans, "; address ")
    append_piece(parts, spans, row["address"], "ADDRESS")
    append_piece(parts, spans, ", ")
    append_piece(parts, spans, row["city"], "CITY")
    append_piece(parts, spans, "; stock ")
    append_piece(parts, spans, row["stock_code"], "STOCK_CODE")
    append_piece(parts, spans, "; job ")
    append_piece(parts, spans, row["job"], "JOB")
    append_piece(parts, spans, ".")

    text = "".join(parts)
    tokens, ner_tags = bio_tags(text, spans)
    return {
        "id": row_id,
        "source": source,
        "text": text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "entities": [span.__dict__ for span in spans],
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_sroie_box_line(line: str) -> str | None:
    parts = line.rstrip("\n").split(",", 8)
    if len(parts) < 9:
        return None
    return parts[8].strip()


def find_sroie_spans(text: str, entities: dict[str, str]) -> list[Span]:
    spans: list[Span] = []
    occupied: list[tuple[int, int]] = []

    for field, raw_value in entities.items():
        label = SROIE_ENTITY_LABELS.get(field)
        value = normalize_text(str(raw_value))
        if not label or not value:
            continue

        patterns = [re.escape(value)]
        compact_value = re.sub(r"\s*,\s*", ", ?", value)
        if compact_value != value:
            patterns.append(re.escape(compact_value).replace(re.escape(", ?"), r"\s*,\s*"))

        match = None
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                break
        if not match:
            value_tokens = [re.escape(token) for token in value.split()]
            if value_tokens:
                match = re.search(r"\s+".join(value_tokens), text, flags=re.IGNORECASE)
        if not match:
            continue

        start, end = match.span()
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        spans.append(Span(start=start, end=end, label=label))
    return sorted(spans, key=lambda span: span.start)


def sroie_record_to_ner(doc_id: str, split: str, lines: list[str], entities: dict[str, str], source: str) -> dict:
    text = "\n".join(normalize_text(line) for line in lines if normalize_text(line))
    spans = find_sroie_spans(text, entities)
    tokens, ner_tags = bio_tags(text, spans)
    return {
        "id": f"{source}-{split}-{doc_id}",
        "source": source,
        "dataset_type": "sroie",
        "split": split,
        "text": text,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "entities": [span.__dict__ for span in spans],
        "entity_values": {SROIE_ENTITY_LABELS[key]: normalize_text(str(value)) for key, value in entities.items() if key in SROIE_ENTITY_LABELS},
    }


def load_sroie_zip(path: Path, source: str, seed: int, valid_size: float) -> tuple[list[dict], list[dict], list[dict], dict]:
    records_by_split: dict[str, list[dict]] = {"train": [], "test": []}
    missing_spans = 0
    total_fields = 0

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for split in ("train", "test"):
            entity_files = sorted(name for name in names if f"/{split}/entities/" in name and name.endswith(".txt"))
            for entity_name in entity_files:
                doc_id = Path(entity_name).stem
                box_name = entity_name.replace(f"/{split}/entities/", f"/{split}/box/")
                if box_name not in names:
                    continue
                entities = json.loads(zf.read(entity_name).decode("utf-8-sig"))
                lines = [
                    parsed
                    for parsed in (parse_sroie_box_line(line) for line in zf.read(box_name).decode("utf-8-sig", errors="replace").splitlines())
                    if parsed
                ]
                row = sroie_record_to_ner(doc_id, split, lines, entities, source)
                expected = sum(1 for key, value in entities.items() if key in SROIE_ENTITY_LABELS and normalize_text(str(value)))
                total_fields += expected
                missing_spans += max(expected - len(row["entities"]), 0)
                records_by_split[split].append(row)

    train_records = records_by_split["train"]
    if valid_size > 0 and len(train_records) > 1:
        train_idx, valid_idx = train_test_split(
            list(range(len(train_records))),
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
        )
        train = [train_records[i] for i in train_idx]
        valid = [train_records[i] for i in valid_idx]
    else:
        train = train_records
        valid = []

    stats = {
        "dataset_type": "sroie",
        "path": str(path),
        "train_documents": len(train),
        "valid_documents": len(valid),
        "test_documents": len(records_by_split["test"]),
        "entity_fields_expected": total_fields,
        "entity_fields_not_aligned_to_ocr": missing_spans,
    }
    return train, valid, records_by_split["test"], stats


def load_invoice_csv(path: Path, source: str, seed: int, valid_size: float, test_size: float) -> tuple[list[dict], list[dict], list[dict], dict]:
    df = normalize_frame(read_csv(path))
    signatures = df.apply(row_signature, axis=1)
    indices = list(range(len(df)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=valid_size + test_size,
        random_state=seed,
        shuffle=True,
    )
    relative_test_size = test_size / (valid_size + test_size)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=seed,
        shuffle=True,
    )
    stats = {
        "dataset_type": "invoice_csv",
        "path": str(path),
        "rows": len(df),
        "duplicate_rows": int(signatures.duplicated().sum()),
        "columns": TEXT_COLUMNS,
    }
    return (
        [row_to_ner(df.iloc[i], source, f"{source}-{i}") for i in train_idx],
        [row_to_ner(df.iloc[i], source, f"{source}-{i}") for i in valid_idx],
        [row_to_ner(df.iloc[i], source, f"{source}-{i}") for i in test_idx],
        stats,
    )


def load_dataset(path: Path, source: str, seed: int, valid_size: float, test_size: float) -> tuple[list[dict], list[dict], list[dict], dict]:
    dataset_type = detect_dataset_type(path)
    if dataset_type == "sroie":
        return load_sroie_zip(path, source, seed, valid_size)
    return load_invoice_csv(path, source, seed, valid_size, test_size)


def bio_tags(text: str, spans: Iterable[Span]) -> tuple[list[str], list[str]]:
    span_list = list(spans)
    tokens: list[str] = []
    tags: list[str] = []
    previous_label: str | None = None
    previous_span_index: int | None = None

    for match in TOKEN_RE.finditer(text):
        token = match.group()
        token_start, token_end = match.span()
        label = "O"
        span_index = None
        for idx, span in enumerate(span_list):
            overlaps = token_start < span.end and token_end > span.start
            if overlaps:
                prefix = "I" if previous_span_index == idx and previous_label == span.label else "B"
                label = f"{prefix}-{span.label}"
                span_index = idx
                break
        tokens.append(token)
        tags.append(label)
        previous_label = label[2:] if label != "O" else None
        previous_span_index = span_index
    return tokens, tags


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_label_map(path: Path, rows: Sequence[dict]) -> list[str]:
    entity_labels = sorted({tag[2:] for row in rows for tag in row["ner_tags"] if tag != "O"})
    labels = ["O"]
    for entity in entity_labels:
        labels.extend([f"B-{entity}", f"I-{entity}"])
    path.write_text(json.dumps({label: idx for idx, label in enumerate(labels)}, indent=2), encoding="utf-8")
    return labels


def summarize_tags(rows: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tag in row["ner_tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def entity_label_set(rows: Iterable[dict]) -> set[str]:
    return {tag[2:] for row in rows for tag in row["ner_tags"] if tag != "O"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--dataset-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--valid-size", type=float, default=0.1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    a_train, a_valid, a_test, a_stats = load_dataset(args.dataset_a, "A", args.seed, args.valid_size, args.test_size)
    b_train, b_valid, b_test, b_stats = load_dataset(args.dataset_b, "B", args.seed, args.valid_size, args.test_size)

    a_labels = entity_label_set(a_train + a_valid + a_test)
    b_labels = entity_label_set(b_train + b_valid + b_test)
    same_structure = a_stats["dataset_type"] == b_stats["dataset_type"]
    same_label_meaning = a_labels == b_labels

    a_signatures = {ner_signature(row) for row in a_train + a_valid + a_test}
    b_unique = [row for row in b_train if ner_signature(row) not in a_signatures]
    train_a_plus_b = a_train + b_unique

    counts = {
        "a_train": write_jsonl(args.out_dir / "a_train.jsonl", a_train),
        "a_valid": write_jsonl(args.out_dir / "a_valid.jsonl", a_valid),
        "a_test": write_jsonl(args.out_dir / "a_test.jsonl", a_test),
        "b_unique_no_a_leakage": write_jsonl(args.out_dir / "b_unique.jsonl", b_unique),
        "b_valid_reference": write_jsonl(args.out_dir / "b_valid.jsonl", b_valid),
        "b_test_reference": write_jsonl(args.out_dir / "b_test.jsonl", b_test),
        "train_a": write_jsonl(args.out_dir / "train_a.jsonl", a_train),
        "train_a_plus_b": write_jsonl(args.out_dir / "train_a_plus_b.jsonl", train_a_plus_b),
    }
    labels = write_label_map(args.out_dir / "label_map.json", a_train + a_valid + a_test + b_train + b_valid + b_test)

    report = [
        "# NER data preparation report",
        "",
        "## Structure and labels",
        f"- Dataset A: `{args.dataset_a}`",
        f"- Dataset B: `{args.dataset_b}`",
        f"- Dataset A type: `{a_stats['dataset_type']}`",
        f"- Dataset B type: `{b_stats['dataset_type']}`",
        f"- Same structure: `{same_structure}`",
        f"- Same label meaning: `{same_label_meaning}`",
        f"- A entity labels: `{', '.join(sorted(a_labels))}`",
        f"- B entity labels: `{', '.join(sorted(b_labels))}`",
        f"- BIO label count: `{len(labels)}`",
        "",
        "## Dataset stats",
        "```json",
        json.dumps({"A": a_stats, "B": b_stats}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Leakage checks",
        f"- B train rows also present in A and removed from A+B training: `{len(b_train) - len(b_unique)}`",
        f"- B unique train rows kept for A+B training: `{len(b_unique)}`",
        "",
        "## Splits",
        *[f"- {name}: `{count}`" for name, count in counts.items()],
        "",
        "## Tag distribution in A train",
        "```json",
        json.dumps(summarize_tags(a_train), indent=2, ensure_ascii=False),
        "```",
    ]
    (args.out_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Wrote report: {args.out_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
