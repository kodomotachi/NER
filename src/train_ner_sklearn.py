#!/usr/bin/env python3
"""Train and compare lightweight sklearn NER baselines on prepared JSONL data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def token_shape(token: str) -> str:
    chars = []
    for char in token:
        if char.isupper():
            chars.append("X")
        elif char.islower():
            chars.append("x")
        elif char.isdigit():
            chars.append("d")
        else:
            chars.append(char)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(chars))


def features_for_sentence(tokens: list[str]) -> list[dict[str, object]]:
    features = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        prev_token = tokens[i - 1] if i > 0 else "<BOS>"
        next_token = tokens[i + 1] if i + 1 < len(tokens) else "<EOS>"
        features.append(
            {
                "bias": 1.0,
                "token.lower": lower,
                "token.shape": token_shape(token),
                "token.isdigit": token.isdigit(),
                "token.isupper": token.isupper(),
                "token.istitle": token.istitle(),
                "prefix2": lower[:2],
                "prefix3": lower[:3],
                "suffix2": lower[-2:],
                "suffix3": lower[-3:],
                "prev.lower": prev_token.lower(),
                "next.lower": next_token.lower(),
                "prev+token": f"{prev_token.lower()} {lower}",
                "token+next": f"{lower} {next_token.lower()}",
                "position": i,
            }
        )
    return features


def flatten(rows: list[dict]) -> tuple[list[dict[str, object]], list[str]]:
    x, y = [], []
    for row in rows:
        x.extend(features_for_sentence(row["tokens"]))
        y.extend(row["ner_tags"])
    return x, y


def entity_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans = set()
    start = None
    label = None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if label is not None:
                spans.add((start, i, label))
                start = None
                label = None
            continue
        prefix, current_label = tag.split("-", 1)
        if prefix == "B" or current_label != label:
            if label is not None:
                spans.add((start, i, label))
            start = i
            label = current_label
    return spans


def entity_scores(rows: list[dict], predictions: list[list[str]]) -> dict[str, float]:
    true_total = pred_total = correct = 0
    for row, pred_tags in zip(rows, predictions):
        true_spans = entity_spans(row["ner_tags"])
        pred_spans = entity_spans(pred_tags)
        true_total += len(true_spans)
        pred_total += len(pred_spans)
        correct += len(true_spans & pred_spans)
    precision = correct / pred_total if pred_total else 0.0
    recall = correct / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def predict_rows(model: Pipeline, rows: list[dict]) -> list[list[str]]:
    predictions = []
    for row in rows:
        predictions.append(list(model.predict(features_for_sentence(row["tokens"]))))
    return predictions


def train_model(train_rows: list[dict]) -> Pipeline:
    x_train, y_train = flatten(train_rows)
    model = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("classifier", SGDClassifier(loss="log_loss", max_iter=40, tol=1e-3, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate(model: Pipeline, rows: list[dict]) -> dict:
    if not rows:
        return {}
    x, y_true = flatten(rows)
    y_pred = model.predict(x)
    token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        labels=sorted({tag for tag in y_true if tag != "O"}),
        zero_division=0,
    )
    predictions = predict_rows(model, rows)
    scores = {
        "token_precision_non_o": float(token_precision),
        "token_recall_non_o": float(token_recall),
        "token_f1_non_o": float(token_f1),
        "token_accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
    }
    scores.update(entity_scores(rows, predictions))
    scores["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
    return scores


def run_experiment(name: str, train_path: Path, valid_rows: list[dict], test_rows: list[dict]) -> dict:
    train_rows = load_jsonl(train_path)
    model = train_model(train_rows)
    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / f"{name}.joblib"
    joblib.dump(model, model_path)
    return {
        "name": name,
        "train_rows": len(train_rows),
        "model_path": str(model_path),
        "a_valid": evaluate(model, valid_rows),
        "a_test": evaluate(model, test_rows),
    }


def compact(scores: dict) -> dict[str, float]:
    return {key: round(value, 4) for key, value in scores.items() if isinstance(value, float)}


def score_cell(result: dict, split: str, metric: str) -> str:
    value = result.get(split, {}).get(metric)
    return "" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()

    valid_rows = load_jsonl(args.processed_dir / "a_valid.jsonl")
    test_rows = load_jsonl(args.processed_dir / "a_test.jsonl")
    b_valid_rows = load_jsonl(args.processed_dir / "b_valid.jsonl")
    b_test_rows = load_jsonl(args.processed_dir / "b_test.jsonl")

    results = [
        run_experiment("baseline_a_only", args.processed_dir / "train_a.jsonl", valid_rows, test_rows),
        run_experiment("mixed_a_plus_b", args.processed_dir / "train_a_plus_b.jsonl", valid_rows, test_rows),
    ]
    if load_jsonl(args.processed_dir / "b_unique.jsonl"):
        results.append(run_experiment("baseline_b_only", args.processed_dir / "b_unique.jsonl", valid_rows, test_rows))

    for result in results:
        model = joblib.load(result["model_path"])
        result["b_valid"] = evaluate(model, b_valid_rows)
        result["b_test"] = evaluate(model, b_test_rows)

    REPORTS.mkdir(exist_ok=True)
    result_path = REPORTS / "ner_experiment_results.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ["# NER experiment results", ""]
    md.append("| model | train rows | A valid entity F1 | A test entity F1 | B valid entity F1 | B test entity F1 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for result in results:
        md.append(
            f"| {result['name']} | {result['train_rows']} | "
            f"{score_cell(result, 'a_valid', 'entity_f1')} | "
            f"{score_cell(result, 'a_test', 'entity_f1')} | "
            f"{score_cell(result, 'b_valid', 'entity_f1')} | "
            f"{score_cell(result, 'b_test', 'entity_f1')} |"
        )
    compact_results = {
        r["name"]: {
            "a_valid": compact(r.get("a_valid", {})),
            "a_test": compact(r.get("a_test", {})),
            "b_valid": compact(r.get("b_valid", {})),
            "b_test": compact(r.get("b_test", {})),
        }
        for r in results
    }
    md.extend(["", "## Compact metrics", "```json", json.dumps(compact_results, indent=2), "```"])
    (REPORTS / "ner_experiment_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(compact_results, indent=2))
    print(f"Wrote report: {REPORTS / 'ner_experiment_results.md'}")


if __name__ == "__main__":
    main()
