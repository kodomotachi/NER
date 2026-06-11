#!/usr/bin/env python3
"""Transformer-based NER experiment helpers used by the notebooks.

The module intentionally keeps imports for optional libraries inside functions so
the classical sklearn pipeline can still run without Transformer dependencies.
"""

from __future__ import annotations

import json
import math
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "deep_ner"
MODELS = ROOT / "models" / "deep_ner"


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_label_map(path: str | Path = PROCESSED / "label_map.json") -> tuple[dict[str, int], dict[int, str]]:
    label2id = json.loads(Path(path).read_text(encoding="utf-8"))
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_label_map(rows: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    entity_labels = sorted({tag[2:] for row in rows for tag in row["ner_tags"] if tag != "O"})
    labels = ["O"]
    for entity in entity_labels:
        labels.extend([f"B-{entity}", f"I-{entity}"])
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


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


def sequence_metrics(rows: list[dict], predictions: list[list[str]]) -> dict[str, float]:
    y_true = []
    y_pred = []
    true_total = pred_total = correct = 0
    for row, pred_tags in zip(rows, predictions):
        gold_tags = row["ner_tags"][: len(pred_tags)]
        y_true.extend(gold_tags)
        y_pred.extend(pred_tags[: len(gold_tags)])
        true_spans = entity_spans(gold_tags)
        pred_spans = entity_spans(pred_tags[: len(gold_tags)])
        true_total += len(true_spans)
        pred_total += len(pred_spans)
        correct += len(true_spans & pred_spans)

    non_o = [idx for idx, tag in enumerate(y_true) if tag != "O"]
    token_correct = sum(1 for idx in non_o if idx < len(y_pred) and y_pred[idx] == y_true[idx])
    token_precision = token_correct / max(sum(1 for tag in y_pred if tag != "O"), 1)
    token_recall = token_correct / max(len(non_o), 1)
    token_f1 = 2 * token_precision * token_recall / max(token_precision + token_recall, 1e-12)
    entity_precision = correct / max(pred_total, 1)
    entity_recall = correct / max(true_total, 1)
    entity_f1 = 2 * entity_precision * entity_recall / max(entity_precision + entity_recall, 1e-12)
    return {
        "token_precision_non_o": float(token_precision),
        "token_recall_non_o": float(token_recall),
        "token_f1_non_o": float(token_f1),
        "entity_precision": float(entity_precision),
        "entity_recall": float(entity_recall),
        "entity_f1": float(entity_f1),
    }


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer: Any,
        label2id: dict[str, int],
        max_length: int = 256,
        label_all_subtokens: bool = False,
        add_special_tokens: bool = True,
        include_word_ids: bool = False,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_subtokens = label_all_subtokens
        self.add_special_tokens = add_special_tokens
        self.include_word_ids = include_word_ids

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=self.add_special_tokens,
        )
        word_ids = encoding.word_ids()
        labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id or self.label_all_subtokens:
                labels.append(self.label2id[row["ner_tags"][word_id]])
            else:
                labels.append(-100)
            previous_word_id = word_id
        item = {key: torch.tensor(value) for key, value in encoding.items()}
        item["labels"] = torch.tensor(labels)
        if self.include_word_ids:
            item["word_ids"] = torch.tensor([-1 if word_id is None else word_id for word_id in word_ids])
        return item


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_result(result: dict, output_dir: Path = REPORTS) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_fast_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    kwargs = {"use_fast": True}
    if "roberta" in model_name.lower():
        kwargs["add_prefix_space"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    allowed = {"input_ids", "attention_mask", "labels"}
    return {key: value for key, value in batch.items() if key in allowed}


def encoder_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    allowed = {"input_ids", "attention_mask"}
    return {key: value for key, value in batch.items() if key in allowed}


def train_hf_token_classifier(
    name: str,
    model_name: str,
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    label2id: dict[str, int],
    id2label: dict[int, str],
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> dict:
    from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

    model_dir = MODELS / name
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_fast_tokenizer(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    train_ds = TokenClassificationDataset(train_rows, tokenizer, label2id, max_length=max_length)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = max(len(loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )
    started = time.time()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**model_inputs(batch))
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        print(f"{name} epoch {epoch + 1}/{epochs} loss={total_loss / max(len(loader), 1):.4f}")

    model.save_pretrained(str(model_dir / "best"))
    tokenizer.save_pretrained(str(model_dir / "best"))

    result = {
        "name": name,
        "architecture": "transformer_token_classification",
        "base_model": model_name,
        "train_rows": len(train_rows),
        "valid": predict_hf_token_classifier(model, tokenizer, valid_rows, label2id, id2label, max_length, batch_size),
        "test": predict_hf_token_classifier(model, tokenizer, test_rows, label2id, id2label, max_length, batch_size),
        "seconds": round(time.time() - started, 2),
        "model_path": str(model_dir / "best"),
    }
    _write_result(result)
    return result


def predict_hf_token_classifier(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict],
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int,
    batch_size: int,
) -> dict:
    device = _device()
    model.to(device)
    model.eval()
    ds = TokenClassificationDataset(rows, tokenizer, label2id, max_length=max_length, include_word_ids=True)
    loader = DataLoader(ds, batch_size=batch_size)
    all_predictions: list[list[str]] = []
    for batch in loader:
        labels = batch.pop("labels")
        word_ids = batch.pop("word_ids")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            pred_ids = model(**encoder_inputs(batch)).logits.argmax(dim=-1).cpu()
        for pred_row, label_row, word_id_row in zip(pred_ids.tolist(), labels.tolist(), word_ids.tolist()):
            tags = []
            seen_words = set()
            for pred, label, word_id in zip(pred_row, label_row, word_id_row):
                if label == -100 or word_id < 0 or word_id in seen_words:
                    continue
                tags.append(id2label[pred])
                seen_words.add(word_id)
            all_predictions.append(tags)
    return sequence_metrics(rows, all_predictions)


class TransformerCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        try:
            from torchcrf import CRF
        except ImportError:
            from TorchCRF import CRF
        crf_params = inspect.signature(CRF.__init__).parameters
        if "batch_first" in crf_params:
            self.crf_style = "torchcrf"
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf_style = "TorchCRF"
            try:
                self.crf = CRF(num_labels, use_gpu=torch.cuda.is_available())
            except TypeError:
                self.crf = CRF(num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))
        mask = attention_mask.bool()
        if labels is None:
            if self.crf_style == "torchcrf":
                return self.crf.decode(emissions, mask=mask)
            return self.crf.viterbi_decode(emissions, mask)
        clean_labels = labels.masked_fill(labels == -100, 0)
        if self.crf_style == "torchcrf":
            loss = -self.crf(emissions, clean_labels, mask=mask, reduction="mean")
        else:
            log_likelihood = self.crf(emissions, clean_labels, mask)
            loss = -log_likelihood.mean()
        return loss


class TransformerGlobalContext(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.context = nn.LSTM(hidden, hidden // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        contextualized, _ = self.context(hidden)
        logits = self.classifier(self.dropout(contextualized))
        if labels is None:
            return logits
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))


def train_custom_sequence_model(
    name: str,
    architecture: str,
    model_name: str,
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    label2id: dict[str, int],
    id2label: dict[int, str],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> dict:
    from transformers import get_linear_schedule_with_warmup

    tokenizer = load_fast_tokenizer(model_name)
    add_special_tokens = architecture != "crf"
    train_ds = TokenClassificationDataset(
        train_rows,
        tokenizer,
        label2id,
        max_length=max_length,
        label_all_subtokens=True,
        add_special_tokens=add_special_tokens,
        include_word_ids=True,
    )
    valid_ds = TokenClassificationDataset(
        valid_rows,
        tokenizer,
        label2id,
        max_length=max_length,
        label_all_subtokens=True,
        add_special_tokens=add_special_tokens,
        include_word_ids=True,
    )
    test_ds = TokenClassificationDataset(
        test_rows,
        tokenizer,
        label2id,
        max_length=max_length,
        label_all_subtokens=True,
        add_special_tokens=add_special_tokens,
        include_word_ids=True,
    )

    if architecture == "crf":
        model = TransformerCRF(model_name, len(label2id))
    elif architecture == "global_context":
        model = TransformerGlobalContext(model_name, len(label2id))
    else:
        raise ValueError(f"Unknown custom architecture: {architecture}")

    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    steps = max(len(loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(steps // 10, 1), num_training_steps=steps)
    started = time.time()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**model_inputs(batch))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    model_dir = MODELS / name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(model_dir)
    result = {
        "name": name,
        "architecture": architecture,
        "base_model": model_name,
        "train_rows": len(train_rows),
        "valid": predict_custom_sequence_model(model, valid_ds, valid_rows, id2label, batch_size, architecture),
        "test": predict_custom_sequence_model(model, test_ds, test_rows, id2label, batch_size, architecture),
        "seconds": round(time.time() - started, 2),
        "model_path": str(model_dir),
    }
    _write_result(result)
    return result


def predict_custom_sequence_model(
    model: nn.Module,
    dataset: Dataset,
    rows: list[dict],
    id2label: dict[int, str],
    batch_size: int,
    architecture: str,
) -> dict:
    device = _device()
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[list[str]] = []
    for batch in loader:
        labels = batch.pop("labels")
        word_ids = batch.pop("word_ids")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            if architecture == "crf":
                pred_batch = model(**encoder_inputs(batch))
            else:
                pred_batch = model(**encoder_inputs(batch)).argmax(dim=-1).cpu().tolist()
        for pred_row, label_row, word_id_row in zip(pred_batch, labels.tolist(), word_ids.tolist()):
            tags = []
            seen_words = set()
            for pred, label, word_id in zip(pred_row, label_row, word_id_row):
                if label == -100 or word_id < 0 or word_id in seen_words:
                    continue
                tags.append(id2label[int(pred)])
                seen_words.add(word_id)
            predictions.append(tags)
    return sequence_metrics(rows, predictions)


class GlobalPointerDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer: Any, label2id: dict[str, int], max_length: int = 256) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.entity_labels = sorted({tag[2:] for tag in label2id if tag.startswith("B-")})
        self.entity2id = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = encoding.word_ids()
        word_to_token = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in word_to_token:
                word_to_token[word_id] = token_idx
        labels = torch.zeros((len(self.entity_labels), self.max_length, self.max_length), dtype=torch.float)
        for start, end, label in entity_spans(row["ner_tags"]):
            if label in self.entity2id and start in word_to_token and end - 1 in word_to_token:
                labels[self.entity2id[label], word_to_token[start], word_to_token[end - 1]] = 1.0
        item = {key: torch.tensor(value) for key, value in encoding.items()}
        item["span_labels"] = labels
        item["word_ids"] = torch.tensor([-1 if word_id is None else word_id for word_id in word_ids])
        return item


class GlobalPointerModel(nn.Module):
    def __init__(self, model_name: str, num_entity_labels: int, inner_dim: int = 64) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.num_entity_labels = num_entity_labels
        self.inner_dim = inner_dim
        self.dense = nn.Linear(hidden, num_entity_labels * inner_dim * 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch, seq_len, _ = hidden.shape
        projected = self.dense(hidden).view(batch, seq_len, self.num_entity_labels, self.inner_dim * 2)
        qw, kw = projected[..., : self.inner_dim], projected[..., self.inner_dim :]
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) / math.sqrt(self.inner_dim)
        mask = attention_mask[:, None, None, :].bool() & attention_mask[:, None, :, None].bool()
        logits = logits.masked_fill(~mask, -1e12)
        return logits


def train_global_pointer(
    name: str,
    model_name: str,
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    label2id: dict[str, int],
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    threshold: float = 0.0,
) -> dict:
    tokenizer = load_fast_tokenizer(model_name)
    train_ds = GlobalPointerDataset(train_rows, tokenizer, label2id, max_length=max_length)
    valid_ds = GlobalPointerDataset(valid_rows, tokenizer, label2id, max_length=max_length)
    test_ds = GlobalPointerDataset(test_rows, tokenizer, label2id, max_length=max_length)
    model = GlobalPointerModel(model_name, len(train_ds.entity_labels))
    device = _device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(50.0, device=device))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    started = time.time()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            labels = batch.pop("span_labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**encoder_inputs(batch))
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model_dir = MODELS / name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(model_dir)
    threshold_candidates = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
    valid_by_threshold = [
        (candidate, predict_global_pointer(model, valid_ds, valid_rows, batch_size, candidate))
        for candidate in threshold_candidates
    ]
    best_threshold, best_valid = max(valid_by_threshold, key=lambda item: item[1]["entity_f1"])
    result = {
        "name": name,
        "architecture": "global_pointer",
        "base_model": model_name,
        "train_rows": len(train_rows),
        "threshold": best_threshold,
        "valid": best_valid,
        "test": predict_global_pointer(model, test_ds, test_rows, batch_size, best_threshold),
        "seconds": round(time.time() - started, 2),
        "model_path": str(model_dir),
    }
    _write_result(result)
    return result


def predict_global_pointer(model: nn.Module, dataset: GlobalPointerDataset, rows: list[dict], batch_size: int, threshold: float) -> dict:
    device = _device()
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions: list[list[str]] = []
    entity_labels = dataset.entity_labels
    for batch in loader:
        batch.pop("span_labels")
        word_ids = batch.pop("word_ids")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            scores = model(**encoder_inputs(batch)).cpu()
        for row_scores, word_id_row in zip(scores, word_ids.tolist()):
            token_to_word = {token_idx: word_id for token_idx, word_id in enumerate(word_id_row) if word_id >= 0}
            word_count = max(token_to_word.values(), default=-1) + 1
            tags = ["O"] * word_count
            candidates = (row_scores > threshold).nonzero(as_tuple=False).tolist()
            candidates = sorted(candidates, key=lambda item: float(row_scores[item[0], item[1], item[2]]), reverse=True)
            occupied = set()
            for label_id, start, end in candidates:
                if start > end or start not in token_to_word or end not in token_to_word:
                    continue
                word_start, word_end = token_to_word[start], token_to_word[end]
                if word_start > word_end or any(i in occupied for i in range(word_start, word_end + 1)):
                    continue
                label = entity_labels[label_id]
                tags[word_start] = f"B-{label}"
                for i in range(word_start + 1, word_end + 1):
                    tags[i] = f"I-{label}"
                occupied.update(range(word_start, word_end + 1))
            predictions.append(tags)
    return sequence_metrics(rows, predictions)


def collect_results(result_dir: str | Path = REPORTS, metric_split: str = "valid") -> list[dict]:
    rows = []
    for path in sorted(Path(result_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get(metric_split, {})
        rows.append(
            {
                "name": data.get("name"),
                "architecture": data.get("architecture"),
                "base_model": data.get("base_model"),
                "train_rows": data.get("train_rows"),
                "valid_entity_f1": data.get("valid", {}).get("entity_f1"),
                "test_entity_f1": data.get("test", {}).get("entity_f1"),
                "valid_token_f1": data.get("valid", {}).get("token_f1_non_o"),
                "test_token_f1": data.get("test", {}).get("token_f1_non_o"),
                "selected_metric": metrics.get("entity_f1"),
                "model_path": data.get("model_path"),
            }
        )
    return sorted(rows, key=lambda item: item.get("selected_metric") or -1, reverse=True)


def write_leaderboard(rows: list[dict], path: str | Path = REPORTS / "leaderboard.md") -> None:
    lines = ["# Deep NER leaderboard", ""]
    lines.append("| rank | model | architecture | base | valid entity F1 | test entity F1 | valid token F1 | test token F1 | path |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|")
    for rank, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {name} | {architecture} | {base_model} | {valid:.4f} | {test:.4f} | {valid_token:.4f} | {test_token:.4f} | `{path}` |".format(
                rank=rank,
                name=row["name"],
                architecture=row["architecture"],
                base_model=row["base_model"],
                valid=row["valid_entity_f1"] or 0.0,
                test=row["test_entity_f1"] or 0.0,
                valid_token=row["valid_token_f1"] or 0.0,
                test_token=row["test_token_f1"] or 0.0,
                path=row["model_path"],
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
