from pathlib import Path

from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "saved_ner_model"

_ner_pipeline = None


def _get_pipeline():
    global _ner_pipeline

    if _ner_pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

        _ner_pipeline = pipeline(
            "token-classification",
            model=str(MODEL_PATH),
            tokenizer=str(MODEL_PATH),
            aggregation_strategy="simple",
        )

    return _ner_pipeline


def predict_entities(text: str):
    ner_pipeline = _get_pipeline()
    predictions = ner_pipeline(text)

    results = []
    for prediction in predictions:
        entity_text = prediction.get("word", "").strip()
        entity_label = prediction.get("entity_group", prediction.get("entity", ""))

        if entity_text and entity_label:
            results.append(
                {
                    "text": entity_text,
                    "label": entity_label,
                }
            )

    return results
