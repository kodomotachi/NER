from functools import lru_cache
from pathlib import Path
from typing import Any

from transformers import pipeline


# Local path to the trained Hugging Face model artifacts.
MODEL_DIR = Path(__file__).resolve().parent / "model"


# Load the model once and reuse it across requests.
@lru_cache(maxsize=1)
def get_ner_pipeline():
    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        raise RuntimeError(
            f"NER model files were not found in {MODEL_DIR}. "
            "Run the training notebook first and save the model there."
        )

    return pipeline(
        "token-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        aggregation_strategy="simple",
    )


# Convert Hugging Face pipeline output into the API response shape.
def predict_entities(text: str) -> list[dict[str, Any]]:
    ner_pipeline = get_ner_pipeline()
    predictions = ner_pipeline(text)

    entities = []
    for item in predictions:
        entities.append(
            {
                "text": item.get("word", ""),
                "label": item.get("entity_group", item.get("entity", "")),
                "score": float(item.get("score", 0.0)),
                "start": int(item.get("start", 0)),
                "end": int(item.get("end", 0)),
            }
        )

    return entities
