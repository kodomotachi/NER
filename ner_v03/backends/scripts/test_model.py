from pathlib import Path


try:
    from transformers import pipeline
except ImportError:
    print("transformers is not installed.")
    print("Install the required packages with:")
    print("pip install transformers torch")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "app" / "models" / "saved_ner_model"

if not model_path.exists():
    print("Model folder not found.")
    print(f"Expected model folder at: {model_path}")
    print("Train the model first or copy saved_ner_model into backends/app/models/.")
    raise SystemExit(1)


ner_pipeline = pipeline(
    "token-classification",
    model=str(model_path),
    tokenizer=str(model_path),
    aggregation_strategy="simple",
)

text = "Tim Cook is the CEO of Apple in California."
predictions = ner_pipeline(text)

print("Detected Entities:")
for entity in predictions:
    entity_text = entity.get("word", "").strip()
    entity_label = entity.get("entity_group", entity.get("entity", ""))
    print(f"- {entity_text} -> {entity_label}")
