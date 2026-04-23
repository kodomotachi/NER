# NER Backend API

FastAPI backend for serving the trained WikiANN Named Entity Recognition model to the Flutter app.

## Files

- `app/main.py`: Creates the FastAPI app, enables CORS, and defines the `/` and `/predict` routes.
- `app/inference.py`: Loads the local Hugging Face token classification model from `app/model/` and runs predictions.
- `app/schemas.py`: Defines the Pydantic request and response models.
- `app/model/`: Stores the trained model and tokenizer files saved by the training notebook.
- `requirements.txt`: Lists the Python packages required to run the API.

## Setup

From the project root:

```bash
cd backEnds/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before starting the API, run the training notebook and save the model into:

```text
backEnds/api/app/model/
```

## Run

From inside `backEnds/api/`:

```bash
uvicorn app.main:app --reload
```

If `uvicorn` is not found, use the venv Python directly:

```bash
./.venv/bin/python -m uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

## Test

Health check:

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "NER backend is running"
}
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Barack Obama visited Paris for a meeting with UNESCO."}'
```

Expected response shape:

```json
{
  "entities": [
    {
      "text": "Barack Obama",
      "label": "PER",
      "score": 0.98,
      "start": 0,
      "end": 12
    }
  ]
}
```

Actual entity text, labels, scores, and offsets depend on the trained model.

## Demo order tonight

1. Run `backEnds/training/wikiann_train_ner.ipynb` and save the trained model to `backEnds/api/app/model/`.
2. Start the backend from `backEnds/api/` with `./.venv/bin/python -m uvicorn app.main:app --reload`.
3. Run the Flutter app from the project root.
4. Test with sample text such as `Nguyen Van A studied at HCMUT and worked at FPT Software in Ho Chi Minh City.`
