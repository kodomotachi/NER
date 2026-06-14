# Receipt OCR + NER

Single-page Streamlit application and training workflow for extracting structured fields from English receipt images.

The local app lets a user upload or capture a receipt image, run OCR, pass the extracted text through a fine-tuned HuggingFace token-classification model, review the extracted fields, and save the final result to SQLite.

## What This Branch Contains

- Streamlit receipt review app in `app.py`
- OCR support with PaddleOCR
- HuggingFace `pipeline("ner")` integration for a fine-tuned RoBERTa token-classification model
- SQLite persistence for reviewed receipt scans
- SROIE/invoice data preparation scripts
- Classical sklearn baseline training
- Deep-learning experiment notebooks for:
  - BERT + CRF
  - RoBERTa + CRF
  - XLM-R + CRF
  - Transformer + Token Classification
  - Global Pointer
  - BERT + Global Context
- Colab GPU launcher notebook for full training runs

Large raw datasets, model checkpoints, processed JSONL files, and local SQLite databases are intentionally not committed.

## Local App Quick Start

```bash
git clone https://github.com/kodomotachi/NER.git
cd NER
git checkout codex/receipt-ocr-ner-app

pip install -r requirements-streamlit.txt
streamlit run app.py
```

PaddleOCR downloads its OCR model weights on first use. If you install manually, make sure both packages are available:

```bash
pip install paddleocr paddlepaddle
```

## Model Path

The Streamlit app expects a local HuggingFace token-classification export. The default path is:

```text
models/deep_ner/roberta_token_cls/best
```

That folder should contain files such as:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

If your model is elsewhere, edit the path in the app sidebar or update `DEFAULT_MODEL_PATH` in `app.py`.

## App Workflow

1. Choose image input:
   - upload image from machine
   - capture image from webcam
2. Extract raw text with OCR.
3. Run the fine-tuned NER model.
4. Review extracted entities.
5. Choose or correct the total amount candidate.
6. Save the reviewed result to SQLite.
7. Download the extraction JSON if needed.

The app deliberately asks the user to review before saving because receipt totals can be confused with subtotal, tax, cash, or change lines.

## Database

The app creates a local SQLite database:

```text
receipt_scans.sqlite3
```

Table:

```sql
receipt_scans(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at TEXT NOT NULL,
    raw_ocr_text TEXT NOT NULL,
    entities_json TEXT NOT NULL
)
```

The database is ignored by git.

## Training Data

This workflow supports two dataset styles:

- Synthetic invoice CSV data, usually:

```text
data/raw/invoices.csv.zip
```

- SROIE receipt dataset zip, usually:

```text
data/raw/SROIE_datasetv2.zip
```

These files are not committed. Put them under `data/raw/` before running data preparation.

Prepare processed NER JSONL files:

```bash
python src/prepare_ner_data.py \
  --dataset-a data/raw/invoices.csv.zip \
  --dataset-b data/raw/SROIE_datasetv2.zip
```

Outputs include:

```text
data/processed/a_train.jsonl
data/processed/a_valid.jsonl
data/processed/a_test.jsonl
data/processed/b_unique.jsonl
data/processed/b_valid.jsonl
data/processed/b_test.jsonl
data/processed/label_map.json
data/processed/audit_report.md
```

## Lightweight Baseline

Run sklearn baselines:

```bash
python src/train_ner_sklearn.py
```

Reports are written to:

```text
reports/ner_experiment_results.md
reports/ner_experiment_results.json
```

## Receipt Pipeline Evaluation

Evaluate the Streamlit-style receipt extraction pipeline on SROIE:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
python src/evaluate_receipt_dataset.py --text-limit 100 --image-limit 10
```

The evaluation compares extracted `vendor`, `address`, `date`, and `total` fields against SROIE ground truth in two modes:

- `dataset_text`: dataset-provided OCR text -> NER -> post-processing
- `image_paddleocr`: receipt image -> PaddleOCR -> NER -> post-processing

Reports are written to:

```text
reports/sroie_receipt_evaluation.md
reports/sroie_receipt_evaluation.json
reports/sroie_receipt_predictions.csv
```

## Deep Learning Experiments

Install deep-learning dependencies:

```bash
pip install -r requirements-deep.txt
```

Notebook order:

```text
notebooks/00_setup_and_prepare.ipynb
notebooks/01_transformer_token_classification.ipynb
notebooks/02_transformer_crf_models.ipynb
notebooks/03_global_pointer_and_global_context.ipynb
notebooks/04_select_best_model.ipynb
```

Current full-training config in the notebooks:

```python
LIMIT_TRAIN = None
LIMIT_EVAL = None
EPOCHS = 3
MAX_LENGTH = 384
```

If you only want a smoke test:

```python
LIMIT_TRAIN = 64
LIMIT_EVAL = 32
EPOCHS = 1
MAX_LENGTH = 192
```

## Colab GPU Workflow

For full model comparison, use Colab GPU rather than a fanless laptop.

1. Upload this project folder to Google Drive using the folder name expected by the launcher:

```text
MyDrive/NLP-test
```

2. Make sure raw dataset files exist:

```text
MyDrive/NLP-test/data/raw/invoices.csv.zip
MyDrive/NLP-test/data/raw/SROIE_datasetv2.zip
```

3. Open:

```text
notebooks/COLAB_GPU_LAUNCHER.ipynb
```

4. Set Colab runtime:

```text
Runtime > Change runtime type > GPU
```

5. Run the launcher cells from top to bottom.

The launcher copies the project from Drive to `/content/NLP-test`, trains locally in Colab runtime to avoid Google Drive mount instability, then syncs results back to Drive.

See `COLAB_GPU_GUIDE.md` for more detail.

## Project Layout

```text
NER/
├── app.py
├── COLAB_GPU_GUIDE.md
├── README.md
├── README_NER.md
├── requirements-streamlit.txt
├── requirements-deep.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── deep_ner/
├── notebooks/
│   ├── 00_setup_and_prepare.ipynb
│   ├── 01_transformer_token_classification.ipynb
│   ├── 02_transformer_crf_models.ipynb
│   ├── 03_global_pointer_and_global_context.ipynb
│   ├── 04_select_best_model.ipynb
│   └── COLAB_GPU_LAUNCHER.ipynb
├── src/
│   ├── prepare_ner_data.py
│   ├── train_ner_sklearn.py
│   └── deep_ner_models.py
└── reports/
```

## Entity Labels

SROIE receipt fields are mapped to:

| Source field | NER label |
|---|---|
| `company` | `COMPANY` |
| `address` | `ADDRESS` |
| `date` | `DATE` |
| `total` | `TOTAL` |

The Streamlit app normalizes these labels into user-facing fields:

```text
vendor
address
date
total
```

## Notes

- The app works best on receipt images similar to SROIE.
- OCR quality strongly affects NER quality.
- The total amount is post-processed and user-reviewable because receipt OCR often confuses total, subtotal, tax, cash, and change lines.
- Model weights and raw datasets are intentionally excluded from git. Store them locally or in external storage.
