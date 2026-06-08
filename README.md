# Invoice NER — Masking System

An end-to-end Named Entity Recognition pipeline for Vietnamese invoices. The system extracts text from invoice images using PaddleOCR, classifies each token into a structured entity label, and surfaces the annotated results in a Streamlit web UI for human review and correction.

Built as an introductory NLP project, with a focus on comparing labelling approaches and iterating toward a more robust production pipeline.

---

## Features

- **OCR ingestion** — PaddleOCR 3.0 extracts text blocks and bounding boxes from invoice images (PNG / JPG)
- **NER tagging layer** — custom token classifier maps OCR output to invoice entity labels (seller, buyer, item, quantity, unit price, total, date, tax ID, etc.)
- **Multi-stage pipeline** — image → OCR → token segmentation → entity classification → structured JSON output
- **Streamlit web UI** — upload an invoice, view colour-coded entity masks overlaid on the image, and correct any mislabelled tokens
- **Reproducible project layout** — cookiecutter-style structure (data, src, notebooks, models, reports, tests)

---

## Tech stack

| Layer | Library |
|---|---|
| OCR | PaddleOCR 3.0 · PaddlePaddle 3.2.0 |
| Image processing | Pillow · NumPy · OpenCV |
| Web UI | Streamlit ≥ 1.28 |
| Notebooks | Jupyter |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/kodomotachi/NER.git
cd NER

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Place `domixi.ico` in the project root if you want a custom favicon; otherwise the app uses a default emoji icon.

---

## Project layout

```
NER/
├── app.py                  # Streamlit entry point
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Token segmentation and feature extraction
│   ├── models/             # NER model definitions and inference
│   └── visualization/      # Streamlit UI (streamlit_app.py)
├── data/
│   ├── raw/                # Original invoice images
│   ├── interim/            # OCR output (bounding boxes + text)
│   └── processed/          # Labelled token datasets
├── notebooks/              # Exploratory analysis and model experiments
├── models/                 # Saved model weights and configs
├── reports/                # Evaluation results and figures
├── references/             # Papers and reference material
├── tests/                  # Unit tests
├── requirements.txt
└── pyproject.toml
```

---

## Entity labels

| Label | Description |
|---|---|
| `SELLER` | Vendor name |
| `BUYER` | Customer name |
| `INV_DATE` | Invoice date |
| `INV_NO` | Invoice number |
| `TAX_ID` | Tax identification number |
| `ITEM` | Line-item description |
| `QTY` | Quantity |
| `UNIT_PRICE` | Unit price |
| `TOTAL` | Line total / grand total |
| `O` | Outside (not an entity) |

---

## Roadmap

- [ ] Fine-tune a transformer-based NER model (PhoBERT / mBERT) on a labelled invoice dataset
- [ ] Add confidence scores per entity prediction
- [ ] Export structured output to JSON / CSV
- [ ] Support multi-page PDF invoices
- [ ] Evaluation metrics dashboard (precision, recall, F1 per entity class)

---

## Notes

- Currently targets invoices; the OCR and label schema may need adaptation for other locales.
- PaddlePaddle 3.2.0 is pinned — other versions may have protobuf conflicts with PaddleOCR 3.0.
- This is a learning project; model accuracy will improve as more labelled data and fine-tuning are added.
