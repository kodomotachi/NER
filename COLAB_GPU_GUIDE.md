# Run NLP-test on Colab GPU

## 1. Upload project to Google Drive

Copy the whole folder:

```text
/Users/kodomotachi/specialist/NLP-test
```

to:

```text
MyDrive/NLP-test
```

The Colab path should become:

```text
/content/drive/MyDrive/NLP-test
```

Make sure these raw files exist in Drive:

```text
/content/drive/MyDrive/NLP-test/data/raw/invoices.csv.zip
/content/drive/MyDrive/NLP-test/data/raw/SROIE_datasetv2.zip
```

If `SROIE_datasetv2.zip` is missing, the launcher will stop before training because the processed JSONL files may be stale.

Meaning:

- `/Users/kodomotachi/specialist/NLP-test` is the project folder on your Mac.
- `MyDrive/NLP-test` is where you put the same folder in Google Drive.
- After Colab mounts Google Drive, `MyDrive/NLP-test` becomes `/content/drive/MyDrive/NLP-test`.

The VS Code note `"Colab": Unknown word` is only a spell-check warning, not a code or Colab error.

## 2. Open Colab and enable GPU

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Then run:

```python
import torch
torch.cuda.is_available()
```

It should return `True`.

## 3. Open launcher notebook

Open:

```text
notebooks/COLAB_GPU_LAUNCHER.ipynb
```

Run cells from top to bottom.

If your project is not at `/content/drive/MyDrive/NLP-test`, edit:

```python
DRIVE_PROJECT_DIR = Path('/content/drive/MyDrive/NLP-test')
```

The launcher copies the project to:

```text
/content/NLP-test
```

and trains from there. This avoids Google Drive mount errors such as:

```text
Transport endpoint is not connected
```

At the end, it syncs these folders back to Drive:

```text
reports/deep_ner/
models/deep_ner/
notebooks/*.executed.ipynb
```

## 4. Recommended order

Run:

1. `00_setup_and_prepare.ipynb`
2. `01_transformer_token_classification.ipynb`
3. `02_transformer_crf_models.ipynb`
4. `03_global_pointer_and_global_context.ipynb`
5. `04_select_best_model.ipynb`

The final leaderboard is:

```text
reports/deep_ner/leaderboard.md
```

## 5. If Colab runs out of VRAM

Keep full data, but reduce batch size:

```python
BATCH_SIZE = 2
```

For Global Pointer / Global Context:

```python
batch_size=1
```

Do not reduce `LIMIT_TRAIN` for the final comparison unless you only want a smoke test.

## 6. Common warnings

These Hugging Face messages are warnings, not fatal errors:

```text
You are not authenticated with the Hugging Face Hub
UNEXPECTED / MISSING weights
```

Public models such as `bert-base-uncased`, `roberta-base`, and `xlm-roberta-base` can still download without a token. `MISSING` classifier weights are expected because the NER classifier head is initialized from scratch.
