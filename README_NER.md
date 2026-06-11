# Invoice + SROIE NER workflow

Pipeline này biến dữ liệu hóa đơn dạng bảng và SROIE receipt OCR thành bài toán NER theo BIO tags, rồi so sánh các cách train:

- `baseline_a_only`: chỉ train trên dataset A.
- `mixed_a_plus_b`: train trên A train + phần B không trùng A.
- `baseline_b_only`: chỉ train trên dataset B nếu B có dữ liệu unique.

## Chạy toàn bộ

```bash
cd /Users/kodomotachi/specialist/NLP-test
python src/prepare_ner_data.py
python src/train_ner_sklearn.py
```

Mặc định:

- Dataset A: `data/interim/invoices.csv`
- Dataset B: `data/raw/SROIE_datasetv2.zip`

## Output chính

- `data/processed/a_train.jsonl`, `a_valid.jsonl`, `a_test.jsonl`: split từ A.
- `data/processed/b_unique.jsonl`: phần B đã loại các dòng trùng A để tránh leakage.
- `data/processed/b_valid.jsonl`, `b_test.jsonl`: split tham chiếu của B để evaluate.
- `data/processed/train_a.jsonl`: train baseline A.
- `data/processed/train_a_plus_b.jsonl`: train A+B sau khi chống leakage.
- `data/processed/audit_report.md`: kiểm tra schema, label, duplicate, split.
- `models/*.joblib`: model sklearn đã train.
- `reports/ner_experiment_results.md`: so sánh kết quả trên A valid/test và B valid/test.

## Chạy 6 mô hình deep learning bằng notebook trên Colab GPU

Khuyến nghị dùng Colab GPU cho phần deep learning. Trên Colab:

1. Vào `Runtime > Change runtime type`.
2. Chọn `Hardware accelerator = GPU`.
3. Upload/copy thư mục `NLP-test` vào `/content/NLP-test` hoặc `/content/drive/MyDrive/NLP-test`.
4. Mở notebook trong thư mục `notebooks/` và chạy theo thứ tự bên dưới.

Các notebook nằm trong thư mục `notebooks/`:

1. `00_setup_and_prepare.ipynb`: cài dependency Transformer và tạo lại dữ liệu processed.
2. `01_transformer_token_classification_khoi.ipynb`: chạy Transformer + Token Classification.
3. `02_transformer_crf_models.ipynb`: chạy BERT+CRF, RoBERTa+CRF, XLM-R+CRF.
4. `03_global_pointer_and_global_context.ipynb`: chạy Global Pointer và BERT + Global Context.
5. `04_select_best_model.ipynb`: đọc toàn bộ kết quả, xếp hạng theo `valid.entity_f1`, và chọn model tốt nhất.

Mở Jupyter local nếu chạy trên máy cá nhân:

```bash
cd /Users/kodomotachi/specialist/NLP-test
jupyter notebook
```

Các notebook hiện đã được đặt ở cấu hình train thật:

```python
LIMIT_TRAIN = None
LIMIT_EVAL = None
EPOCHS = 3
MAX_LENGTH = 384
```

Nếu muốn test nhanh trước khi train thật, trong mỗi notebook đặt:

```python
LIMIT_TRAIN = 64
LIMIT_EVAL = 32
EPOCHS = 1
```

Khi train thật, dùng:

```python
LIMIT_TRAIN = None
LIMIT_EVAL = None
EPOCHS = 3
MAX_LENGTH = 384
```

Kết quả deep learning được lưu ở:

- `reports/deep_ner/*.json`
- `reports/deep_ner/leaderboard.md`
- `models/deep_ner/`

## Nhãn NER

Dataset A dạng CSV map các cột thành entity:

- `first_name` -> `FIRST_NAME`
- `last_name` -> `LAST_NAME`
- `email` -> `EMAIL`
- `product_id` -> `PRODUCT_ID`
- `qty` -> `QUANTITY`
- `amount` -> `AMOUNT`
- `invoice_date` -> `INVOICE_DATE`
- `address` -> `ADDRESS`
- `city` -> `CITY`
- `stock_code` -> `STOCK_CODE`
- `job` -> `JOB`

Dataset B dạng SROIE map các field trong `entities/*.txt` thành entity:

- `company` -> `COMPANY`
- `address` -> `ADDRESS`
- `date` -> `DATE`
- `total` -> `TOTAL`

## Lưu ý về trộn dataset

`audit_report.md` sẽ cho biết hai dataset có cùng cấu trúc và cùng ý nghĩa label không. Với dữ liệu hiện tại, A là CSV invoice synthetic còn B là SROIE OCR receipt, nên `Same structure` và `Same label meaning` đều là `False`. Có thể train A+B để thử nghiệm, nhưng kết quả tốt nhất cho B hiện tại là model `baseline_b_only`.

Một số field SROIE không align được vào OCR text do OCR khác ground truth, ví dụ ký tự bị nhận sai. Các field này được bỏ qua khi tạo BIO span và được thống kê trong report.

Nếu muốn đổi dataset:

```bash
python src/prepare_ner_data.py --dataset-a data/interim/invoices.csv --dataset-b data/raw/SROIE_datasetv2.zip
python src/train_ner_sklearn.py
```
