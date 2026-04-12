# Intro to NLP — project

Cấu trúc thư mục theo hướng dẫn cookiecutter-style (data, src, notebooks, …).

## Chạy ứng dụng Streamlit

Từ thư mục gốc của project:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Đặt file `domixi.ico` ở thư mục gốc nếu muốn dùng làm favicon; nếu không có, app dùng emoji mặc định.

## Cấu trúc

- `app.py` — điểm vào Streamlit (`streamlit run app.py`).
- `src/` — mã nguồn (data pipeline, features, models, visualization).
- `data/` — dữ liệu raw / interim / processed / external.
- `notebooks/`, `docs/`, `reports/`, `models/`, `references/` — theo mục đích tên thư mục.
- `tests/` — kiểm thử.

Cài đặt editable (tùy chọn): `pip install -e .`
