import re
import os
import csv
import random
from collections import Counter

def preprocess_midd_csv(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    clean_ocr: bool = True
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Tỷ lệ chia phải bằng 1.0"
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Regex làm sạch ký tự rác OCR
    ocr_clean_pattern = re.compile(r'[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF\u00A0]')
    # Marker nhận biết bắt đầu hóa đơn mới
    doc_start_pattern = re.compile(r'^(Invoice|TAX\s*INVOICE|GSTIN|Original\s*Invoice)$', re.IGNORECASE)

    raw_tokens = []
    print(f"Đang đọc {input_path}...")
    
    # 1. Đọc toàn bộ file dưới dạng danh sách phẳng
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            raw_tokens.extend([cell.strip() for cell in row if cell.strip()])

    # 2. Ghép cặp (Token, Tag) từ dãy phẳng
    paired_data = []
    i = 0
    while i < len(raw_tokens) - 1:
        token, tag = raw_tokens[i], raw_tokens[i+1]
        # Kiểm tra tag có đúng định dạng BIO/O không
        if re.match(r'^(O|B-[A-Za-z0-9_]+|I-[A-Za-z0-9_]+)$', tag):
            if clean_ocr:
                token = ocr_clean_pattern.sub('', token)
                token = re.sub(r'\s+', ' ', token).strip()
            if token:  # Chỉ giữ token còn nội dung
                paired_data.append((token, tag))
            i += 2
        else:
            # Nếu không khớp cặp, bỏ qua token hiện tại và thử lại
            i += 1

    print(f"Ghép được {len(paired_data)} cặp (Token, Tag) hợp lệ.")

    # 3. Tách thành các document (mỗi hóa đơn)
    documents = []
    current_doc = []
    for token, tag in paired_data:
        # Gặp marker bắt đầu hóa đơn + doc hiện tại đủ dài -> ngắt doc
        if doc_start_pattern.search(token) and len(current_doc) > 15:
            documents.append(current_doc)
            current_doc = []
        current_doc.append((token, tag))
    if current_doc:
        documents.append(current_doc)

    print(f"Tách được {len(documents)} documents (hóa đơn).")

    # 4. Xáo trộn & Chia tập
    random.shuffle(documents)
    n = len(documents)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': documents[:train_end],
        'valid': documents[train_end:val_end],
        'test': documents[val_end:]
    }

    # 5. Ghi file CoNLL chuẩn
    for split_name, docs in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.conll")
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(docs):
                for token, tag in doc:
                    f.write(f"{token}\t{tag}\n")
                if i < len(docs) - 1:
                    f.write("\n")
        print(f"Đã lưu {split_name}.conll ({len(docs)} docs, {sum(len(d) for d in docs)} tokens)")
        
    return splits

if __name__ == "__main__":
    preprocess_midd_csv(
        input_path="D:\\NER\\data\\raw\\MIDD.csv",
        output_dir="D:\\NER\\data\\processed",
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        seed=42, clean_ocr=True
    )