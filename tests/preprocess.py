import re
import os
import random
from collections import defaultdict

def parse_flat_csv(filepath):
    # Đọc file CSV phẳng và tách thành các bộ (Token, Tag, LayoutID)
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read().replace('\n', ',').replace('\r', '')
    
    # Tách bằng regex để giữ nguyên token có dấu ngoặc kép hoặc dấu phẩy bên trong
    raw_tokens = re.findall(r'"[^"]*"|[^,]+', raw)
    raw_tokens = [t.strip().strip('"') for t in raw_tokens if t.strip()]
    
    triples = []
    for i in range(0, len(raw_tokens) - 2, 3):
        token, tag, layout = raw_tokens[i], raw_tokens[i+1], raw_tokens[i+2]
        # Lọc bỏ token rỗng hoặc tag không hợp lệ
        if token and (tag == 'O' or tag.startswith(('B-', 'I-'))):
            triples.append((token, tag, layout))
    return triples

def fix_bio_tags(doc_triples):
    # Chuẩn hóa BIO: I-XXX không được đứng sau O hoặc B-/I-YYY khác loại
    fixed = []
    for i, (tok, tag, lay) in enumerate(doc_triples):
        if tag.startswith('I-'):
            entity = tag[2:]
            prev_tag = fixed[i-1][1] if i > 0 else 'O'
            prev_entity = prev_tag[2:] if prev_tag.startswith(('B-', 'I-')) else ''
            
            # Nếu I- đứng đầu câu hoặc khác thực thể với token trước -> chuyển thành B-
            if prev_tag == 'O' or prev_entity != entity:
                tag = f'B-{entity}'
        fixed.append((tok, tag, lay))
    return fixed

def segment_documents(triples, max_tokens=250):
    """
    Tách chuỗi token phẳng thành các "tài liệu" giả lập.
    Heuristic: Ngắt khi gặp từ khóa đầu hóa đơn hoặc vượt quá max_tokens.
    """
    docs = []
    current_doc = []
    start_keywords = {'GSTIN', 'TAX', 'INVOICE', 'BILL', 'ORIGINAL', 'DUPLICATE', 
                      'SUPPLIER', 'BUYER', 'RECEIPT', 'CHALLAN'}
    
    for tok, tag, lay in triples:
        is_start = tok.upper() in start_keywords and len(current_doc) > 30
        if is_start or len(current_doc) >= max_tokens:
            if current_doc:
                docs.append(fix_bio_tags(current_doc))
            current_doc = []
        current_doc.append((tok, tag, lay))
        
    if current_doc:
        docs.append(fix_bio_tags(current_doc))
    return docs

def write_conll(docs, filepath, include_layout=False):
    # Ghi ra file CoNLL chuẩn (Token\tTAG) hoặc mở rộng (Token\tTAG\tLayout)
    with open(filepath, 'w', encoding='utf-8') as f:
        for doc in docs:
            for tok, tag, lay in doc:
                if include_layout:
                    f.write(f"{tok}\t{tag}\t{lay}\n")
                else:
                    f.write(f"{tok}\t{tag}\n")
            f.write("\n")  # Dòng trống phân cách tài liệu

def main():
    INPUT_FILE = r"D:\NER\data\raw\MIDD_full_merged.csv"
    OUTPUT_DIR = r"D:\NER\data\processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Đang parse dữ liệu phẳng...")
    triples = parse_flat_csv(INPUT_FILE)
    print(f"Parse thành công {len(triples)} tokens.")
    
    print("Đang phân đoạn tài liệu & chuẩn hóa BIO tags...")
    documents = segment_documents(triples, max_tokens=250)
    print(f"Tạo được {len(documents)} documents.")
    
    # Shuffle & Split 80/10/10 ở cấp độ document
    random.seed(42)
    random.shuffle(documents)
    n = len(documents)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_docs = documents[:train_end]
    val_docs   = documents[train_end:val_end]
    test_docs  = documents[val_end:]
    
    print("Đang ghi file CoNLL...")
    write_conll(train_docs, os.path.join(OUTPUT_DIR, "train.txt"))
    write_conll(val_docs,   os.path.join(OUTPUT_DIR, "valid.txt"))
    write_conll(test_docs,  os.path.join(OUTPUT_DIR, "test.txt"))
    
    print(f"Hoàn tất! Split ratio: Train={len(train_docs)} | Valid={len(val_docs)} | Test={len(test_docs)}")
    print(f"File được lưu tại: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()