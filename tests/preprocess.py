import os
import json
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
TARGET_LANG = "en"

# WikiANN tag mapping (CoNLL-2003 style)
WIKIANN_TAG_MAP = {
    0: "O", 1: "B-PER", 2: "I-PER",
    3: "B-LOC", 4: "I-LOC",
    5: "B-ORG", 6: "I-ORG",
    7: "B-MISC", 8: "I-MISC"
}

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(f"{OUT_DIR}/wikiann", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/nemotron", exist_ok=True)

def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} samples to {filepath}")

def verify_bio_consistency(tokens, tags):
    """Cảnh báo nếu I- tag đứng sau O- hoặc B- khác loại"""
    for i, (t, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("I-"):
            entity_type = tag[2:]
            prev_tag = tags[i-1] if i > 0 else "O"
            if prev_tag not in [f"B-{entity_type}", f"I-{entity_type}"]:
                return False, f"Mismatch at token {i} ('{t}'): {tag} after {prev_tag}"
    return True, "OK"

def char_spans_to_bio(text, entities):
    """
    Chuyển character-level spans (start, end, label) thành word-level BIO tags.
    Dùng whitespace tokenization để tương thích CRF/BiLSTM/spaCy.
    """
    tokens = []
    bio_tags = []
    
    # Tìm tất cả words (bỏ qua pure whitespace)
    word_matches = list(re.finditer(r'\S+', text))
    
    for match in word_matches:
        start, end = match.span()
        word = match.group()
        tokens.append(word)
        
        # Tìm entity overlap (giả sử không có nested entities cho PII)
        tag = "O"
        for ent in entities:
            e_start, e_end, e_label = ent.get("start"), ent.get("end"), ent.get("label", ent.get("type", "UNKNOWN"))
            if e_start is None or e_end is None:
                continue
                
            # Overlap condition
            if start < e_end and end > e_start:
                if start == e_start:
                    tag = f"B-{e_label}"
                else:
                    tag = f"I-{e_label}"
                break  # Priority: first match wins
        bio_tags.append(tag)
        
    return tokens, bio_tags

# ─────────────────────────────────────────────────────────────
# PROCESS WIKIANN
# ─────────────────────────────────────────────────────────────
def process_wikiann():
    print("\n" + "="*60)
    print("Processing WikiANN (English only)")
    print("="*60)
    
    dataset = load_dataset("wikiann", TARGET_LANG)
    all_processed = {}
    
    for split in dataset.keys():
        print(f"\nSplit: {split}")
        split_data = []
        
        for item in tqdm(dataset[split], desc=f"  Converting {split}"):
            # Filter English (dù đã load 'en', vẫn verify cột lang)
            if item.get("langs", [TARGET_LANG])[0] != TARGET_LANG:
                continue
                
            tokens = item["tokens"]
            raw_tags = item["ner_tags"]
            
            # Map integer tags → string BIO
            bio_tags = [WIKIANN_TAG_MAP.get(t, "O") for t in raw_tags]
            
            # Validation
            valid, msg = verify_bio_consistency(tokens, bio_tags)
            if not valid:
                print(f"BIO warning: {msg} (sample skipped)")
                continue
                
            split_data.append({"tokens": tokens, "ner_tags": bio_tags})
            
        all_processed[split] = split_data
        save_jsonl(split_data, f"{OUT_DIR}/wikiann/wikiann_{TARGET_LANG}_{split}.jsonl")
        
    return all_processed

# ─────────────────────────────────────────────────────────────
# PROCESS NEMOTRON-PII
# ─────────────────────────────────────────────────────────────
def process_nemotron():
    print("\n" + "="*60)
    print("Processing NVIDIA Nemotron-PII (English only)")
    print("="*60)
    
    # Nemotron chủ yếu là English US, nhưng ta vẫn filter nếu có cột locale
    dataset = load_dataset("nvidia/Nemotron-PII")
    
    # Chỉ lấy split có sẵn (thường là 'train' và 'test')
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    for split in available_splits:
        print(f"\nSplit: {split}")
        split_data = []
        
        for item in tqdm(dataset[split], desc=f"  Aligning {split}"):
            text = item.get("text", "")
            entities = item.get("ner", item.get("entities", []))
            
            if not text or not entities:
                continue
                
            # Convert spans → BIO
            tokens, bio_tags = char_spans_to_bio(text, entities)
            
            if len(tokens) != len(bio_tags):
                continue  # Skip misaligned samples
                
            valid, msg = verify_bio_consistency(tokens, bio_tags)
            if not valid:
                continue
                
            split_data.append({
                "text": text,
                "tokens": tokens,
                "ner_tags": bio_tags
            })
            
        save_jsonl(split_data, f"{OUT_DIR}/nemotron/nemotron_pii_{split}.jsonl")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_dirs()
    
    print("Starting preprocessing pipeline...")
    print(f"Language filter: {TARGET_LANG}")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Output dir: {OUT_DIR}")
    
    # 1. WikiANN
    wikiann_data = process_wikiann()
    
    # 2. Nemotron-PII
    process_nemotron()
    
    # 3. Summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"WikiANN: {sum(len(v) for v in wikiann_data.values())} samples (EN)")
    print(f"Nemotron-PII: Check output files for counts")
    print(f"Processed data saved to: {OUT_DIR}/")