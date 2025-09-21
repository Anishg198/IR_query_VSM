# indexer.py
# Build index.pkl and corpus.pkl from a directory of .txt files
# Removes stopwords during indexing.

import os
import pickle
import math
from collections import defaultdict, Counter

# --- Tokenizer ---
import re
_tok_re = re.compile(r"\b[a-zA-Z]+\b")

def tokenize(text):
    return [t.lower() for t in _tok_re.findall(text)]

# --- Stopwords ---
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

def read_text_file(path):
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def build_index_from_folder(folder_path, min_size_bytes=10):
    corpus = {}
    doc_term_freqs = {}
    postings = defaultdict(list)

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(folder_path, fname)
        raw = read_text_file(path)
        if not raw or len(raw) < min_size_bytes:
            continue

        doc_id = fname  # just filename
        corpus[doc_id] = raw

        tokens = [t for t in tokenize(raw) if t not in STOPWORDS]
        if not tokens:
            continue

        tf = Counter(tokens)
        doc_term_freqs[doc_id] = tf

    # postings + df
    df = {}
    for doc_id, tfmap in doc_term_freqs.items():
        for term, cnt in tfmap.items():
            postings[term].append((doc_id, cnt))
    for term, plist in postings.items():
        df[term] = len(plist)

    # doc lengths (lnc)
    doc_lengths = {}
    for doc_id, tfmap in doc_term_freqs.items():
        sum_sq = 0.0
        for tf in tfmap.values():
            if tf <= 0:
                continue
            w = 1.0 + math.log10(tf)
            sum_sq += w * w
        length = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
        doc_lengths[doc_id] = length

    index_data = {
        "postings": dict(postings),
        "df": df,
        "doc_term_freqs": doc_term_freqs,
        "doc_lengths": doc_lengths,
        "N": len(doc_term_freqs),
    }
    return corpus, index_data

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")

if __name__ == "__main__":
    folder = "corpus"  # fixed to your corpus folder
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    print(f"Indexing files under: {folder}")
    corpus, index_data = build_index_from_folder(folder)
    print(f"Documents indexed: {len(corpus)}")
    print(f"Vocabulary size: {len(index_data['postings'])}")

    save_pickle(index_data, "index.pkl")
    save_pickle(corpus, "corpus.pkl")
    print("Done.")
