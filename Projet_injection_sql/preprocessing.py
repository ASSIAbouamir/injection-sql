#!/usr/bin/env python3
import argparse
import json
import logging
import re
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("preprocess_hq")

# Candidate column names
LABEL_CANDS = {"label", "class", "target", "is_attack", "attack"}
TEXT_CANDS = {"sentence", "text", "payload", "query", "input", "request"}

# -------------------------
# Utils
# -------------------------
def find_file(input_path: str = None) -> Path:
    if input_path:
        p = Path(input_path)
        if p.exists(): return p.resolve()
        raise FileNotFoundError(f"Input file not found: {input_path}")
    cwd = Path.cwd()
    candidates = [cwd / "cleaned_data.csv", cwd / "data" / "processed" / "cleaned_data.csv", cwd / "data" / "cleaned_data.csv"]
    for c in candidates:
        if c.exists(): return c.resolve()
    raise FileNotFoundError("cleaned_data.csv not found in cwd or data/processed/; pass --input /path/file.csv")

def detect_columns(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    sent_col, label_col = None, None
    for cand in TEXT_CANDS:
        if cand in cols_lower:
            sent_col = cols_lower[cand]; break
    for cand in LABEL_CANDS:
        if cand in cols_lower:
            label_col = cols_lower[cand]; break
    if sent_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        if not obj_cols:
            raise ValueError("No text-like column found in CSV.")
        med = {c: df[c].dropna().astype(str).map(len).median() for c in obj_cols}
        sent_col = max(med, key=med.get)
        log.warning(f"No standard text column found. Using '{sent_col}'.")
    if label_col is None:
        for c in df.columns:
            if c == sent_col: continue
            sample = df[c].dropna().astype(str).head(200).tolist()
            if all(re.match(r"^\s*\d+(\.\d+)?\s*$", s) or s.strip().lower() in ("true","false","yes","no","attack","sqli","benign","normal") for s in sample):
                label_col = c
                log.warning(f"No standard label column found. Using '{label_col}'.")
                break
    return sent_col, label_col

def normalize_label(x):
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s in ("1","true","attack","sqli","malicious","yes","pos"): return 1
    if s in ("0","false","benign","normal","no","neg"): return 0
    try:
        v = float(s); return 1 if v != 0 else 0
    except: return None

# Advanced cleaning tailored for SQL payloads
def advanced_clean(text: str) -> str:
    if text is None: return ""
    s = str(text)
    # decode common URL encodings (kept lightweight)
    try:
        s = re.sub(r'%0a|%0d|%0A|%0D', ' ', s)
        s = re.sub(r'%20', ' ', s)
    except Exception:
        pass
    # remove SQL comments (line and block)
    s = re.sub(r'--.*?(\n|$)', ' ', s, flags=re.MULTILINE)
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.DOTALL)
    # remove non-printable characters
    s = re.sub(r'[^\x20-\x7E]', ' ', s)
    # unify spacing
    s = re.sub(r'\s+', ' ', s)
    # remove excessive punctuation runs but keep meaningful ones
    s = re.sub(r'([^\w\s])\1{2,}', r'\1', s)  # compress repeated punctuation
    s = s.strip().lower()
    return s

# Simple balancing helpers
def oversample_minority(df: pd.DataFrame, label_col="Label", seed=42):
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        log.warning("Only one class present; skipping oversample.")
        return df
    max_n = counts.max()
    parts = []
    for lab in counts.index:
        sub = df[df[label_col] == lab]
        if len(sub) < max_n:
            sub = sub.sample(max_n, replace=True, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

def undersample_majority(df: pd.DataFrame, label_col="Label", seed=42):
    counts = df[label_col].value_counts()
    if len(counts) < 2:
        log.warning("Only one class present; skipping undersample.")
        return df
    min_n = counts.min()
    parts = [df[df[label_col] == lab].sample(min_n, random_state=seed) for lab in counts.index]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out

# -------------------------
# Main
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="High-quality preprocessing for SQLi cleaned_data.csv")
    parser.add_argument("--input", type=str, default=None, help="Path to cleaned CSV (auto-find if omitted)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balance", choices=["none","oversample","undersample"], default="oversample", help="Rebalance strategy (default: oversample recommended)")
    parser.add_argument("--analyzer", choices=["char","word"], default="char", help="TF-IDF analyzer (default: char)")
    parser.add_argument("--ngram-min", type=int, default=3, help="ngram min (char)")
    parser.add_argument("--ngram-max", type=int, default=6, help="ngram max (char)")
    parser.add_argument("--max-features", type=int, default=8000, help="TF-IDF max features")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        infile = find_file(args.input)
    except FileNotFoundError as e:
        log.error(str(e))
        return

    log.info(f"Loading file: {infile}")
    df_raw = pd.read_csv(infile, dtype=str, low_memory=False, on_bad_lines="skip")
    log.info(f"Rows read: {len(df_raw)} ; Columns: {list(df_raw.columns)}")

    sent_col, label_col = detect_columns(df_raw)
    log.info(f"Using text column: '{sent_col}' ; label column: '{label_col}'")

    # Build working df
    df = pd.DataFrame()
    df["Sentence"] = df_raw[sent_col].fillna("").astype(str).map(advanced_clean)
    if label_col:
        df["Label"] = df_raw[label_col].map(normalize_label)
    else:
        df["Label"] = None

    # fallback: label heuristic using patterns (only if no labels present)
    if df["Label"].isna().sum() > 0:
        log.info("Some labels are missing; filling NULLs with 0 and/or using heuristic not applied by default.")

    # fill remaining null labels with 0 (conservative)
    df["Label"] = df["Label"].fillna(0).astype(int)

    # drop empty & duplicates
    before = len(df)
    df = df[df["Sentence"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["Sentence"])
    after = len(df)
    log.info(f"After cleaning dedup: {before} -> {after} rows (removed {before-after})")

    # Basic stats
    counts = Counter(df["Label"].tolist())
    total = len(df)
    log.info(f"Label distribution: {counts} (total={total})")

    # Optional balancing
    if args.balance == "oversample":
        df = oversample_minority(df, label_col="Label", seed=args.seed)
        log.info(f"After oversample distribution: {Counter(df['Label'].tolist())}")
    elif args.balance == "undersample":
        df = undersample_majority(df, label_col="Label", seed=args.seed)
        log.info(f"After undersample distribution: {Counter(df['Label'].tolist())}")
    else:
        log.info("No balancing applied.")

    # Train/Test split (stratified if possible)
    strat = df["Label"] if df["Label"].nunique() > 1 else None
    try:
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=strat)
    except Exception as e:
        log.warning(f"Stratified split failed ({e}); using simple split.")
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)

    # Ensure output directory
    out_dir = Path("data") / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save splits textually
    train_out = out_dir / "train.csv"
    test_out = out_dir / "test.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    log.info(f"Saved train ({len(train_df)}) -> {train_out}")
    log.info(f"Saved test  ({len(test_df)}) -> {test_out}")

    # TF-IDF vectorization (recommended: char ngrams)
    vectorizer = TfidfVectorizer(analyzer=args.analyzer, ngram_range=(args.ngram_min, args.ngram_max), max_features=args.max_features)
    X_train = vectorizer.fit_transform(train_df["Sentence"].astype(str).tolist())
    X_test = vectorizer.transform(test_df["Sentence"].astype(str).tolist())

    # Save vectorizer + matrices
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": train_df["Label"].values, "y_test": test_df["Label"].values}, out_dir / "split_tfidf.joblib")
    log.info("Saved tfidf_vectorizer.joblib and split_tfidf.joblib")

    # Build preprocessing report
    report = {
        "input_file": str(infile),
        "rows_before": int(before),
        "rows_after": int(after),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_distribution_after": dict(Counter(df["Label"].tolist())),
        "tfidf": {
            "analyzer": args.analyzer,
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "max_features": args.max_features
        }
    }
    report_path = out_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved preprocessing report -> {report_path}")
    log.info("Preprocessing high-quality complete ✅")

if __name__ == "__main__":
    main()
