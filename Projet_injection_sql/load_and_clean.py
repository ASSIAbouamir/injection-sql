#!/usr/bin/env python3
# clean_data.py

import pandas as pd
import re
from pathlib import Path
import argparse
import sys

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "cleaned_data.csv"

# nettoyage basique
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('"', ' ').replace("'", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)
    return text.strip().lower()

# heuristique pour trouver colonnes Sentence/Label si noms différents
def find_columns(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    # candidate names for sentence and label
    SENT_CANDS = ["sentence", "text", "payload", "query", "input"]
    LABEL_CANDS = ["label", "class", "target", "is_attack", "attack"]
    sent_col = None
    label_col = None
    for cand in SENT_CANDS:
        if cand in cols:
            sent_col = df.columns[cols.index(cand)]
            break
    for cand in LABEL_CANDS:
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    # fallback: first two object-like columns
    if sent_col is None:
        # pick the column with the largest median length (likely text)
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols:
            medians = {c: df[c].dropna().astype(str).map(len).median() for c in obj_cols}
            sent_col = max(medians, key=medians.get)
    if label_col is None:
        # try second column if different and numeric-like
        if len(df.columns) >= 2:
            cand = df.columns[1]
            # if values look like 0/1 or numeric, accept
            sample = df[cand].dropna().astype(str).head(100).tolist()
            if all(re.match(r"^\s*\d+(\.\d+)?\s*$", s) or s.strip().lower() in ("true","false","yes","no","attack","sqli","benign","normal") for s in sample):
                label_col = cand
    return sent_col, label_col

def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ("1","true","attack","sqli","malicious","yes","pos"):
        return 1
    if s in ("0","false","benign","normal","no","neg"):
        return 0
    # numeric fallback
    try:
        v = float(s)
        return 1 if v != 0 else 0
    except Exception:
        return None

def load_csv_robust(path: Path):
    # essaye plusieurs façons de lire pour être robuste
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip", dtype=str)
        return df
    except Exception as e:
        print(f"[WARN] Lecture directe échouée ({e}), tentative engine=python")
    try:
        df = pd.read_csv(path, low_memory=False, engine="python", on_bad_lines="skip", dtype=str)
        return df
    except Exception as e:
        print(f"[ERROR] Impossible de lire le CSV: {e}")
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description="Nettoie CSV SQLi et produit data/processed/cleaned_data.csv")
    parser.add_argument("input", nargs="?", default="SQLiV3.csv", help="Chemin vers le CSV brut (par défaut: SQLiV3.csv)")
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        print(f"[ERROR] Le fichier {path} n'existe pas. Donne le chemin correct ou place le CSV dans le dossier courant.")
        sys.exit(1)

    print(f"[INFO] Lecture de {path}")
    df = load_csv_robust(path)

    sent_col, label_col = find_columns(df)
    if sent_col is None:
        print("[ERROR] Impossible de détecter la colonne texte. Vérifie ton CSV.")
        print("Colonnes trouvées:", df.columns.tolist())
        sys.exit(1)
    print(f"[INFO] Colonne texte détectée: '{sent_col}' ; Colonne label détectée: '{label_col}'")

    data = pd.DataFrame()
    data["Sentence"] = df[sent_col].astype(str).fillna("").map(clean_text)

    if label_col:
        data["Label"] = df[label_col].map(normalize_label)
    else:
        data["Label"] = None

    # Si labels manquants, on met 0 par défaut (ou tu peux changer)
    data["Label"] = data["Label"].fillna(0).astype(int)

    # drop empty sentences
    before = len(data)
    data = data[data["Sentence"].str.strip().astype(bool)]
    data = data.drop_duplicates(subset=["Sentence"])
    after = len(data)
    print(f"[INFO] Lignes initiales: {before} -> après nettoyage et dedup: {after} (removed {before-after})")

    data.to_csv(OUT_FILE, index=False)
    print(f"[✅] Nettoyage terminé. Fichier sauvegardé: {OUT_FILE}")

if __name__ == "__main__":
    main()
