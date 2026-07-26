#!/usr/bin/env python3
"""
build_pipeline.py
Combine TF-IDF vectorizer and best model into a single pipeline.
"""

import joblib
import json
from pathlib import Path
from sklearn.pipeline import Pipeline

def main():
    models_dir = Path("data/processed/models")
    report_path = models_dir / "training_report.json"

    if not report_path.exists():
        raise FileNotFoundError(f"Training report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    selected_model_file = Path(report["selected_model_file"])
    if not selected_model_file.exists():
        raise FileNotFoundError(f"Selected model file not found: {selected_model_file}")

    # Charger le TF-IDF vectorizer
    vect_path = models_dir / "tfidf_vectorizer.joblib"
    if not vect_path.exists():
        vect_path = Path("data/processed/tfidf_vectorizer.joblib")
    if not vect_path.exists():
        raise FileNotFoundError("tfidf_vectorizer.joblib non trouvé.")
    
    print("[INFO] Loading TF-IDF vectorizer and selected model...")
    vectorizer = joblib.load(vect_path)
    model = joblib.load(selected_model_file)

    # Créer le pipeline
    print("[INFO] Creating pipeline...")
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", model)
    ])

    # Sauvegarder le pipeline complet
    out_path = models_dir / "final_pipeline.joblib"
    joblib.dump(pipeline, out_path)
    print(f"[SAVED] Pipeline complet -> {out_path}")

    print("[DONE] You can now load this pipeline and call:")
    print(">>> import joblib")
    print(">>> pipeline = joblib.load('data/processed/models/final_pipeline.joblib')")
    print(">>> pipeline.predict(['Exemple de texte à classifier'])")

if __name__ == "__main__":
    main()
