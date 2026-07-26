#!/usr/bin/env python3
"""
ensemble_models.py
Charge les meilleurs modèles (optuna) et construit VotingClassifier & StackingClassifier,
évalue sur le test set et sauvegarde pipelines finaux (incluant TF-IDF vectorizer).
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

def load_split(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["X_train"], obj["X_test"], np.array(obj["y_train"]), np.array(obj["y_test"])
    raise ValueError("split_tfidf.joblib should be a dict with X_train,X_test,y_train,y_test")

def basic_metrics(y_true, y_pred, y_proba=None):
    m = {}
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_proba is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            m["roc_auc"] = None
    else:
        m["roc_auc"] = None
    return m

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/split_tfidf.joblib")
    parser.add_argument("--optuna_dir", type=str, default="data/processed/models_optuna")
    parser.add_argument("--outdir", type=str, default="data/processed/models_ensembles")
    args = parser.parse_args()

    optuna_dir = Path(args.optuna_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading split:", args.input)
    X_train, X_test, y_train, y_test = load_split(args.input)
    print("Shapes:", getattr(X_train, "shape", None), getattr(X_test, "shape", None))

    # Find candidate models saved from optuna
    candidates = []
    # possible names
    mapping = {
        "lr": ["best_lr_calibrated.joblib", "best_lr.joblib"],
        "rf": ["best_rf_calibrated.joblib", "best_rf.joblib"],
        "xgb": ["best_xgb.joblib"]
    }
    for label, names in mapping.items():
        for n in names:
            p = optuna_dir / n
            if p.exists():
                candidates.append((label, p))
                break

    if not candidates:
        raise FileNotFoundError("No tuned models found in optuna_dir. Run optuna_tune.py first.")

    # load models
    estimators = []
    for label, p in candidates:
        est = joblib.load(p)
        # ensure a readable name
        name = f"{label}"
        estimators.append((name, est))
        print("[INFO] Loaded", label, "->", p)

    # Voting Classifier (soft if probs available)
    voting = VotingClassifier(estimators=estimators, voting="soft")
    print("[TRAIN] Fitting VotingClassifier (on training set)...")
    voting.fit(X_train, y_train)
    y_pred_v = voting.predict(X_test)
    try:
        y_proba_v = voting.predict_proba(X_test)[:,1]
    except Exception:
        y_proba_v = None
    metrics_v = basic_metrics(y_test, y_pred_v, y_proba_v)
    cm_v = confusion_matrix(y_test, y_pred_v).tolist()

    # Stacking Classifier (meta: LogisticRegression)
    final_est = ("meta_lr", LogisticRegression(max_iter=2000, solver="lbfgs"))
    stacking = StackingClassifier(estimators=estimators, final_estimator=final_est[1], stack_method="predict_proba", n_jobs=-1, passthrough=False)
    print("[TRAIN] Fitting StackingClassifier (on training set)...")
    stacking.fit(X_train, y_train)
    y_pred_s = stacking.predict(X_test)
    try:
        y_proba_s = stacking.predict_proba(X_test)[:,1]
    except Exception:
        y_proba_s = None
    metrics_s = basic_metrics(y_test, y_pred_s, y_proba_s)
    cm_s = confusion_matrix(y_test, y_pred_s).tolist()

    # Save pipelines with TF-IDF vectorizer included if available
    vect_path = Path("data/processed/tfidf_vectorizer.joblib")
    if vect_path.exists():
        vectorizer = joblib.load(vect_path)
        pipe_v = Pipeline([("tfidf", vectorizer), ("clf", voting)])
        pipe_s = Pipeline([("tfidf", vectorizer), ("clf", stacking)])
        joblib.dump(pipe_v, outdir / "final_pipeline_voting.joblib")
        joblib.dump(pipe_s, outdir / "final_pipeline_stacking.joblib")
        print("[SAVED] final_pipeline_voting.joblib & final_pipeline_stacking.joblib")
    else:
        # if no TFIDF available (unlikely), save classifiers only
        joblib.dump(voting, outdir / "voting_only.joblib")
        joblib.dump(stacking, outdir / "stacking_only.joblib")
        print("[WARN] tfidf_vectorizer.joblib not found; saved classifiers without vectorizer.")

    # Save report
    report = {
        "voting": {
            "metrics": metrics_v,
            "confusion_matrix": cm_v,
            "estimators": [n for n,_ in estimators]
        },
        "stacking": {
            "metrics": metrics_s,
            "confusion_matrix": cm_s,
            "estimators": [n for n,_ in estimators]
        }
    }
    with open(outdir / "ensemble_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[SAVED] ensemble_report.json")
    print("[DONE] Ensembles built and saved in", outdir)

if __name__ == "__main__":
    main()
