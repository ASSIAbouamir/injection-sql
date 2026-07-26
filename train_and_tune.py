#!/usr/bin/env python3
"""
train_and_tune.py

Usage examples:
    python train_and_tune.py --input data/processed/split_tfidf.joblib --outdir data/processed/models --cv 5 --n_iter 50
"""

import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# optional: xgboost if available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import warnings
warnings.filterwarnings("ignore")

def load_split(path):
    obj = joblib.load(path)
    # supports either dict or directly joblib dump with keys
    if isinstance(obj, dict):
        X_train = obj.get("X_train")
        X_test = obj.get("X_test")
        y_train = obj.get("y_train")
        y_test  = obj.get("y_test")
    else:
        raise ValueError("split_tfidf.joblib should be a dict with X_train,X_test,y_train,y_test")
    return X_train, X_test, np.array(y_train), np.array(y_test)

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

def run_random_search(clf, param_dist, X, y, cv=5, n_iter=40, scoring="f1", random_state=42, n_jobs=-1):
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, scoring=scoring, cv=cv_split,
                            random_state=random_state, verbose=1, n_jobs=n_jobs, return_train_score=False)
    start = time()
    rs.fit(X, y)
    elapsed = time() - start
    return rs, elapsed

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/split_tfidf.joblib", help="path to split_tfidf.joblib")
    parser.add_argument("--outdir", type=str, default="data/processed/models", help="output dir")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading split:", args.input)
    X_train, X_test, y_train, y_test = load_split(args.input)
    print(f"[INFO] Shapes -> X_train: {getattr(X_train, 'shape', None)}, X_test: {getattr(X_test, 'shape', None)}")

    results = {"runs": []}

    # ----------------------------
    # 1) Logistic Regression (baseline)
    # ----------------------------
    print("\n[STAGE] LogisticRegression (baseline + tuning)")
    lr = LogisticRegression(solver="saga", max_iter=2000, random_state=args.seed, n_jobs=-1)
    lr_params = {
        "C": np.logspace(-4, 3, 20),
        "penalty": ["l1","l2"],
        "class_weight": [None, "balanced"]
    }
    rs_lr, t_lr = run_random_search(lr, lr_params, X_train, y_train, cv=args.cv, n_iter=min(args.n_iter, 30), random_state=args.seed)
    best_lr = rs_lr.best_estimator_
    print("[INFO] LR best params:", rs_lr.best_params_, "best_score(cv):", rs_lr.best_score_)
    # calibrate probabilities (optional)
    calib_lr = CalibratedClassifierCV(best_lr, cv=3)  # calibrate on train
    calib_lr.fit(X_train, y_train)

    y_pred = calib_lr.predict(X_test)
    y_proba = calib_lr.predict_proba(X_test)[:,1]
    metrics = basic_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    results["runs"].append({
        "model": "logistic_regression",
        "best_params": rs_lr.best_params_,
        "cv_best_score": float(rs_lr.best_score_),
        "train_time_sec": t_lr,
        "metrics_test": metrics,
        "confusion_matrix": cm
    })

    # save model
    joblib.dump(calib_lr, outdir / "model_lr_calibrated.joblib")
    print("[SAVED] model_lr_calibrated.joblib")

    # ----------------------------
    # 2) Random Forest
    # ----------------------------
    print("\n[STAGE] RandomForestClassifier (tuning)")
    rf = RandomForestClassifier(random_state=args.seed, n_jobs=-1)
    rf_params = {
        "n_estimators": [100, 200, 400, 800],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2,4,8],
        "min_samples_leaf": [1,2,4],
        "class_weight": [None, "balanced"]
    }
    rs_rf, t_rf = run_random_search(rf, rf_params, X_train, y_train, cv=args.cv, n_iter=min(args.n_iter, 30), random_state=args.seed)
    best_rf = rs_rf.best_estimator_

    # optional: calibrate
    calib_rf = CalibratedClassifierCV(best_rf, cv=3)
    calib_rf.fit(X_train, y_train)
    y_pred = calib_rf.predict(X_test)
    y_proba = calib_rf.predict_proba(X_test)[:,1]
    metrics = basic_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    results["runs"].append({
        "model": "random_forest",
        "best_params": rs_rf.best_params_,
        "cv_best_score": float(rs_rf.best_score_),
        "train_time_sec": t_rf,
        "metrics_test": metrics,
        "confusion_matrix": cm
    })
    joblib.dump(calib_rf, outdir / "model_rf_calibrated.joblib")
    print("[SAVED] model_rf_calibrated.joblib")

    # ----------------------------
    # 3) XGBoost (si dispo)
    # ----------------------------
    if HAS_XGB:
        print("\n[STAGE] XGBoost (tuning)")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=args.seed, n_jobs=-1)
        xgb_params = {
            "n_estimators": [100, 300, 600],
            "max_depth": [3,5,7,9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.8, 1.0],
            "scale_pos_weight": [1, 5, 10]  # if imbalance exists
        }
        rs_xgb, t_xgb = run_random_search(xgb, xgb_params, X_train, y_train, cv=args.cv, n_iter=min(args.n_iter, 30), random_state=args.seed)
        best_xgb = rs_xgb.best_estimator_
        best_xgb.fit(X_train, y_train)
        y_pred = best_xgb.predict(X_test)
        y_proba = best_xgb.predict_proba(X_test)[:,1]
        metrics = basic_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        results["runs"].append({
            "model": "xgboost",
            "best_params": rs_xgb.best_params_,
            "cv_best_score": float(rs_xgb.best_score_),
            "train_time_sec": t_xgb,
            "metrics_test": metrics,
            "confusion_matrix": cm
        })
        joblib.dump(best_xgb, outdir / "model_xgb.joblib")
        print("[SAVED] model_xgb.joblib")
    else:
        print("[INFO] xgboost not installed; skipped.")

    # ----------------------------
    # Choose best model by test f1 (simple rule)
    # ----------------------------
    best_run = max(results["runs"], key=lambda r: r["metrics_test"]["f1"])
    print("\n[SUMMARY] Best on test (by f1):", best_run["model"], "f1=", best_run["metrics_test"]["f1"])

    # copy the corresponding saved file path
    model_filename_map = {
        "logistic_regression": "model_lr_calibrated.joblib",
        "random_forest": "model_rf_calibrated.joblib",
        "xgboost": "model_xgb.joblib"
    }
    chosen_model_file = outdir / model_filename_map.get(best_run["model"], "")
    # also save report
    report = {
        "selected_model": best_run["model"],
        "selected_model_file": str(chosen_model_file),
        "runs": results["runs"]
    }
    report_path = outdir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[SAVED] training_report.json ->", report_path)

    # Save the TFIDF vectorizer also (if in same folder)
    # try to find it at expected place
    possible_vect = Path("data/processed/tfidf_vectorizer.joblib")
    if possible_vect.exists():
        joblib.dump(joblib.load(possible_vect), outdir / "tfidf_vectorizer.joblib")
        print("[SAVED] tfidf_vectorizer.joblib to", outdir)
    else:
        print("[WARN] tfidf_vectorizer.joblib not found at data/processed/ (not copied).")

    print("[DONE] Training and tuning finished.")

if __name__ == "__main__":
    main()
