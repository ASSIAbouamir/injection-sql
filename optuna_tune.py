#!/usr/bin/env python3
"""
optuna_tune.py
Optuna-based hyperparameter search for LogisticRegression, RandomForest, and optionally XGBoost.
Saves best estimators and best params for each model.

Usage:
    python optuna_tune.py --input data/processed/split_tfidf.joblib --outdir data/processed/models_optuna --n_trials 80
"""
import argparse
import json
import os
from pathlib import Path
import joblib
import time
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV

# optional xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def load_split(path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["X_train"], obj["X_test"], np.array(obj["y_train"]), np.array(obj["y_test"])
    raise ValueError("split_tfidf.joblib should be a dict with X_train,X_test,y_train,y_test")

def cv_score_estimator(est, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # use f1 as primary objective
    scores = cross_val_score(est, X, y, scoring=make_scorer(f1_score), cv=skf, n_jobs=-1)
    return float(np.mean(scores))

def objective_lr(trial, X, y):
    C = trial.suggest_loguniform("C", 1e-5, 1e2)
    penalty = trial.suggest_categorical("penalty", ["l1","l2"])
    solver = "saga"  # supports l1/l2
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    est = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=3000, class_weight=class_weight, n_jobs=-1)
    return cv_score_estimator(est, X, y, cv=5)

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_categorical("n_estimators", [100,200,400,800])
    max_depth = trial.suggest_categorical("max_depth", [None, 10, 20, 40, 80])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    est = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42
    )
    return cv_score_estimator(est, X, y, cv=5)

def objective_xgb(trial, X, y):
    n_estimators = trial.suggest_categorical("n_estimators", [100,200,400])
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", [1, 5, 10])
    est = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )
    return cv_score_estimator(est, X, y, cv=5)

def run_study(name, objective, n_trials, X, y, outdir):
    study = optuna.create_study(direction="maximize", study_name=name, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    # save study and best params
    joblib.dump(study, outdir / f"{name}_study.joblib")
    with open(outdir / f"{name}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2)
    return study

def train_best_from_params(model_name, best_params, X_train, y_train):
    if model_name == "logistic":
        est = LogisticRegression(
            C=best_params.get("C"),
            penalty=best_params.get("penalty"),
            solver="saga",
            max_iter=5000,
            class_weight=best_params.get("class_weight"),
            n_jobs=-1,
            random_state=42
        )
    elif model_name == "rf":
        est = RandomForestClassifier(
            n_estimators=int(best_params.get("n_estimators")),
            max_depth=None if best_params.get("max_depth") is None else int(best_params.get("max_depth")),
            min_samples_split=int(best_params.get("min_samples_split")),
            min_samples_leaf=int(best_params.get("min_samples_leaf")),
            class_weight=best_params.get("class_weight"),
            n_jobs=-1,
            random_state=42
        )
    elif model_name == "xgb":
        est = XGBClassifier(
            n_estimators=int(best_params.get("n_estimators")),
            max_depth=int(best_params.get("max_depth")),
            learning_rate=float(best_params.get("learning_rate")),
            subsample=float(best_params.get("subsample")),
            colsample_bytree=float(best_params.get("colsample_bytree")),
            scale_pos_weight=int(best_params.get("scale_pos_weight")),
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError("Unknown model_name")
    est.fit(X_train, y_train)
    return est

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/split_tfidf.joblib")
    parser.add_argument("--outdir", type=str, default="data/processed/models_optuna")
    parser.add_argument("--n_trials", type=int, default=80)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading dataset split:", args.input)
    X_train, X_test, y_train, y_test = load_split(args.input)
    print("Shapes:", getattr(X_train, "shape", None), getattr(X_test, "shape", None))

    # convert sparse to acceptable format for cross_val_score (works fine)
    X = X_train
    y = y_train

    # 1) Logistic
    print("\n[OPT] Starting Optuna study for LogisticRegression")
    study_lr = run_study("logistic_regression", objective_lr, args.n_trials//3, X, y, outdir)
    print("[OPT] LR best params:", study_lr.best_trial.params)

    # train final best LR
    best_lr = train_best_from_params("logistic", study_lr.best_trial.params, X_train, y_train)
    # calibrate
    try:
        calib_lr = CalibratedClassifierCV(best_lr, cv=3)
        calib_lr.fit(X_train, y_train)
        joblib.dump(calib_lr, outdir / "best_lr_calibrated.joblib")
        print("[SAVED] best_lr_calibrated.joblib")
    except Exception as e:
        joblib.dump(best_lr, outdir / "best_lr.joblib")
        print("[SAVED] best_lr.joblib (calibration failed)", e)

    # 2) RandomForest
    print("\n[OPT] Starting Optuna study for RandomForest")
    study_rf = run_study("random_forest", objective_rf, args.n_trials//3, X, y, outdir)
    print("[OPT] RF best params:", study_rf.best_trial.params)
    best_rf = train_best_from_params("rf", study_rf.best_trial.params, X_train, y_train)
    try:
        calib_rf = CalibratedClassifierCV(best_rf, cv=3)
        calib_rf.fit(X_train, y_train)
        joblib.dump(calib_rf, outdir / "best_rf_calibrated.joblib")
        print("[SAVED] best_rf_calibrated.joblib")
    except Exception as e:
        joblib.dump(best_rf, outdir / "best_rf.joblib")
        print("[SAVED] best_rf.joblib (calibration failed)", e)

    # 3) XGBoost (optional)
    if HAS_XGB:
        print("\n[OPT] Starting Optuna study for XGBoost")
        study_xgb = run_study("xgboost", objective_xgb, args.n_trials - (args.n_trials//3)*2, X, y, outdir)
        print("[OPT] XGB best params:", study_xgb.best_trial.params)
        best_xgb = train_best_from_params("xgb", study_xgb.best_trial.params, X_train, y_train)
        joblib.dump(best_xgb, outdir / "best_xgb.joblib")
        print("[SAVED] best_xgb.joblib")
    else:
        print("[INFO] xgboost not available; skip XGB tuning.")

    # Save a summary
    summary = {
        "logistic": study_lr.best_trial.params,
        "random_forest": study_rf.best_trial.params
    }
    if HAS_XGB:
        summary["xgboost"] = study_xgb.best_trial.params
    with open(outdir / "optuna_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[SAVED] optuna_summary.json")

    print("[DONE] Optuna tuning complete.")

if __name__ == "__main__":
    main()
