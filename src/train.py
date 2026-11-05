from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (precision_recall_curve, roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, classification_report, RocCurveDisplay,
                             PrecisionRecallDisplay)
from src.data_validation import validate_transactions
from src.features import split_features_target, build_preprocessor

def choose_threshold(y_true, probs, objective="f1", min_precision=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    if objective == "recall_at_precision":
        valid = np.where(p >= min_precision)[0]
        if len(valid):
            return float(thresholds[valid[np.argmax(r[valid])]])
    return float(thresholds[int(np.argmax(f1))])

def build_models(seed=42, use_smote=True):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "RandomForest": RandomForestClassifier(
            n_estimators=350, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            reg_lambda=2.0, eval_metric="logloss",
            random_state=seed, n_jobs=-1
        )
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable; skipping it ({exc})")
    return models

def make_pipeline(preprocessor, estimator, use_smote):
    if use_smote:
        try:
            from imblearn.pipeline import Pipeline
            from imblearn.over_sampling import SMOTE
            return Pipeline([
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", estimator)
            ])
        except ImportError:
            warnings.warn("imbalanced-learn unavailable; training without SMOTE.")
    return SkPipeline([("preprocess", preprocessor), ("model", estimator)])

def metrics_at_threshold(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/creditcard.csv")
    ap.add_argument("--target", default="Class")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--threshold-objective", choices=["f1", "recall_at_precision"], default="f1")
    ap.add_argument("--min-precision", type=float, default=0.80)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    validate_transactions(df, args.target)
    X, y = split_features_target(df, args.target)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    preprocessor = build_preprocessor(X_train)
    candidates = build_models(use_smote=not args.no_smote)
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(preprocessor, estimator, use_smote=not args.no_smote)
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_probs, args.threshold_objective, args.min_precision)
        results[name] = metrics_at_threshold(y_val, val_probs, threshold)
        fitted[name] = (pipe, threshold)
        print(f"{name}: validation PR-AUC={results[name]['pr_auc']:.4f}, "
              f"precision={results[name]['precision']:.3f}, recall={results[name]['recall']:.3f}")

    # Select by validation PR-AUC, a useful ranking metric for rare positives.
    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model, threshold = fitted[best_name]
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_probs, threshold)

    out_model, out_report = Path(args.output_dir), Path(args.report_dir)
    out_model.mkdir(parents=True, exist_ok=True)
    out_report.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_model / "fraud_model.joblib")
    metadata = {
        "model_name": best_name, "threshold": threshold,
        "feature_columns": list(X.columns), "target": args.target,
        "threshold_objective": args.threshold_objective,
        "smote_requested": not args.no_smote,
        "selection_metric": "validation_pr_auc"
    }
    (out_model / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report = {
        "best_model": best_name,
        "validation_comparison": results,
        "test_metrics": test_metrics,
        "test_fraud_prevalence": float(y_test.mean()),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
    }
    (out_report / "metrics.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.array(test_metrics["confusion_matrix"]),
                           display_labels=["Legitimate", "Fraud"]).plot(ax=ax, values_format="d")
    ax.set_title(f"{best_name} — held-out test set")
    fig.tight_layout()
    fig.savefig(out_report / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"\nSelected: {best_name}")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2))
    print(f"Artifacts saved to {out_model} and {out_report}")

if __name__ == "__main__":
    main()
