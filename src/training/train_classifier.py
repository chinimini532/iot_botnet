"""
Train and compare six classifiers on known Bot-IoT classes, to justify
XGBoost as the Tier-1 primary classifier with an actual comparison
rather than an assumption.

Included: XGBoost (primary candidate), LightGBM, Random Forest,
Decision Tree (unensembled, shows ensembling's added value),
Logistic Regression (linear baseline), Naive Bayes (classic fast baseline).

Excluded: SVM, KNN -- both scale poorly past ~100K rows; at 2.5M+ rows
here they'd require subsampling that would break comparability with the
other models trained on the full set. State this explicitly in the
paper if a reviewer asks.

This step ONLY evaluates in-distribution (known-class) performance --
N-BaIoT / zero-day evaluation comes later, once Tier 2 (MC Dropout gate)
is built.

Usage:
    python src/training/train_classifier.py
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import (
    load_split, save_model,
    train_xgboost, train_lightgbm, train_random_forest,
    train_decision_tree, train_logistic_regression, train_naive_bayes,
)

SPLITS_DIR = Path("data/processed/splits")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model, X_test, y_test, model_name: str, scaler=None) -> dict:
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)

    # XGBoost predicts encoded integers -- decode back to original string labels
    if hasattr(model, "label_encoder_"):
        y_pred = model.label_encoder_.inverse_transform(y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    labels = sorted(set(y_test) | set(y_pred))
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n--- {model_name} classification report ---")
    print(classification_report(y_test, y_pred))

    return {
        "model": model_name,
        "macro_f1": macro_f1,
        "accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
    }


def main(sample_per_class: int = None):
    print("Loading train/test splits ...")
    if sample_per_class is not None:
        print(f"*** SMOKE TEST MODE: subsampling to {sample_per_class} rows/class ***")
    X_train, y_train, _ = load_split(SPLITS_DIR / "bot_iot_train.csv", sample_per_class)
    X_test, y_test, _ = load_split(SPLITS_DIR / "bot_iot_test.csv", sample_per_class)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    all_results = []

    # ── Tree-based models (primary + comparisons) ──────────────────────
    tree_models = [
        ("XGBoost", train_xgboost, "xgboost_classifier"),
        ("LightGBM", train_lightgbm, "lightgbm_classifier"),
        ("Random Forest", train_random_forest, "random_forest_baseline"),
        ("Decision Tree", train_decision_tree, "decision_tree_baseline"),
    ]
    for name, train_fn, save_name in tree_models:
        print(f"\nTraining {name} ...")
        t0 = time.time()
        model = train_fn(X_train, y_train)
        print(f"  done in {time.time() - t0:.1f}s")
        save_model(model, save_name)
        all_results.append(evaluate(model, X_test, y_test, name))

    # ── Logistic Regression (needs scaling) ─────────────────────────────
    print("\nTraining Logistic Regression ...")
    t0 = time.time()
    scaler = StandardScaler()
    scaler.fit(X_train)
    lr_model = train_logistic_regression(X_train, y_train, scaler)
    print(f"  done in {time.time() - t0:.1f}s")
    save_model(lr_model, "logistic_regression_baseline")
    save_model(scaler, "feature_scaler")
    all_results.append(evaluate(lr_model, X_test, y_test, "Logistic Regression", scaler=scaler))

    # ── Naive Bayes ──────────────────────────────────────────────────────
    print("\nTraining Naive Bayes ...")
    t0 = time.time()
    nb_model = train_naive_bayes(X_train, y_train)
    print(f"  done in {time.time() - t0:.1f}s")
    save_model(nb_model, "naive_bayes_baseline")
    all_results.append(evaluate(nb_model, X_test, y_test, "Naive Bayes"))

    # ── Summary table ────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {"model": r["model"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
        for r in all_results
    ]).sort_values("macro_f1", ascending=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (sorted by macro F1)")
    print("=" * 60)
    print(summary.to_string(index=False))

    out_name = "tier1_classifier_results" + ("_smoketest" if sample_per_class else "")
    summary_name = "tier1_comparison_summary" + ("_smoketest" if sample_per_class else "")

    out_path = RESULTS_DIR / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    summary.to_csv(RESULTS_DIR / f"{summary_name}.csv", index=False)
    print(f"\nFull results saved to {out_path}")
    print(f"Summary table saved to {RESULTS_DIR / f'{summary_name}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-per-class", type=int, default=None,
        help="Smoke-test mode: subsample up to N rows per class for a fast "
             "local run (e.g. --sample-per-class 2000). Omit for the full "
             "real run on Kaggle.",
    )
    args = parser.parse_args()
    main(sample_per_class=args.sample_per_class)