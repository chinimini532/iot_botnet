"""
Isolation Forest as an alternative zero-day/OOD detection method, trained
purely on known-class Bot-IoT traffic (unsupervised -- no labels used).
Gives a genuinely different comparison point against the MC Dropout gate,
not just another neural variant.

Usage:
    python src/evaluation/isolation_forest_baseline.py --split-suffix _dedup --tag isoforest
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]

SPLITS_DIR = Path("data/processed/splits")
MODELS_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main(split_suffix: str, tag: str, target_fpr: float):
    print("Loading Bot-IoT training data (known classes, unsupervised) ...")
    train_df = pd.read_csv(SPLITS_DIR / f"bot_iot_train{split_suffix}.csv")
    X_train = train_df[CANONICAL_FEATURES].values

    print("Training Isolation Forest ...")
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_train)
    joblib.dump(iso, MODELS_DIR / f"isolation_forest_{tag}.joblib")

    print("Calibrating threshold on known-class validation data ...")
    calib_df = pd.read_csv(SPLITS_DIR / f"bot_iot_val_calibration{split_suffix}.csv")
    X_calib = calib_df[CANONICAL_FEATURES].values
    calib_scores = -iso.score_samples(X_calib)

    threshold = float(np.percentile(calib_scores, (1 - target_fpr) * 100))
    achieved_fpr = float((calib_scores > threshold).mean())
    print(f"Calibrated threshold: {threshold:.4f}  achieved FPR: {achieved_fpr*100:.2f}%")

    print("\nEvaluating on zero-day test set ...")
    zeroday_df = pd.read_csv(SPLITS_DIR / f"n_baiot_zeroday_test{split_suffix}.csv")
    X_test = zeroday_df[CANONICAL_FEATURES].values
    is_attack = zeroday_df["attack"].values.astype(bool)

    test_scores = -iso.score_samples(X_test)
    flagged = test_scores > threshold

    catch_rate = flagged[is_attack].mean()
    fpr_on_benign = flagged[~is_attack].mean()
    auroc = roc_auc_score(is_attack, test_scores)
    auprc = average_precision_score(is_attack, test_scores)

    print("\n" + "=" * 60)
    print("ISOLATION FOREST ZERO-DAY RESULTS")
    print("=" * 60)
    print(f"Zero-day catch rate: {catch_rate*100:.2f}%")
    print(f"FPR on held-out benign: {fpr_on_benign*100:.2f}%")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    print("\n--- Catch rate by attack family ---")
    for fam in zeroday_df.loc[is_attack, "category"].unique():
        fam_mask = is_attack & (zeroday_df["category"].values == fam)
        print(f"  {fam:<12} n={fam_mask.sum():<10} catch_rate={flagged[fam_mask].mean()*100:.2f}%")

    results = {
        "method": "isolation_forest", "tag": tag, "threshold": threshold,
        "achieved_fpr_on_calibration": achieved_fpr,
        "zero_day_catch_rate": float(catch_rate),
        "fpr_on_held_out_benign": float(fpr_on_benign),
        "auroc": float(auroc), "auprc": float(auprc),
    }
    out_path = RESULTS_DIR / f"isolation_forest_evaluation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-suffix", type=str, default="")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    main(split_suffix=args.split_suffix, tag=args.tag, target_fpr=args.target_fpr)