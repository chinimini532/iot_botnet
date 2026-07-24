"""
THE real test, using the COMBINED signal (XGBoost confidence + MLP MC
Dropout uncertainty) against n_baiot_zeroday_test.

Usage:
    python src/evaluation/evaluate_zeroday.py --split-suffix _dedup --tag combined --xgb-tag dedup --mlp-tag dedup2
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import load_model, CANONICAL_FEATURES
from mlp_gate import MCDropoutMLP
from combined_gate import xgb_uncertainty, mlp_uncertainty, combined_score

MODELS_DIR = Path("models/checkpoints")
SPLITS_DIR = Path("data/processed/splits")
RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_mlp_gate(suffix: str):
    checkpoint = torch.load(MODELS_DIR / f"mc_dropout_mlp{suffix}.pt", weights_only=False)
    model = MCDropoutMLP(
        n_features=checkpoint["n_features"],
        n_classes=checkpoint["n_classes"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = joblib.load(MODELS_DIR / f"mlp_feature_scaler{suffix}.joblib")
    return model, scaler


def main(split_suffix: str, tag: str, xgb_tag: str, mlp_tag: str, n_passes: int):
    calib_path = RESULTS_DIR / f"calibration_result_{tag}.json"
    with open(calib_path) as f:
        calib = json.load(f)
    threshold = calib["threshold"]
    stats = calib["calibration_stats"]
    print(f"Loaded calibrated threshold: {threshold:.4f} (from {calib_path.name})")

    print(f"\nLoading Tier-1 XGBoost (suffix='{xgb_tag}') ...")
    xgb_model = load_model("xgboost_classifier", suffix=f"_{xgb_tag}" if xgb_tag else "")

    print(f"Loading Tier-2 MC Dropout MLP (suffix='{mlp_tag}') ...")
    mlp_model, mlp_scaler = load_mlp_gate(mlp_tag)

    zeroday_path = SPLITS_DIR / f"n_baiot_zeroday_test{split_suffix}.csv"
    print(f"\nLoading zero-day test set from {zeroday_path.name} ...")
    df = pd.read_csv(zeroday_path)
    X_test = df[CANONICAL_FEATURES].values
    is_attack = df["attack"].values
    print(f"  {X_test.shape[0]:,} rows  "
          f"({is_attack.sum():,} attack, {(~is_attack.astype(bool)).sum():,} benign)")

    print("Computing XGBoost confidence ...")
    xgb_unc = xgb_uncertainty(xgb_model, X_test)

    print(f"Computing MLP MC Dropout uncertainty ({n_passes} passes, "
          f"this may take a while on {X_test.shape[0]:,} rows) ...")
    mlp_unc = mlp_uncertainty(mlp_model, mlp_scaler, X_test, n_passes=n_passes)

    combined = combined_score(xgb_unc, mlp_unc, stats)
    flagged_unknown = combined > threshold

    attack_mask = is_attack.astype(bool)
    benign_mask = ~attack_mask

    catch_rate = flagged_unknown[attack_mask].mean()
    fpr_on_benign = flagged_unknown[benign_mask].mean()
    auroc = roc_auc_score(attack_mask, combined)
    auprc = average_precision_score(attack_mask, combined)

    print("\n" + "=" * 60)
    print("ZERO-DAY EVALUATION RESULTS (combined signal)")
    print("=" * 60)
    print(f"Zero-day catch rate (attacks flagged UNKNOWN): {catch_rate*100:.2f}%")
    print(f"FPR on held-out benign (benign flagged UNKNOWN): {fpr_on_benign*100:.2f}%")
    print(f"AUROC (combined score vs. is-attack): {auroc:.4f}")
    print(f"AUPRC (combined score vs. is-attack): {auprc:.4f}")

    print("\n--- Catch rate by attack family ---")
    for fam in df.loc[attack_mask, "category"].unique():
        fam_mask = attack_mask & (df["category"].values == fam)
        fam_catch = flagged_unknown[fam_mask].mean()
        print(f"  {fam:<12} n={fam_mask.sum():<10} catch_rate={fam_catch*100:.2f}%")

    auroc_xgb_alone = roc_auc_score(attack_mask, xgb_unc)
    auroc_mlp_alone = roc_auc_score(attack_mask, mlp_unc)
    print(f"\n--- Component signals alone (for comparison) ---")
    print(f"  XGBoost confidence alone AUROC: {auroc_xgb_alone:.4f}")
    print(f"  MLP uncertainty alone AUROC:    {auroc_mlp_alone:.4f}")
    print(f"  Combined AUROC:                 {auroc:.4f}")

    results = {
        "tag": tag, "xgb_tag": xgb_tag, "mlp_tag": mlp_tag,
        "threshold_used": threshold,
        "n_test_rows": int(X_test.shape[0]),
        "n_attack": int(attack_mask.sum()),
        "n_benign": int(benign_mask.sum()),
        "zero_day_catch_rate": float(catch_rate),
        "fpr_on_held_out_benign": float(fpr_on_benign),
        "auroc_combined": float(auroc),
        "auprc_combined": float(auprc),
        "auroc_xgb_alone": float(auroc_xgb_alone),
        "auroc_mlp_alone": float(auroc_mlp_alone),
    }
    out_path = RESULTS_DIR / f"zeroday_evaluation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-suffix", type=str, default="")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--xgb-tag", type=str, default="")
    parser.add_argument("--mlp-tag", type=str, default="")
    parser.add_argument("--n-passes", type=int, default=30)
    args = parser.parse_args()
    main(split_suffix=args.split_suffix, tag=args.tag, xgb_tag=args.xgb_tag,
         mlp_tag=args.mlp_tag, n_passes=args.n_passes)