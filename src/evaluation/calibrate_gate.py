"""
Calibrate the COMBINED uncertainty threshold (XGBoost confidence + MLP
MC Dropout uncertainty) using bot_iot_val_calibration (known classes,
includes N-BaIoT benign supplement).

Usage:
    python src/evaluation/calibrate_gate.py --split-suffix _dedup --tag combined --xgb-tag dedup --mlp-tag dedup2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import load_split, load_model
from mlp_gate import MCDropoutMLP
from combined_gate import xgb_uncertainty, mlp_uncertainty, fit_calibration_stats, combined_score

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


def main(split_suffix: str, tag: str, xgb_tag: str, mlp_tag: str,
         target_fpr: float, n_passes: int):
    print(f"Loading Tier-1 XGBoost (suffix='{xgb_tag}') ...")
    xgb_model = load_model("xgboost_classifier", suffix=f"_{xgb_tag}" if xgb_tag else "")

    print(f"Loading Tier-2 MC Dropout MLP (suffix='{mlp_tag}') ...")
    mlp_model, mlp_scaler = load_mlp_gate(mlp_tag)

    calib_path = SPLITS_DIR / f"bot_iot_val_calibration{split_suffix}.csv"
    print(f"Loading calibration data from {calib_path.name} ...")
    X_calib, y_calib, _ = load_split(calib_path)
    print(f"  {X_calib.shape[0]:,} known-class calibration rows")

    print("Computing XGBoost confidence ...")
    xgb_unc = xgb_uncertainty(xgb_model, X_calib)

    print(f"Computing MLP MC Dropout uncertainty ({n_passes} passes) ...")
    mlp_unc = mlp_uncertainty(mlp_model, mlp_scaler, X_calib, n_passes=n_passes)

    print("Fitting calibration stats and computing combined score ...")
    stats = fit_calibration_stats(xgb_unc, mlp_unc)
    combined = combined_score(xgb_unc, mlp_unc, stats)

    print(f"\nCombined score distribution on KNOWN traffic:")
    print(f"  min={combined.min():.4f}  max={combined.max():.4f}  "
          f"mean={combined.mean():.4f}  median={np.median(combined):.4f}")

    threshold = float(np.percentile(combined, (1 - target_fpr) * 100))
    achieved_fpr = float((combined > threshold).mean())

    print(f"\nTarget FPR on knowns: {target_fpr*100:.1f}%")
    print(f"Calibrated threshold: {threshold:.4f}")
    print(f"Achieved FPR at this threshold: {achieved_fpr*100:.2f}%")

    print(f"\n--- FPR by class (at threshold={threshold:.4f}) ---")
    for cls in np.unique(y_calib):
        mask = y_calib == cls
        cls_fpr = (combined[mask] > threshold).mean()
        print(f"  {cls:<16} n={mask.sum():<8} FPR={cls_fpr*100:.2f}%")

    result = {
        "tag": tag,
        "xgb_tag": xgb_tag,
        "mlp_tag": mlp_tag,
        "target_fpr": target_fpr,
        "threshold": threshold,
        "achieved_fpr": achieved_fpr,
        "calibration_stats": stats,
        "n_calibration_samples": int(X_calib.shape[0]),
        "n_mc_passes": n_passes,
    }
    out_path = RESULTS_DIR / f"calibration_result_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved calibration result to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-suffix", type=str, default="")
    parser.add_argument("--tag", type=str, required=True,
                         help="Name for this calibration run, e.g. --tag combined")
    parser.add_argument("--xgb-tag", type=str, default="",
                         help="Tier-1 XGBoost checkpoint suffix to load")
    parser.add_argument("--mlp-tag", type=str, default="",
                         help="Tier-2 MLP checkpoint suffix to load")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--n-passes", type=int, default=30)
    args = parser.parse_args()
    main(split_suffix=args.split_suffix, tag=args.tag, xgb_tag=args.xgb_tag,
         mlp_tag=args.mlp_tag, target_fpr=args.target_fpr, n_passes=args.n_passes)