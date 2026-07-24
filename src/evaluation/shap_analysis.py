"""
SHAP analysis for the paper's supporting XAI subsection (not a core
pillar -- see project notes). Two parts:

1. Standard SHAP feature importance for the XGBoost classifier
   (expected/boilerplate, quick to include).
2. The more useful part: compare SHAP attributions between (a) known
   traffic the gate confidently leaves alone, and (b) zero-day traffic
   the gate correctly flags as unknown -- showing WHICH features drive
   the gate's uncertainty, not just which drive classification.

Usage:
    python src/evaluation/shap_analysis.py --split-suffix _dedup --xgb-tag dedup --mlp-tag dedup2
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import load_model, CANONICAL_FEATURES
from mlp_gate import MCDropoutMLP
from combined_gate import mlp_uncertainty

SPLITS_DIR = Path("data/processed/splits")
MODELS_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 5000  # SHAP is expensive; sample rather than run on millions of rows


def reduce_to_feature_importance(shap_values, n_features: int) -> np.ndarray:
    """
    Reduce SHAP output to one mean-|value| per feature, regardless of
    which shap version/shape convention was returned (list of per-class
    arrays, or a single ndarray with the class axis in different
    positions). Finds whichever axis has size n_features and averages
    over every other axis.
    """
    if isinstance(shap_values, list):
        arr = np.stack([np.abs(sv) for sv in shap_values], axis=0)
    else:
        arr = np.abs(np.asarray(shap_values))

    feature_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
    if not feature_axes:
        raise ValueError(
            f"Could not find an axis of size {n_features} in SHAP output "
            f"shape {arr.shape} -- inspect shap_values manually."
        )
    feat_axis = feature_axes[-1]
    other_axes = tuple(i for i in range(arr.ndim) if i != feat_axis)
    return arr.mean(axis=other_axes)


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


def main(split_suffix: str, xgb_tag: str, mlp_tag: str, calib_tag: str):
    print(f"Loading XGBoost (suffix='{xgb_tag}') ...")
    xgb_model = load_model("xgboost_classifier", suffix=f"_{xgb_tag}" if xgb_tag else "")

    print(f"Loading MLP gate (suffix='{mlp_tag}') ...")
    mlp_model, mlp_scaler = load_mlp_gate(mlp_tag)

    calib_path = RESULTS_DIR / f"calibration_result_{calib_tag}.json"
    with open(calib_path) as f:
        threshold = json.load(f)["threshold"]
    print(f"Using calibrated threshold: {threshold:.6f} (from {calib_path.name})")

    explainer = shap.TreeExplainer(xgb_model)

    print(f"\n--- Part 1: Standard SHAP importance (known test data) ---")
    test_df = pd.read_csv(SPLITS_DIR / f"bot_iot_test{split_suffix}.csv")
    sample = test_df.sample(n=min(SAMPLE_SIZE, len(test_df)), random_state=42)
    X_sample = sample[CANONICAL_FEATURES].values

    shap_values = explainer.shap_values(X_sample)
    print(f"  Raw SHAP output shape info: "
          f"{'list of ' + str(len(shap_values)) + ' arrays, each ' + str(np.array(shap_values[0]).shape) if isinstance(shap_values, list) else np.asarray(shap_values).shape}")
    mean_abs_shap = reduce_to_feature_importance(shap_values, len(CANONICAL_FEATURES))

    importance_df = pd.DataFrame({
        "feature": CANONICAL_FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title("XGBoost Feature Importance (Known-Class Classification)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_path = FIGURES_DIR / "shap_feature_importance.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved figure to {fig_path}")

    print(f"\n--- Part 2: Confident-known vs. correctly-flagged zero-day ---")

    known_unc = mlp_uncertainty(mlp_model, mlp_scaler, X_sample)
    confident_known_mask = known_unc < threshold
    X_confident_known = X_sample[confident_known_mask]
    print(f"  Confident-known samples: {len(X_confident_known):,} / {len(X_sample):,}")

    zeroday_df = pd.read_csv(SPLITS_DIR / f"n_baiot_zeroday_test{split_suffix}.csv")
    zeroday_attacks = zeroday_df[zeroday_df["attack"] == 1]
    zd_sample = zeroday_attacks.sample(n=min(SAMPLE_SIZE, len(zeroday_attacks)), random_state=42)
    X_zd_sample = zd_sample[CANONICAL_FEATURES].values

    zd_unc = mlp_uncertainty(mlp_model, mlp_scaler, X_zd_sample)
    flagged_mask = zd_unc > threshold
    X_flagged_zeroday = X_zd_sample[flagged_mask]
    print(f"  Correctly-flagged zero-day samples: {len(X_flagged_zeroday):,} / {len(X_zd_sample):,}")

    if len(X_confident_known) < 10 or len(X_flagged_zeroday) < 10:
        print("  [!] Too few samples in one group to compare reliably -- skipping Part 2.")
        return

    shap_confident = explainer.shap_values(X_confident_known)
    shap_flagged = explainer.shap_values(X_flagged_zeroday)

    comparison_df = pd.DataFrame({
        "feature": CANONICAL_FEATURES,
        "mean_abs_shap_confident_known": reduce_to_feature_importance(shap_confident, len(CANONICAL_FEATURES)),
        "mean_abs_shap_flagged_zeroday": reduce_to_feature_importance(shap_flagged, len(CANONICAL_FEATURES)),
    })
    comparison_df["difference"] = (
        comparison_df["mean_abs_shap_flagged_zeroday"] -
        comparison_df["mean_abs_shap_confident_known"]
    )
    comparison_df = comparison_df.sort_values("difference", ascending=False)
    print(comparison_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CANONICAL_FEATURES))
    width = 0.35
    ordered = comparison_df.set_index("feature").loc[
        comparison_df.sort_values("mean_abs_shap_flagged_zeroday", ascending=False)["feature"]
    ]
    ax.bar(x - width/2, ordered["mean_abs_shap_confident_known"], width, label="Confident known")
    ax.bar(x + width/2, ordered["mean_abs_shap_flagged_zeroday"], width, label="Flagged zero-day")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered.index, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature Attribution: Confident-Known vs. Flagged Zero-Day")
    ax.legend()
    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "shap_known_vs_zeroday.png"
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    print(f"Saved figure to {fig_path2}")

    comparison_df.to_csv(RESULTS_DIR / "shap_known_vs_zeroday_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-suffix", type=str, default="")
    parser.add_argument("--xgb-tag", type=str, default="")
    parser.add_argument("--mlp-tag", type=str, default="")
    parser.add_argument("--calib-tag", type=str, default="",
                         help="Which calibration_result_*.json to use for the threshold "
                              "(defaults to --mlp-tag if not given).")
    args = parser.parse_args()
    main(split_suffix=args.split_suffix, xgb_tag=args.xgb_tag, mlp_tag=args.mlp_tag,
         calib_tag=args.calib_tag or args.mlp_tag)