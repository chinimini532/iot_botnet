"""
Combined two-tier inference pipeline: this is the script that actually
connects Tier 1 (tree classifier) and Tier 2 (MC Dropout MLP gate) --
until now they've only existed as two separately trained, separately
saved models with no code linking them.

IMPORTANT: UNCERTAINTY_THRESHOLD below is a PLACEHOLDER for pipeline
testing only. The real, calibrated threshold must come from a dedicated
calibration script run against bot_iot_val_calibration.csv (setting the
threshold to hit a target false-positive rate on KNOWN classes) -- that
script does not exist yet. Do not use this placeholder's output for any
real result; it exists only to verify the plumbing works end-to-end.

Usage (against the smoke-test checkpoints already saved):
    python src/evaluation/gate_pipeline.py --suffix _smoketest
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import load_model, load_split, CANONICAL_FEATURES
from mlp_gate import MCDropoutMLP, mc_dropout_predict

MODELS_DIR = Path("models/checkpoints")
SPLITS_DIR = Path("data/processed/splits")

# PLACEHOLDER ONLY -- see module docstring. Real value comes from calibration.
UNCERTAINTY_THRESHOLD = 0.02


def load_mlp_gate(suffix: str = ""):
    checkpoint = torch.load(MODELS_DIR / f"mc_dropout_mlp{suffix}.pt", weights_only=False)
    model = MCDropoutMLP(
        n_features=checkpoint["n_features"],
        n_classes=checkpoint["n_classes"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    scaler = joblib.load(MODELS_DIR / f"mlp_feature_scaler{suffix}.joblib")
    label_encoder = joblib.load(MODELS_DIR / f"mlp_label_encoder{suffix}.joblib")
    return model, scaler, label_encoder


def run_pipeline(X: np.ndarray, xgb_model, mlp_model, mlp_scaler, mlp_label_encoder,
                  threshold: float = UNCERTAINTY_THRESHOLD, n_passes: int = 30):
    """
    Run both tiers on the same input and apply the uncertainty gate.

    Returns a list of dicts, one per sample:
        tree_prediction   -- what Tier 1 (XGBoost) predicted
        uncertainty       -- Tier 2's MC Dropout uncertainty score
        final_decision    -- tree_prediction, OR "UNKNOWN_ZERO_DAY" if
                              uncertainty exceeds the threshold
    """
    # Tier 1: tree classifier prediction
    tree_preds_encoded = xgb_model.predict(X)
    tree_preds = xgb_model.label_encoder_.inverse_transform(tree_preds_encoded)

    # Tier 2: MC Dropout uncertainty
    X_scaled = mlp_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    _, uncertainty = mc_dropout_predict(mlp_model, X_t, n_passes=n_passes)
    uncertainty = uncertainty.numpy()

    results = []
    for tree_pred, unc in zip(tree_preds, uncertainty):
        flagged = unc > threshold
        results.append({
            "tree_prediction": tree_pred,
            "uncertainty": float(unc),
            "final_decision": "UNKNOWN_ZERO_DAY" if flagged else tree_pred,
        })
    return results


def main(suffix: str = "", n_samples: int = 20):
    print(f"Loading Tier-1 (XGBoost) with suffix='{suffix}' ...")
    xgb_model = load_model("xgboost_classifier", suffix=suffix)

    print(f"Loading Tier-2 (MC Dropout MLP) with suffix='{suffix}' ...")
    mlp_model, mlp_scaler, mlp_label_encoder = load_mlp_gate(suffix=suffix)

    print(f"\nLoading a few test rows to run through the pipeline ...")
    X_test, y_test, _ = load_split(SPLITS_DIR / "bot_iot_test.csv", sample_per_class=5)
    X_sample = X_test[:n_samples]
    y_sample = y_test[:n_samples]

    print(f"\n*** PLACEHOLDER threshold={UNCERTAINTY_THRESHOLD} -- "
          f"NOT calibrated, for pipeline verification only ***\n")

    results = run_pipeline(X_sample, xgb_model, mlp_model, mlp_scaler, mlp_label_encoder)

    print(f"{'true_label':<16}{'tree_pred':<16}{'uncertainty':<14}{'final_decision':<20}")
    for true_label, r in zip(y_sample, results):
        print(f"{true_label:<16}{r['tree_prediction']:<16}"
              f"{r['uncertainty']:<14.6f}{r['final_decision']:<20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="",
                         help="Model checkpoint suffix, e.g. '_smoketest'")
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()
    main(suffix=args.suffix, n_samples=args.n_samples)