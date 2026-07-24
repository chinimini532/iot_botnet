"""
Combined-signal uncertainty: fuses Tier-1's own prediction confidence
(XGBoost's max softmax probability) with Tier-2's MC Dropout epistemic
uncertainty into one score. Two independently weak signals can combine
into a stronger one -- this is standard ensemble-uncertainty practice,
not just "adding more stuff."

Both raw signals are z-score normalized using statistics fit on the
KNOWN-class calibration set, then summed. The same fitted stats must be
reused at evaluation time (never re-fit on test data).
"""

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_gate import mc_dropout_predict


def xgb_uncertainty(xgb_model, X: np.ndarray) -> np.ndarray:
    """1 - max predicted probability. High value = XGBoost itself is unsure."""
    proba = xgb_model.predict_proba(X)
    return 1.0 - proba.max(axis=1)


def mlp_uncertainty(mlp_model, mlp_scaler, X: np.ndarray, n_passes: int = 30) -> np.ndarray:
    X_scaled = mlp_scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    _, uncertainty = mc_dropout_predict(mlp_model, X_t, n_passes=n_passes)
    return uncertainty.numpy()


def fit_calibration_stats(xgb_unc: np.ndarray, mlp_unc: np.ndarray) -> dict:
    return {
        "xgb_mean": float(xgb_unc.mean()),
        "xgb_std": float(xgb_unc.std() + 1e-8),  # avoid div-by-zero
        "mlp_mean": float(mlp_unc.mean()),
        "mlp_std": float(mlp_unc.std() + 1e-8),
    }


def combined_score(xgb_unc: np.ndarray, mlp_unc: np.ndarray, stats: dict) -> np.ndarray:
    z_xgb = (xgb_unc - stats["xgb_mean"]) / stats["xgb_std"]
    z_mlp = (mlp_unc - stats["mlp_mean"]) / stats["mlp_std"]
    return z_xgb + z_mlp