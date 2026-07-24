"""
Runs ON the CM5 itself. Loads the packaged models and measures REAL
inference latency and memory footprint for:
  1. Tier 1 alone (XGBoost single prediction)
  2. Full two-tier gate (XGBoost + 30-pass MC Dropout MLP)

This is the actual edge-deployment result for the paper -- native
framework inference on real ARM hardware, not simulated or claimed.

Usage (run this ON the CM5, from inside the transferred deploy_package/ folder):
    python cm5_benchmark.py
"""

import json
import time
import resource
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PACKAGE_DIR = Path(__file__).resolve().parent
CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]
N_MC_PASSES = 30


class MCDropoutMLP(nn.Module):
    """Must match src/models/mlp_gate.py's architecture exactly."""
    def __init__(self, n_features, n_classes, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = self.drop1(x)
        x = F.relu(self.fc2(x)); x = self.drop2(x)
        x = F.relu(self.fc3(x)); x = self.drop3(x)
        return self.fc4(x)


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def get_peak_memory_mb():
    """Peak resident set size for this process, in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    print("=" * 60)
    print("CM5 EDGE DEPLOYMENT BENCHMARK")
    print("=" * 60)

    print("\nLoading models ...")
    xgb_model = joblib.load(PACKAGE_DIR / "xgboost_classifier.joblib")
    mlp_scaler = joblib.load(PACKAGE_DIR / "mlp_feature_scaler.joblib")

    checkpoint = torch.load(PACKAGE_DIR / "mc_dropout_mlp.pt", map_location="cpu", weights_only=False)
    mlp_model = MCDropoutMLP(
        n_features=checkpoint["n_features"],
        n_classes=checkpoint["n_classes"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    mlp_model.load_state_dict(checkpoint["model_state_dict"])

    with open(PACKAGE_DIR / "calibration.json") as f:
        threshold = json.load(f)["threshold"]
    print(f"Calibrated threshold: {threshold:.6f}")

    print("Loading benchmark sample ...")
    df = pd.read_csv(PACKAGE_DIR / "benchmark_sample.csv")
    X = df[CANONICAL_FEATURES].values.astype(np.float32)
    n_samples = len(X)
    print(f"  {n_samples:,} rows loaded")

    mem_before = get_peak_memory_mb()

    print(f"\nBenchmarking Tier 1 (XGBoost) alone, per-sample latency ...")
    latencies_xgb = []
    for i in range(n_samples):
        x = X[i:i+1]
        t0 = time.perf_counter()
        _ = xgb_model.predict(x)
        latencies_xgb.append((time.perf_counter() - t0) * 1000)

    print(f"Benchmarking full two-tier gate (XGBoost + {N_MC_PASSES}-pass MC Dropout) ...")
    latencies_full = []
    X_scaled = mlp_scaler.transform(X)
    for i in range(n_samples):
        x_raw = X[i:i+1]
        x_scaled = torch.tensor(X_scaled[i:i+1], dtype=torch.float32)

        t0 = time.perf_counter()
        _ = xgb_model.predict(x_raw)

        enable_mc_dropout(mlp_model)
        probs = []
        with torch.no_grad():
            for _ in range(N_MC_PASSES):
                logits = mlp_model(x_scaled)
                probs.append(F.softmax(logits, dim=1))
        stacked = torch.stack(probs, dim=0)
        uncertainty = stacked.var(dim=0).mean(dim=1).item()
        _ = uncertainty > threshold

        latencies_full.append((time.perf_counter() - t0) * 1000)

    mem_after = get_peak_memory_mb()

    results = {
        "n_samples": n_samples,
        "n_mc_passes": N_MC_PASSES,
        "xgboost_only_ms": {
            "mean": float(np.mean(latencies_xgb)),
            "median": float(np.median(latencies_xgb)),
            "p95": float(np.percentile(latencies_xgb, 95)),
            "std": float(np.std(latencies_xgb)),
        },
        "full_gate_ms": {
            "mean": float(np.mean(latencies_full)),
            "median": float(np.median(latencies_full)),
            "p95": float(np.percentile(latencies_full, 95)),
            "std": float(np.std(latencies_full)),
        },
        "throughput_samples_per_sec_full_gate": float(1000 / np.mean(latencies_full)),
        "peak_memory_mb": float(mem_after),
        "memory_delta_mb": float(mem_after - mem_before),
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"XGBoost alone:      mean={results['xgboost_only_ms']['mean']:.3f} ms  "
          f"median={results['xgboost_only_ms']['median']:.3f} ms  "
          f"p95={results['xgboost_only_ms']['p95']:.3f} ms")
    print(f"Full two-tier gate: mean={results['full_gate_ms']['mean']:.3f} ms  "
          f"median={results['full_gate_ms']['median']:.3f} ms  "
          f"p95={results['full_gate_ms']['p95']:.3f} ms")
    print(f"Throughput (full gate): {results['throughput_samples_per_sec_full_gate']:.1f} samples/sec")
    print(f"Peak memory (RSS): {results['peak_memory_mb']:.1f} MB")

    out_path = PACKAGE_DIR / "cm5_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    print("Copy this JSON file back to your main machine for figure/table generation.")


if __name__ == "__main__":
    main()