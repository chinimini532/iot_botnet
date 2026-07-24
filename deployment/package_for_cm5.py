"""
Package everything needed for CM5 deployment into one folder, ready to
scp to the device.

Run this locally/on Kaggle AFTER training + calibration are finalized.

Usage:
    python deployment/package_for_cm5.py --xgb-tag dedup --mlp-tag dedup2 --calib-tag dedup2

Note on tag format: --xgb-tag is passed WITHOUT a leading underscore
(e.g. "dedup") -- classifier.py's save_model adds the underscore itself.
--mlp-tag is also passed WITHOUT a leading underscore (e.g. "dedup2")
-- train_gate.py's naming convention does not add one automatically.
Check your actual models/checkpoints/ filenames if unsure.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

MODELS_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("results/metrics")
SPLITS_DIR = Path("data/processed/splits")
OUT_DIR = Path("deployment/deploy_package")

N_BENCHMARK_SAMPLES = 2000  # enough for a stable latency average, small enough to transfer easily


def main(split_suffix: str, xgb_tag: str, mlp_tag: str, calib_tag: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Copying model files ...")
    shutil.copy(MODELS_DIR / f"xgboost_classifier_{xgb_tag}.joblib", OUT_DIR / "xgboost_classifier.joblib")
    shutil.copy(MODELS_DIR / f"mc_dropout_mlp{mlp_tag}.pt", OUT_DIR / "mc_dropout_mlp.pt")
    shutil.copy(MODELS_DIR / f"mlp_feature_scaler{mlp_tag}.joblib", OUT_DIR / "mlp_feature_scaler.joblib")

    print("Copying calibrated threshold ...")
    calib_path = RESULTS_DIR / f"calibration_result_{calib_tag}.json"
    with open(calib_path) as f:
        calib = json.load(f)
    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump({"threshold": calib["threshold"], "target_fpr": calib["target_fpr"]}, f, indent=2)

    print(f"Sampling {N_BENCHMARK_SAMPLES} rows from zero-day test set for on-device benchmarking ...")
    zeroday_df = pd.read_csv(SPLITS_DIR / f"n_baiot_zeroday_test{split_suffix}.csv")
    sample = zeroday_df.sample(n=min(N_BENCHMARK_SAMPLES, len(zeroday_df)), random_state=42)
    sample.to_csv(OUT_DIR / "benchmark_sample.csv", index=False)

    manifest = {
        "xgb_tag": xgb_tag, "mlp_tag": mlp_tag, "calib_tag": calib_tag,
        "n_benchmark_samples": len(sample),
        "required_packages": {
            "xgboost": "2.0.3", "torch": "2.3.1", "scikit-learn": "1.5.0",
            "joblib": "1.4.2", "pandas": "2.2.2", "numpy": "1.26.4",
        },
        "instructions": (
            "1. scp this entire deploy_package/ folder to the CM5. "
            "2. On the CM5: pip install xgboost torch scikit-learn joblib pandas numpy "
            "(matching versions above where possible). "
            "3. Copy cm5_benchmark.py alongside this folder. "
            "4. Run: python cm5_benchmark.py"
        ),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Package ready at {OUT_DIR}/")
    print("Contents:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-suffix", type=str, default="")
    parser.add_argument("--xgb-tag", type=str, required=True,
                         help="e.g. 'dedup' -- no leading underscore")
    parser.add_argument("--mlp-tag", type=str, required=True,
                         help="e.g. 'dedup2' -- no leading underscore")
    parser.add_argument("--calib-tag", type=str, required=True)
    args = parser.parse_args()
    main(split_suffix=args.split_suffix, xgb_tag=args.xgb_tag,
         mlp_tag=args.mlp_tag, calib_tag=args.calib_tag)