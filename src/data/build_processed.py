"""
Build aligned, canonical-feature processed CSVs from raw Bot-IoT and
N-BaIoT files.

Run order:
    1. scripts/inspect_datasets.py     (already done -- confirms schemas)
    2. src/data/canonical_features.py  (optional -- prints coverage report)
    3. src/data/build_processed.py     (this file -- produces the CSVs
                                         used by everything downstream)

Usage:
    python src/data/build_processed.py
"""

import pandas as pd
from pathlib import Path

from canonical_features import align_bot_iot, align_nbaiot, coverage_report

BOT_IOT_RAW = Path("data/raw/bot_iot")
NBAIOT_RAW = Path("data/raw/n_baiot")
PROCESSED_DIR = Path("data/processed")

NBAIOT_SKIP_FILES = {"features.csv", "device_info.csv", "data_summary.csv", "README.md"}


def build_bot_iot() -> None:
    files = sorted(BOT_IOT_RAW.glob("reduced_data_*.csv"))
    if not files:
        print(f"[!] No reduced_data_*.csv found in {BOT_IOT_RAW}")
        return

    frames = []
    for f in files:
        print(f"  Processing {f.name} ...")
        df = pd.read_csv(f)
        frames.append(align_bot_iot(df))

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "bot_iot_aligned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"  -> Wrote {len(combined):,} rows to {out_path}\n")


def build_nbaiot() -> None:
    files = sorted(f for f in NBAIOT_RAW.glob("*.csv") if f.name not in NBAIOT_SKIP_FILES)
    if not files:
        print(f"[!] No device CSVs found in {NBAIOT_RAW}")
        return

    frames = []
    for f in files:
        print(f"  Processing {f.name} ...")
        df = pd.read_csv(f)
        frames.append(align_nbaiot(df, f))

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "n_baiot_aligned.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"  -> Wrote {len(combined):,} rows to {out_path}\n")


if __name__ == "__main__":
    print("=" * 70)
    print("COVERAGE DISCLOSURE (BRIDGE-style)")
    print("=" * 70)
    coverage_report()

    print("\n" + "=" * 70)
    print("BUILDING BOT-IOT (known classes)")
    print("=" * 70)
    build_bot_iot()

    print("=" * 70)
    print("BUILDING N-BAIOT (zero-day test)")
    print("=" * 70)
    build_nbaiot()

    print("Done. Processed files are in data/processed/")