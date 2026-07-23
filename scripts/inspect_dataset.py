"""
Inspect raw dataset schemas before writing any preprocessing/alignment code.

Run this locally (LG Gram) first. It only reads small slices of each file,
so it's safe to run even on the full 8GB N-BaIoT folder.

Usage:
    python scripts/inspect_datasets.py
"""

import pandas as pd
from pathlib import Path

BOT_IOT_DIR = Path("data/raw/bot_iot")
NBAIOT_DIR = Path("data/raw/n_baiot")


def inspect_bot_iot():
    print("=" * 70)
    print("BOT-IOT")
    print("=" * 70)

    files = sorted(BOT_IOT_DIR.glob("reduced_data_*.csv"))
    if not files:
        print(f"No reduced_data_*.csv files found in {BOT_IOT_DIR}")
        return

    # Just read the first file's header + a few rows
    df = pd.read_csv(files[0], nrows=5)
    print(f"\nFile: {files[0].name}")
    print(f"Shape (first 5 rows read): {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}  (dtype: {df[col].dtype})")

    print("\nSample rows:")
    print(df.head(3).to_string())

    # Check for label-like columns across all 4 files
    print("\n--- Checking label columns across all files ---")
    for f in files:
        df_head = pd.read_csv(f, nrows=1)
        label_cols = [c for c in df_head.columns if c.lower() in
                      ("attack", "category", "subcategory", "label")]
        print(f"{f.name}: label-like columns = {label_cols}")


def inspect_nbaiot():
    print("\n" + "=" * 70)
    print("N-BAIOT")
    print("=" * 70)

    # Look at metadata files first
    for meta_file in ["features.csv", "device_info.csv", "data_summary.csv"]:
        path = NBAIOT_DIR / meta_file
        if path.exists():
            print(f"\n--- {meta_file} ---")
            df = pd.read_csv(path)
            print(df.head(10).to_string())

    # Inspect one benign and one attack file to see feature schema
    sample_files = {
        "benign": NBAIOT_DIR / "1.benign.csv",
        "mirai_ack": NBAIOT_DIR / "1.mirai.ack.csv",
        "gafgyt_combo": NBAIOT_DIR / "1.gafgyt.combo.csv",
    }

    for label, path in sample_files.items():
        if not path.exists():
            print(f"\n[skip] {path} not found")
            continue
        df = pd.read_csv(path, nrows=5)
        print(f"\n--- {label} ({path.name}) ---")
        print(f"Shape (first 5 rows): {df.shape}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"First 10 columns: {list(df.columns[:10])}")
        print(f"Last 5 columns: {list(df.columns[-5:])}")

    # Row counts per device (quick line count, not full load)
    print("\n--- Row counts per file (this may take a moment on 8GB total) ---")
    for f in sorted(NBAIOT_DIR.glob("*.csv")):
        if f.name in ("features.csv", "device_info.csv", "data_summary.csv"):
            continue
        with open(f, "r") as fh:
            n_lines = sum(1 for _ in fh) - 1  # minus header
        print(f"{f.name}: {n_lines} rows")


if __name__ == "__main__":
    inspect_bot_iot()
    inspect_nbaiot()