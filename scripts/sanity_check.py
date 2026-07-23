"""
Sanity check on the processed/aligned datasets, before any model gets
built on top of them.

Checks:
    1. Class distribution (how imbalanced is each dataset?)
    2. Per-class mean/std of the 9 canonical features -- do benign and
       attack traffic actually look different?
    3. NaN / inf / degenerate (near-constant) feature check

Run after src/data/build_processed.py.

Usage:
    python scripts/sanity_check.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]

BOT_IOT_PATH = Path("data/processed/bot_iot_aligned.csv")
NBAIOT_PATH = Path("data/processed/n_baiot_aligned.csv")


def check_dataset(path: Path, name: str, class_col: str):
    print("=" * 70)
    print(f"{name}  ({path})")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"Total rows: {len(df):,}")

    print(f"\n--- Class distribution ({class_col}) ---")
    counts = df[class_col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    dist = pd.DataFrame({"count": counts, "pct": pct})
    print(dist.to_string())

    print(f"\n--- Per-class feature means (grouped by {class_col}) ---")
    means = df.groupby(class_col)[CANONICAL_FEATURES].mean()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(means.to_string())

    print("\n--- NaN / inf check ---")
    nan_counts = df[CANONICAL_FEATURES].isna().sum()
    inf_counts = df[CANONICAL_FEATURES].apply(lambda c: np.isinf(c).sum())
    problems = pd.DataFrame({"nan_count": nan_counts, "inf_count": inf_counts})
    problems = problems[(problems["nan_count"] > 0) | (problems["inf_count"] > 0)]
    if problems.empty:
        print("None found.")
    else:
        print(problems.to_string())

    print("\n--- Near-constant feature check (std == 0 within a class) ---")
    stds = df.groupby(class_col)[CANONICAL_FEATURES].std()
    degenerate = (stds == 0).any(axis=0)
    flagged = degenerate[degenerate].index.tolist()
    if flagged:
        print(f"WARNING -- these features are constant within at least one class: {flagged}")
    else:
        print("None found.")

    print()


if __name__ == "__main__":
    if BOT_IOT_PATH.exists():
        check_dataset(BOT_IOT_PATH, "BOT-IOT", class_col="category")
    else:
        print(f"[!] {BOT_IOT_PATH} not found -- run build_processed.py first.")

    if NBAIOT_PATH.exists():
        check_dataset(NBAIOT_PATH, "N-BAIOT", class_col="category")
    else:
        print(f"[!] {NBAIOT_PATH} not found -- run build_processed.py first.")