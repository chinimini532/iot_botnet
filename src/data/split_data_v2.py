"""
Train/val/test split for Bot-IoT, plus calibration-strengthening using a
held-out slice of N-BaIoT's benign traffic.

*** UPDATED: deduplication added before splitting ***
A post-hoc check found 97.02% of a naive stratified test split was exact
feature-vector duplicates of training rows (98.65% for DDoS, 97.88% for
DoS) -- Bot-IoT's flood-style attacks produce many near-identical rows by
nature of the attack, and a random split scatters duplicates across train
and test, inflating Tier-1 scores to near-perfect without genuine
generalization. Fixed by deduplicating on (9 canonical features +
category_grouped) BEFORE splitting, for both Bot-IoT and N-BaIoT. State
this explicitly in the paper's methodology -- it's a real, disclosed
correction, not swept under the rug.

Design decisions (documented here -- carry into paper methodology section):

1. Deduplication: rows with identical (canonical features + label) are
   collapsed to one representative row before any split. This guarantees
   zero train/test overlap by construction, rather than hoping a random
   split avoids it.

2. Theft folded into Reconnaissance for MULTI-CLASS reporting only (see
   earlier notes -- unaffected by this update).

3. N-BaIoT benign borrowed for calibration (see earlier notes) --
   deduplication applied here too, for the same reason.

Outputs (all in data/processed/splits/):
    bot_iot_train.csv
    bot_iot_val_calibration.csv
    bot_iot_test.csv
    n_baiot_zeroday_test.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = PROCESSED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_FRAC = 0.70


def fold_theft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category_grouped"] = df["category"].replace({"Theft": "Reconnaissance"})
    return df


def deduplicate(df: pd.DataFrame, dedup_cols: list, label: str) -> pd.DataFrame:
    before = len(df)
    df_dedup = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    after = len(df_dedup)
    print(f"  [{label}] deduplication: {before:,} -> {after:,} rows "
          f"({after/before*100:.2f}% retained, "
          f"{(before-after):,} exact duplicates removed)")
    return df_dedup


def split_bot_iot(suffix: str = ""):
    print("Loading Bot-IoT aligned data ...")
    df = pd.read_csv(PROCESSED_DIR / "bot_iot_aligned.csv")
    df = fold_theft(df)

    df = deduplicate(df, CANONICAL_FEATURES + ["category_grouped"], "Bot-IoT")

    print("  Post-dedup class distribution:")
    print(df["category_grouped"].value_counts().to_string())

    print("Splitting Bot-IoT (stratified on category_grouped) ...")
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df["category_grouped"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["category_grouped"],
        random_state=RANDOM_STATE,
    )

    train_path = SPLITS_DIR / f"bot_iot_train{suffix}.csv"
    test_path = SPLITS_DIR / f"bot_iot_test{suffix}.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  train: {len(train_df):,} rows -> {train_path}")
    print(f"  val:   {len(val_df):,} rows  (before N-BaIoT benign supplement)")
    print(f"  test:  {len(test_df):,} rows -> {test_path}")

    return val_df


def split_nbaiot_and_supplement(bot_iot_val_df: pd.DataFrame, suffix: str = ""):
    print("\nLoading N-BaIoT aligned data ...")
    df = pd.read_csv(PROCESSED_DIR / "n_baiot_aligned.csv")

    benign_mask = df["category"] == "benign"
    benign_df = df[benign_mask].copy()
    attack_df = df[~benign_mask].copy()

    print(f"  N-BaIoT benign total (pre-dedup): {len(benign_df):,}")
    print(f"  N-BaIoT attack total (pre-dedup): {len(attack_df):,}")

    benign_df = deduplicate(benign_df, CANONICAL_FEATURES, "N-BaIoT benign")
    attack_df = deduplicate(attack_df, CANONICAL_FEATURES + ["category", "subcategory"],
                             "N-BaIoT attack")

    calib_support_df, zeroday_benign_df = train_test_split(
        benign_df, test_size=0.5, random_state=RANDOM_STATE,
    )
    calib_support_df["category_grouped"] = "Normal"
    val_calibration_df = pd.concat([bot_iot_val_df, calib_support_df], ignore_index=True)
    calib_path = SPLITS_DIR / f"bot_iot_val_calibration{suffix}.csv"
    val_calibration_df.to_csv(calib_path, index=False)

    zeroday_test_df = pd.concat([attack_df, zeroday_benign_df], ignore_index=True)
    zeroday_path = SPLITS_DIR / f"n_baiot_zeroday_test{suffix}.csv"
    zeroday_test_df.to_csv(zeroday_path, index=False)

    print(f"\n  calibration-support from N-BaIoT benign: {len(calib_support_df):,}")
    print(f"  -> {calib_path} total: {len(val_calibration_df):,} rows")
    print(f"  -> {zeroday_path}: {len(zeroday_test_df):,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix", type=str, default="_dedup",
        help="Suffix appended to output split filenames, so the new "
             "(deduplicated) splits don't overwrite the original ones. "
             "e.g. --suffix _dedup produces bot_iot_train_dedup.csv etc. "
             "Pass an empty string to overwrite the originals instead.",
    )
    args = parser.parse_args()
    suffix = args.suffix

    bot_iot_val_df = split_bot_iot(suffix=suffix)
    split_nbaiot_and_supplement(bot_iot_val_df, suffix=suffix)

    print("\n" + "=" * 60)
    print("VERIFYING: zero train/test duplication after fix")
    print("=" * 60)
    train_check = pd.read_csv(SPLITS_DIR / f"bot_iot_train{suffix}.csv")
    test_check = pd.read_csv(SPLITS_DIR / f"bot_iot_test{suffix}.csv")
    train_keys = set(map(tuple, train_check[CANONICAL_FEATURES].round(6).values))
    test_keys = test_check[CANONICAL_FEATURES].round(6).values
    dup_count = sum(1 for row in test_keys if tuple(row) in train_keys)
    print(f"Test rows still duplicated in train: {dup_count} / {len(test_check):,} "
          f"({dup_count/len(test_check)*100:.4f}%)")

    print(f"\nDone. Splits written to data/processed/splits/ with suffix '{suffix}'")