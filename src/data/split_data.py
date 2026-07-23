"""
Train/val/test split for Bot-IoT, plus calibration-strengthening using a
held-out slice of N-BaIoT's benign traffic.

Design decisions (documented here -- carry into paper methodology section):

1. Theft folded into Reconnaissance for MULTI-CLASS reporting only.
   Bot-IoT's Theft category has only 79 total rows -- any precision/recall
   computed on a ~12-row test slice is statistically unreliable regardless
   of the number it produces. Theft (keylogging, data exfiltration) and
   Reconnaissance (OS/service scanning) are both low-volume, non-flooding
   attack types, semantically distinct from the high-volume DDoS/DoS
   flooding categories -- so merging them is a defensible grouping, not an
   arbitrary one. The original 'category' column is preserved unchanged;
   'category_grouped' is added alongside it. Binary attack/benign framing
   (the 'attack' column) is completely unaffected by this -- Theft rows
   are still counted as attacks, just not reported as their own class.

2. Bot-IoT's Normal (benign) class has only 477 rows total. Rather than
   calibrate the MC Dropout gate's false-positive rate on a ~71-row
   validation slice, we supplement it with a held-out 50% slice of
   N-BaIoT's benign traffic (555,932 rows total, no scarcity problem).
   IMPORTANT: only N-BaIoT's BENIGN rows are used this way. All of
   N-BaIoT's ATTACK rows (mirai, gafgyt) remain 100% unseen during
   training and calibration -- the zero-day evaluation is entirely
   unaffected, since it's built around unseen ATTACK families, not
   benign traffic. State this precisely in the paper's methodology.

Outputs (all in data/processed/splits/):
    bot_iot_train.csv                    -- classifier + MLP training
    bot_iot_val_calibration.csv          -- Bot-IoT val slice + N-BaIoT
                                             benign calibration-support slice
    bot_iot_test.csv                     -- in-distribution test (known-class accuracy)
    n_baiot_zeroday_test.csv             -- N-BaIoT attacks (all) + remaining
                                             50% of N-BaIoT benign (untouched
                                             zero-day evaluation set)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = PROCESSED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15   # of the remaining 30%, split evenly -> 15% val, 15% test
TEST_FRAC = 0.15


def fold_theft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category_grouped"] = df["category"].replace({"Theft": "Reconnaissance"})
    return df


def split_bot_iot():
    print("Loading Bot-IoT aligned data ...")
    df = pd.read_csv(PROCESSED_DIR / "bot_iot_aligned.csv")
    df = fold_theft(df)

    print("Splitting Bot-IoT (stratified on category_grouped) ...")
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df["category_grouped"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["category_grouped"],
        random_state=RANDOM_STATE,
    )

    train_df.to_csv(SPLITS_DIR / "bot_iot_train.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "bot_iot_test.csv", index=False)

    print(f"  train: {len(train_df):,} rows")
    print(f"  val:   {len(val_df):,} rows  (before N-BaIoT benign supplement)")
    print(f"  test:  {len(test_df):,} rows")

    return val_df


def split_nbaiot_and_supplement(bot_iot_val_df: pd.DataFrame):
    print("\nLoading N-BaIoT aligned data ...")
    df = pd.read_csv(PROCESSED_DIR / "n_baiot_aligned.csv")

    benign_mask = df["category"] == "benign"
    benign_df = df[benign_mask]
    attack_df = df[~benign_mask]

    print(f"  N-BaIoT benign total: {len(benign_df):,}")
    print(f"  N-BaIoT attack total (mirai + gafgyt, fully held out): {len(attack_df):,}")

    calib_support_df, zeroday_benign_df = train_test_split(
        benign_df, test_size=0.5, random_state=RANDOM_STATE,
    )

    # Calibration validation set = Bot-IoT val slice + N-BaIoT benign supplement
    val_calibration_df = pd.concat([bot_iot_val_df, calib_support_df], ignore_index=True)
    val_calibration_df.to_csv(SPLITS_DIR / "bot_iot_val_calibration.csv", index=False)

    # Zero-day test set = ALL N-BaIoT attacks (untouched) + remaining benign half
    zeroday_test_df = pd.concat([attack_df, zeroday_benign_df], ignore_index=True)
    zeroday_test_df.to_csv(SPLITS_DIR / "n_baiot_zeroday_test.csv", index=False)

    print(f"\n  calibration-support from N-BaIoT benign: {len(calib_support_df):,}")
    print(f"  -> bot_iot_val_calibration.csv total: {len(val_calibration_df):,} rows")
    print(f"  zero-day test (attacks + remaining benign): {len(zeroday_test_df):,} rows")


if __name__ == "__main__":
    bot_iot_val_df = split_bot_iot()
    split_nbaiot_and_supplement(bot_iot_val_df)
    print("\nDone. Splits written to data/processed/splits/")