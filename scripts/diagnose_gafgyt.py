"""
Diagnostic: why does gafgyt catch so poorly (~3%) compared to mirai
(~14-20%)? Compare gafgyt's and mirai's feature distributions against
each known Bot-IoT class -- if gafgyt's 9-feature profile closely
resembles a known class (most likely DDoS/DoS, given flood-style
traffic), that would explain why the gate treats it as "known" rather
than flagging it as unusual.

Usage:
    python scripts/diagnose_gafgyt.py
"""

import pandas as pd
from pathlib import Path

CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]

SPLITS_DIR = Path("data/processed/splits")


def main():
    print("Loading Bot-IoT known-class training data ...")
    bot_iot = pd.read_csv(SPLITS_DIR / "bot_iot_train_dedup.csv")
    known_means = bot_iot.groupby("category_grouped")[CANONICAL_FEATURES].mean()

    print("Loading N-BaIoT zero-day test data ...")
    nbaiot = pd.read_csv(SPLITS_DIR / "n_baiot_zeroday_test_dedup.csv")
    zeroday_means = nbaiot[nbaiot["attack"] == 1].groupby("category")[CANONICAL_FEATURES].mean()

    print("\n" + "=" * 80)
    print("KNOWN CLASS MEANS (Bot-IoT)")
    print("=" * 80)
    print(known_means.to_string())

    print("\n" + "=" * 80)
    print("ZERO-DAY FAMILY MEANS (N-BaIoT)")
    print("=" * 80)
    print(zeroday_means.to_string())

    # Normalized distance from each zero-day family to each known class,
    # using known-class std as the scale (so we see "how many known-class
    # standard deviations away" each family is -- closer to 0 = looks more
    # like that known class, easier to see why the gate might not flag it)
    known_stds = bot_iot.groupby("category_grouped")[CANONICAL_FEATURES].std()

    print("\n" + "=" * 80)
    print("NORMALIZED DISTANCE: zero-day family vs. each known class")
    print("(lower = zero-day family looks MORE like that known class)")
    print("=" * 80)
    for fam in zeroday_means.index:
        print(f"\n--- {fam} ---")
        for known_cls in known_means.index:
            diff = (zeroday_means.loc[fam] - known_means.loc[known_cls]).abs()
            scale = known_stds.loc[known_cls].replace(0, 1)
            normalized_dist = (diff / scale).mean()
            print(f"  vs {known_cls:<16} avg normalized distance = {normalized_dist:.3f}")


if __name__ == "__main__":
    main()