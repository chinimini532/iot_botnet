"""
Canonical feature vocabulary for cross-dataset alignment between Bot-IoT
and N-BaIoT.

METHODOLOGY NOTE (cite in paper's methodology section):
This module follows the disciplined cross-dataset feature alignment
approach established by Bhilwarawala et al. (2026), "BRIDGE and TCH-Net:
Heterogeneous Benchmark and Multi-Branch Baseline for Cross-Domain IoT
Botnet Detection" (arXiv:2604.11324):
    1. Genuine-equivalence-only mapping -- never fabricate a proxy feature
       just to fill a slot.
    2. Explicit zero-fill for features one dataset cannot provide, with
       full coverage disclosed (not hidden).
    3. Rejected mappings are documented explicitly, with the reasoning,
       not just silently dropped.

We do NOT use BRIDGE's own 46-feature CICFlowMeter-style vocabulary,
because it is built around forward/backward DIRECTIONAL flow features
(pkt_count_fwd, bwd_iat_mean, etc.) that assume a CICFlowMeter-style
bidirectional flow model. N-BaIoT's Kitsune-style features have no
forward/backward concept at all -- BRIDGE's own shipped code confirms
this, reaching only ~15% coverage for N-BaIoT and dropping it from later
versions of their pipeline.

Instead, this vocabulary is grounded in what BOTH datasets genuinely
express well: multi-window traffic statistics computed per source.
  - Bot-IoT: per-source/protocol windowed aggregates (TnBPSrcIP,
    TnP_PSrcIP, AR_P_Proto_P_SrcIP, N_IN_Conn_P_SrcIP) alongside
    immediate per-flow stats (rate, bytes, pkts).
  - N-BaIoT: decayed time-window statistics (L5 = fast/recent,
    L1 = slower/longer window) at the MI_dir (source MAC-IP) and
    H (source IP) aggregation scopes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from label_utils import parse_nbaiot_filename


# ── Canonical vocabulary (9 features) ───────────────────────────────────
CANONICAL_FEATURES = [
    "traffic_rate_fast",   # 0  immediate / most-recent-window traffic rate
    "traffic_rate_slow",   # 1  longer-window traffic rate
    "avg_pkt_size_fast",   # 2  average packet size, fast window
    "avg_pkt_size_slow",   # 3  average packet size, slow window
    "pkt_size_var_fast",   # 4  packet size variability, fast window
    "pkt_size_var_slow",   # 5  packet size variability, slow window
    "src_volume_bytes",    # 6  total byte volume attributable to the source
    "src_pkt_count",       # 7  total packet count attributable to the source
    "src_conn_count",      # 8  connection/host-pair intensity for the source
]
N_CANON = len(CANONICAL_FEATURES)


def align_bot_iot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Bot-IoT's raw columns onto the canonical vocabulary.

    Accepted mappings (genuine equivalence):
      traffic_rate_fast <- rate                     (direct: pkts/sec)
      traffic_rate_slow <- AR_P_Proto_P_SrcIP        (direct: windowed avg rate per proto/srcIP)
      avg_pkt_size_fast <- bytes / pkts               (derived: per-flow avg size)
      avg_pkt_size_slow <- TnBPSrcIP / TnP_PSrcIP     (derived: windowed avg size per srcIP)
      src_volume_bytes  <- TnBPSrcIP                  (direct: total bytes per srcIP window)
      src_pkt_count     <- TnP_PSrcIP                 (direct: total pkts per srcIP window)
      src_conn_count    <- N_IN_Conn_P_SrcIP           (direct: connection count per srcIP)

    Approximated (documented caveat -- state explicitly in paper limitations):
      pkt_size_var_fast <- abs(sbytes - dbytes) / pkts
          Bot-IoT does not expose a true per-flow packet-size variance;
          this is a directional-asymmetry proxy, not an equivalent quantity.

    Rejected / zero-filled (genuine schema gap, not an oversight):
      pkt_size_var_slow -- Bot-IoT has no windowed packet-size variance
          equivalent at the per-source-IP aggregation level. Zero-filled.
    """
    out = pd.DataFrame(0.0, index=df.index, columns=CANONICAL_FEATURES)

    out["traffic_rate_fast"] = df["rate"]
    out["traffic_rate_slow"] = df["AR_P_Proto_P_SrcIP"]

    pkts_safe = df["pkts"].replace(0, np.nan)
    out["avg_pkt_size_fast"] = df["bytes"] / pkts_safe

    tnp_safe = df["TnP_PSrcIP"].replace(0, np.nan)
    out["avg_pkt_size_slow"] = df["TnBPSrcIP"] / tnp_safe

    # Approximation -- see docstring caveat above.
    out["pkt_size_var_fast"] = (df["sbytes"] - df["dbytes"]).abs() / pkts_safe
    # pkt_size_var_slow: no equivalent in Bot-IoT -> left as 0.0 (zero-filled)

    out["src_volume_bytes"] = df["TnBPSrcIP"]
    out["src_pkt_count"] = df["TnP_PSrcIP"]
    out["src_conn_count"] = df["N_IN_Conn_P_SrcIP"]

    out["attack"] = df["attack"]
    out["category"] = df["category"]
    out["subcategory"] = df["subcategory"]
    out["dataset_source"] = "bot_iot"

    out = out.dropna(subset=[
        "traffic_rate_fast", "avg_pkt_size_fast", "pkt_size_var_fast"
    ])
    return out


def align_nbaiot(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    Map N-BaIoT's raw columns onto the canonical vocabulary.

    Accepted mappings (genuine equivalence):
      traffic_rate_fast <- MI_dir_L5_weight    (direct: fast-window pkt-count weight)
      traffic_rate_slow <- MI_dir_L1_weight    (direct: slower-window pkt-count weight)
      avg_pkt_size_fast <- MI_dir_L5_mean      (direct: fast-window packet-size mean)
      avg_pkt_size_slow <- MI_dir_L1_mean      (direct: slow-window packet-size mean)
      pkt_size_var_fast <- MI_dir_L5_variance  (direct: fast-window packet-size variance)
      pkt_size_var_slow <- MI_dir_L1_variance  (direct: slow-window packet-size variance)
      src_pkt_count     <- H_L5_weight          (direct: source-IP-level packet-count weight)
      src_conn_count    <- HH_L5_weight         (direct: source-dest host-pair weight,
                                                  closest available proxy for connection
                                                  intensity -- N-BaIoT has no raw
                                                  connection-count field)

    Derived (documented, not fabricated -- weight * mean approximates volume,
    the same relationship used internally by the Kitsune feature extractor):
      src_volume_bytes  <- H_L5_weight * H_L5_mean
    """
    out = pd.DataFrame(0.0, index=df.index, columns=CANONICAL_FEATURES)

    out["traffic_rate_fast"] = df["MI_dir_L5_weight"]
    out["traffic_rate_slow"] = df["MI_dir_L1_weight"]
    out["avg_pkt_size_fast"] = df["MI_dir_L5_mean"]
    out["avg_pkt_size_slow"] = df["MI_dir_L1_mean"]
    out["pkt_size_var_fast"] = df["MI_dir_L5_variance"]
    out["pkt_size_var_slow"] = df["MI_dir_L1_variance"]

    out["src_volume_bytes"] = df["H_L5_weight"] * df["H_L5_mean"]
    out["src_pkt_count"] = df["H_L5_weight"]
    out["src_conn_count"] = df["HH_L5_weight"]

    meta = parse_nbaiot_filename(file_path)
    out["attack"] = int(meta["is_attack"])
    out["category"] = meta["family"]
    out["subcategory"] = meta["subtype"]
    out["device_id"] = meta["device_id"]
    out["dataset_source"] = "n_baiot"

    out = out.dropna(subset=[
        "traffic_rate_fast", "avg_pkt_size_fast", "pkt_size_var_fast"
    ])
    return out


def coverage_report() -> pd.DataFrame:
    """
    Explicit coverage disclosure per BRIDGE's methodology -- report which
    canonical features are genuinely populated (non-zero-fill) for each
    dataset, so the paper can state coverage honestly instead of implying
    a full 9/9 match.
    """
    bot_iot_zero_filled = {"pkt_size_var_slow"}
    n_baiot_zero_filled = set()  # all 9 slots genuinely populated for N-BaIoT

    rows = []
    for feat in CANONICAL_FEATURES:
        rows.append({
            "feature": feat,
            "bot_iot": "zero-filled" if feat in bot_iot_zero_filled else "populated",
            "n_baiot": "zero-filled" if feat in n_baiot_zero_filled else "populated",
        })
    report = pd.DataFrame(rows)

    bot_iot_coverage = (N_CANON - len(bot_iot_zero_filled)) / N_CANON
    n_baiot_coverage = (N_CANON - len(n_baiot_zero_filled)) / N_CANON
    print(f"Bot-IoT coverage:  {bot_iot_coverage*100:.0f}% "
          f"({N_CANON - len(bot_iot_zero_filled)}/{N_CANON})")
    print(f"N-BaIoT coverage:  {n_baiot_coverage*100:.0f}% "
          f"({N_CANON - len(n_baiot_zero_filled)}/{N_CANON})")
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    coverage_report()