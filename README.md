# Uncertainty-Aware Zero-Day Botnet Detection for IoT Networks

### Real-World Edge Deployment on Resource-Constrained Hardware (Raspberry Pi CM5)

> Target venue: **MDPI Sensors** (Special Issue — Guest Editor: Prof. Prashant Kumar)

---

## Overview

This project builds a botnet traffic classifier for IoT networks that can
recognize **known** attack families *and* flag **unseen (zero-day)** attack
families as uncertain, instead of confidently mislabeling them — using
MC Dropout uncertainty gating. The final model is deployed and profiled on
real Raspberry Pi CM5 hardware to measure practical edge-deployment cost.

| Goal | Approach |
|---|---|
| **Detect known botnet families** | Supervised classifier trained on Bot-IoT |
| **Flag zero-day attacks** | MC Dropout uncertainty gating, tested on held-out N-BaIoT families |
| **Prove edge feasibility** | Quantized deployment + real latency/memory profiling on CM5 |

---

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| **Bot-IoT** | Training (known families) | UNSW Canberra, realistic IoT testbed traffic |
| **N-BaIoT** | Zero-day test (unseen families) | Real infected commercial IoT device traffic — different collection methodology than Bot-IoT, used to test genuine cross-dataset generalization |

Raw files are **not** committed to this repo (see `.gitignore`) — download
instructions are in [`docs/DATASETS.md`](docs/DATASETS.md).

---

## Project Structure

```
iot-zeroday-botnet/
├── data/
│   ├── raw/                  # Downloaded datasets (gitignored)
│   └── processed/            # Cleaned / feature-aligned data (gitignored)
├── notebooks/                # Kaggle/exploratory notebooks
├── src/
│   ├── data/                 # Loading, cleaning, feature alignment
│   ├── models/                # Classifier + MC Dropout model definitions
│   ├── training/               # Training loops, config-driven runs
│   ├── evaluation/            # Zero-day evaluation, uncertainty metrics
│   └── deployment/            # ONNX export, CM5 inference + profiling
├── configs/                  # YAML configs (hyperparameters, paths)
├── models/checkpoints/        # Saved model weights (gitignored)
├── results/
│   ├── figures/               # Plots for the paper
│   ├── logs/                  # Training/eval logs
│   └── metrics/               # Saved metrics (JSON/CSV)
├── scripts/                  # Entry-point scripts (train.py, evaluate.py, etc.)
├── docs/                     # Dataset notes, deployment notes
├── requirements.txt
└── README.md
```

---

## Workflow

```
   Local (RTX 3050, 4GB)                 Kaggle (Free GPU)
 ┌──────────────────────┐        push   ┌──────────────────────┐
 │ Write & debug code    │ ────────────▶ │ Clone repo            │
 │ Small-sample testing  │   GitHub      │ Full training runs    │
 └──────────────────────┘               │ Save checkpoints      │
                                          └───────────┬──────────┘
                                                        │ pull results
                                                        ▼
                                          ┌──────────────────────┐
                                          │ CM5 (Raspberry Pi)    │
                                          │ ONNX inference        │
                                          │ Latency/memory profile│
                                          └──────────────────────┘
```

1. Develop and unit-test code locally (small data samples only)
2. Push to GitHub
3. Clone repo inside a Kaggle notebook, run full training on Kaggle GPU
4. Pull trained model back down, export to ONNX
5. Deploy + profile on CM5 hardware

---

## Setup

```bash
git clone <this-repo-url>
cd iot-zeroday-botnet
pip install -r requirements.txt
```

On Kaggle, add this repo as a notebook data source or clone directly in a
notebook cell:

```python
!git clone <this-repo-url>
%cd iot-zeroday-botnet
!pip install -r requirements.txt
```

---

## Status

- [x] Project scoped and approved by supervisor
- [x] Datasets selected (Bot-IoT + N-BaIoT)
- [ ] Data preprocessing + feature alignment
- [ ] Baseline classifier (no uncertainty gating)
- [ ] MC Dropout uncertainty-gated model
- [ ] Zero-day evaluation on N-BaIoT holdout
- [ ] ONNX export + quantization
- [ ] CM5 deployment + latency/memory profiling
- [ ] Paper writing

---

## Authors

- Cinmoy Purkaystha — Woosong University
- Yesha Nilesh Gandhi — Woosong University
- Prof. Prashant Kumar (Supervisor) — Woosong University