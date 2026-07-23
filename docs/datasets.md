# Dataset Notes

## Bot-IoT (training — known botnet families)
- Source: Kaggle mirror of UNSW's 5% reduced subset — `reduced_data_1-4.csv`
- 3,668,522 aligned rows. Severe class imbalance (matches published
  literature exactly): DDoS 52.52%, DoS 44.98%, Reconnaissance 2.48%,
  **Normal 0.01% (477 rows)**, **Theft 0.00% (79 rows)**
- Citation: Koroniotis, N., Moustafa, N., Sitnikova, E., & Turnbull, B.
  (2019). Future Generation Computer Systems, 100, 779-796.

## N-BaIoT (zero-day test — unseen families)
- Source: Kaggle mirror / UCI ML Repository (id=442)
- 7,062,606 aligned rows. Distribution: mirai 51.94%, gafgyt 40.19%,
  benign 7.87% (555,932 rows — no scarcity problem)
- Citation: Meidan, Y. et al. (2018). UCI ML Repository. DOI: 10.24432/C5RC8J

## Feature Alignment
See canonical vocabulary and mapping table in `src/data/canonical_features.py`
(methodology follows Bhilwarawala et al. 2026, arXiv:2604.11324 — see that
file's docstring for full justification). Coverage: Bot-IoT 8/9 (89%),
N-BaIoT 9/9 (100%).

## Class Imbalance Handling (see `src/data/split_data.py`)

Bot-IoT's Normal (477) and Theft (79) classes are too small for reliable
standalone reporting after a train/val/test split (~71 and ~12 test rows
respectively). Two decisions address this — **both must be stated
explicitly in the paper's methodology section**:

1. **Theft folded into Reconnaissance** for multi-class reporting only
   (`category_grouped` column). Both are low-volume, non-flooding attack
   types, semantically distinct from DDoS/DoS flooding — a defensible
   grouping, not arbitrary. The binary `attack` label and original
   `category` column are unaffected; Theft rows still count as attacks.

2. **N-BaIoT's benign traffic used to strengthen calibration only.**
   N-BaIoT benign (555,932 rows) is split 50/50: one half supplements
   Bot-IoT's tiny Normal-class validation slice for MC Dropout threshold
   calibration (`bot_iot_val_calibration.csv`); the other half stays in
   the zero-day test set untouched. **All N-BaIoT attack rows (mirai,
   gafgyt — 6,506,674 rows) remain 100% unseen during training and
   calibration** — this only strengthens the benign/false-positive side
   of calibration and does not affect the zero-day attack-detection claim.

## Split Files (in `data/processed/splits/`)

| File | Contents | Purpose |
|---|---|---|
| `bot_iot_train.csv` | ~70% of Bot-IoT, stratified | Classifier + MLP gate training |
| `bot_iot_val_calibration.csv` | ~15% of Bot-IoT + 50% of N-BaIoT benign | MC Dropout threshold calibration |
| `bot_iot_test.csv` | ~15% of Bot-IoT, stratified | In-distribution (known-class) accuracy |
| `n_baiot_zeroday_test.csv` | All N-BaIoT attacks + remaining 50% N-BaIoT benign | Zero-day evaluation |

## Run Order

1. `scripts/inspect_datasets.py` — confirms raw schemas (done)
2. `src/data/canonical_features.py` — coverage disclosure (done)
3. `src/data/build_processed.py` — builds aligned CSVs (done)
4. `src/data/split_data.py` — builds train/val-calibration/test/zero-day splits