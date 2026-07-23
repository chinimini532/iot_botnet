"""
Tier 1 of the two-tier architecture: the primary known-class classifier.

Target: `category_grouped` (DDoS, DoS, Reconnaissance [includes folded
Theft], Normal) -- see src/data/split_data.py for the Theft-folding
rationale.

XGBoost is the primary model (tree-based classifiers consistently
dominate tabular IoT traffic benchmarks -- see project notes). Random
Forest is included as a secondary baseline for comparison, not as a
competing primary choice.

Class weighting (not oversampling) is used to handle Bot-IoT's severe
imbalance (Normal 0.01%, Theft 0.00% pre-folding) -- oversampling was
deliberately avoided given the earlier decision that synthetic minority
samples could distort MC Dropout gate calibration downstream.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

CANONICAL_FEATURES = [
    "traffic_rate_fast", "traffic_rate_slow",
    "avg_pkt_size_fast", "avg_pkt_size_slow",
    "pkt_size_var_fast", "pkt_size_var_slow",
    "src_volume_bytes", "src_pkt_count", "src_conn_count",
]
TARGET_COL = "category_grouped"

MODELS_DIR = Path("models/checkpoints")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_split(path: Path, sample_per_class: int = None):
    """
    Load a split CSV. If sample_per_class is given, stratified-subsample
    up to that many rows per category_grouped class (fewer if a class has
    less than that available) -- for a fast local smoke test before the
    full Kaggle run. Leave as None for the real, full-scale run.
    """
    df = pd.read_csv(path)
    if sample_per_class is not None:
        df = (
            df.groupby(TARGET_COL, group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), sample_per_class), random_state=42))
        )
        print(f"  [smoke test] subsampled to {len(df):,} rows "
              f"(up to {sample_per_class} per class)")
    X = df[CANONICAL_FEATURES].values
    y = df[TARGET_COL].values
    return X, y, df


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_to_weight = dict(zip(classes, weights))
    return np.array([class_to_weight[label] for label in y])


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    XGBoost's sklearn wrapper requires integer-encoded labels (unlike RF,
    LightGBM, Decision Tree, Logistic Regression, Naive Bayes, which all
    handle string labels fine). We encode here and stash the LabelEncoder
    on the model itself (model.label_encoder_) so evaluate() can decode
    predictions back to the original string labels for comparison.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    sample_weights = compute_sample_weights(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    model.label_encoder_ = label_encoder
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,  # capped -- unbounded depth risks very long training time at 2.5M+ rows
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> LGBMClassifier:
    sample_weights = compute_sample_weights(y_train)
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Single unensembled tree -- shows the value XGBoost's ensembling adds."""
    model = DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, scaler: StandardScaler):
    """Linear baseline. Requires scaled features -- pass a fitted scaler."""
    X_scaled = scaler.transform(X_train)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_train)
    return model


def train_naive_bayes(X_train, y_train) -> GaussianNB:
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str, suffix: str = ""):
    path = MODELS_DIR / f"{name}{suffix}.joblib"
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(name: str, suffix: str = ""):
    path = MODELS_DIR / f"{name}{suffix}.joblib"
    return joblib.load(path)