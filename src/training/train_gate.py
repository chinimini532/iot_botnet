"""
Train the Tier-2 MC Dropout MLP (standard training pass -- calibration
of the uncertainty threshold happens in a separate later script, since
it needs bot_iot_val_calibration.csv and MC Dropout inference, not
standard training).

Usage:
    python src/training/train_gate.py                       # full run
    python src/training/train_gate.py --sample-per-class 2000  # smoke test
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from classifier import load_split
from mlp_gate import MCDropoutMLP

SPLITS_DIR = Path("data/processed/splits")
MODELS_DIR = Path("models/checkpoints")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_tensors(X, y, scaler: StandardScaler, label_encoder: LabelEncoder):
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    return X_t, y_t


def main(sample_per_class: int = None, epochs: int = 20, batch_size: int = 512):
    print(f"Using device: {DEVICE}")
    print("Loading training data ...")
    if sample_per_class is not None:
        print(f"*** SMOKE TEST MODE: subsampling to {sample_per_class} rows/class ***")

    X_train, y_train, _ = load_split(SPLITS_DIR / "bot_iot_train.csv", sample_per_class)

    scaler = StandardScaler().fit(X_train)
    label_encoder = LabelEncoder().fit(y_train)
    n_classes = len(label_encoder.classes_)
    print(f"  classes: {list(label_encoder.classes_)}")

    X_t, y_t = prepare_tensors(X_train, y_train, scaler, label_encoder)

    # class weights for the loss (same imbalance-handling principle as Tier 1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_t.numpy(),
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MCDropoutMLP(n_features=X_t.shape[1], n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    print(f"\nTraining MC Dropout MLP for {epochs} epochs ...")
    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.4f}")

    print(f"Training done in {time.time() - t0:.1f}s")

    # Save model + preprocessing artifacts together -- all three are needed
    # at inference time and must stay in sync.
    suffix = "_smoketest" if sample_per_class else ""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_features": X_t.shape[1],
        "n_classes": n_classes,
        "hidden_dim": 64,
    }
    torch.save(checkpoint, MODELS_DIR / f"mc_dropout_mlp{suffix}.pt")
    joblib.dump(scaler, MODELS_DIR / f"mlp_feature_scaler{suffix}.joblib")
    joblib.dump(label_encoder, MODELS_DIR / f"mlp_label_encoder{suffix}.joblib")
    print(f"Saved model + scaler + label encoder to {MODELS_DIR} (suffix='{suffix}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-class", type=int, default=None,
                         help="Smoke-test mode: subsample up to N rows per class.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    main(sample_per_class=args.sample_per_class, epochs=args.epochs,
         batch_size=args.batch_size)