"""
Tier 2: MC Dropout MLP uncertainty gate.

This model's ONLY job is estimating epistemic uncertainty over the same
9-feature input the Tier-1 tree classifier sees. It is deliberately kept
small and simple -- the paper's novelty is the uncertainty-gating
framework + real edge deployment, not architectural complexity (see
earlier project notes on why CNN/LSTM/Transformer were ruled out).

At inference, MC Dropout keeps dropout ACTIVE (unlike normal inference)
and runs N stochastic forward passes. The variance across those passes
estimates epistemic uncertainty: high variance -> the model is unsure,
which is exactly the signal used to flag zero-day / unknown traffic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATURES = 9
DROPOUT_RATE = 0.2


class MCDropoutMLP(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = 4,
                 hidden_dim: int = 128, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        return self.fc4(x)  # raw logits -- softmax applied outside


def enable_mc_dropout(model: nn.Module):
    """
    Keep dropout layers active during inference (normally .eval() would
    disable them). Everything else (batchnorm, etc. -- none used here,
    but future-proofing) stays in eval mode.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 30):
    """
    Run N stochastic forward passes with dropout active.

    Returns:
        mean_probs: (batch, n_classes) -- averaged softmax probabilities
        uncertainty: (batch,) -- predictive variance (mean over classes
                     of per-class variance across passes), the score used
                     for the zero-day uncertainty gate
    """
    enable_mc_dropout(model)

    all_probs = []
    for _ in range(n_passes):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)  # (n_passes, batch, n_classes)
    mean_probs = stacked.mean(dim=0)
    variance = stacked.var(dim=0)  # (batch, n_classes)
    uncertainty = variance.mean(dim=1)  # (batch,) -- single scalar per sample

    return mean_probs, uncertainty