"""
LSTM-based architectures for MotionSense HAR (baseline + enhanced).
"""

import torch
import torch.nn as nn

# Feature dimensions for MotionSense (accelerometer + gyroscope etc.)
FEATURE_DIM = 12
NUM_CLASSES = 6


# === Baseline ===
class MotionSenseLSTM(nn.Module):
    """
    Baseline LSTM model:
      Input: (batch_size, seq_len=64, features=12)
      Architecture:
        - 1-layer unidirectional LSTM (hidden_dim=64)
        - Fully connected classifier
        - Softmax applied during loss/metrics (not inside forward)
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)         # hn shape: (1, B, H)
        out = self.fc(hn[-1])             # (B, num_classes)
        return out


# === Enhanced ===
class MotionSenseBiLSTMProj(nn.Module):
    """
    Enhanced BiLSTM with projection head:
      Input: (batch_size, seq_len=64, features=12)
      Architecture:
        - 1-layer bidirectional LSTM (hidden_dim=64 per direction → 128)
        - Concatenate last hidden states from forward & backward
        - Projection head: LayerNorm → Dense → GELU → Dropout
        - Output classifier
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        num_classes: int = NUM_CLASSES,
        num_layers: int = 1,
        dropout: float = 0.2,
        proj_mult: int = 2,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d_in = 2 * hidden_dim
        d_mid = proj_mult * hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.bilstm(x)                       # hn shape: (2, B, H)
        h_cat = torch.cat([hn[-2], hn[-1]], dim=1)        # (B, 2*hidden_dim)
        return self.head(h_cat)
