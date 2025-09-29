"""
MLP-based architectures for MotionSense HAR (baseline + enhanced).
"""

import torch
import torch.nn as nn

# Constants
FEATURE_DIM = 12
SEQ_LEN = 64
NUM_CLASSES = 6


# === Baseline MLP ===
class MotionSenseMLP(nn.Module):
    """
    Baseline MLP:
      - Flatten input (64 × 12)
      - Linear → ReLU → Linear
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        seq_len: int = SEQ_LEN,
        hidden_dim: int = 64,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# === MLP Block (used in Mixer) ===
class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# === Enhanced MLP-Mixer ===
class MotionSenseMLPMixer(nn.Module):
    """
    Enhanced MLP-Mixer:
      - Token mixer (mixes across features per timestep)
      - Channel mixer (mixes across timesteps for each feature)
      - Repeated mixer blocks
      - Flatten → Classifier
    """
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        input_dim: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        mixer_dim: int = 64,
        num_blocks: int = 4,
        dropout_rate: float = 0.2,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        # Token mixer: feature mixing per timestep
        self.token_mixer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
            ) for _ in range(num_blocks)
        ])

        # Channel mixer: mixes across time
        self.channel_mixer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(seq_len),
                MLPBlock(seq_len, mixer_dim, dropout_rate),
            ) for _ in range(num_blocks)
        ])

        # Classifier on flattened output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for token_mix, channel_mix in zip(self.token_mixer, self.channel_mixer):
            x = x + token_mix(x)          # token mixing
            x = x.transpose(1, 2)         # (B, F, T)
            x = x + channel_mix(x)        # channel mixing
            x = x.transpose(1, 2)         # back to (B, T, F)
        return self.classifier(x)
