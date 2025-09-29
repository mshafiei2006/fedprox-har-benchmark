"""
PatchTST-based architectures for MotionSense HAR (baseline + enhanced).
"""

import torch
import torch.nn as nn

# Constants
FEATURE_DIM = 12
NUM_CLASSES = 6
SEQ_LEN = 64


# === Baseline PatchTST ===
class PatchTSTModel(nn.Module):
    """
    Baseline PatchTST:
      - Conv1D Patch Embedding (patch_len=16, stride=8)
      - Transformer Encoder ×1 (d_model=64, nhead=4, FF=256)
      - Mean pooling over patches
      - Linear classifier
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        patch_len: int = 16,
        stride: int = 8,
        embed_dim: int = 64,
        num_classes: int = NUM_CLASSES,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,
        ff_mult: int = 4,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Conv1D patch embedding
        self.patch_embed = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_len,
            stride=stride,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)             # (B, F, T)
        x = self.patch_embed(x)           # (B, D, P)
        x = x.transpose(1, 2)             # (B, P, D)

        # Transformer encoder
        x = self.transformer(x)           # (B, P, D)

        # Mean pool → classifier
        x = x.mean(dim=1)                 # (B, D)
        return self.head(x)


# === Enhanced PatchTST (CLS + Projection Head) ===
class PatchTSTEnhanced(nn.Module):
    """
    Enhanced PatchTST:
      - Conv1D Patch Embedding (patch_len=16, stride=8)
      - [CLS] token prepend + Positional Embeddings
      - Transformer Encoder ×2 (d_model=64, nhead=4, FF=256)
      - Projection Head: LN → Linear → GELU → Dropout → Linear
      - Classification from CLS token
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        num_classes: int = NUM_CLASSES,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.2,
        ff_mult: int = 4,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_embed = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
        )

        # Num patches for positional embedding
        num_patches = (seq_len - patch_len) // stride + 1

        # CLS token + Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Projection Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, _, _ = x.shape
        x = x.transpose(1, 2)                     # (B, F, T)
        x = self.patch_embed(x)                   # (B, D, P)
        x = x.transpose(1, 2)                     # (B, P, D)

        # Add CLS + Positional Embedding
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat([cls, x], dim=1)            # (B, 1+P, D)
        x = x + self.pos_embed[:, : x.size(1), :]

        # Transformer encoder
        x = self.encoder(x)                       # (B, 1+P, D)

        # Classify from CLS
        cls_out = x[:, 0]                         # (B, D)
        return self.head(cls_out)
