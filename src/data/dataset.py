import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset

FEATURE_COLS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
]

TARGET_MAP = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5,
}
IDX_TO_LABEL = {v: k for k, v in TARGET_MAP.items()}

def create_sequences(
    df: pd.DataFrame,
    seq_len: int = 64,
    step: int = 32,
) -> List[Tuple[np.ndarray, str]]:
    """Return [(x:(T,F), y_str)] with majority label per window."""
    pairs = []
    for i in range(0, len(df) - seq_len, step):
        window = df.iloc[i : i + seq_len]
        x = window[FEATURE_COLS].values
        y = Counter(window["Activity"].values).most_common(1)[0][0]
        pairs.append((x, y))
    return pairs

class MotionSenseDataset(Dataset):
    """Takes [(x:(T,F), y_str)] and returns tensors with encoded labels."""
    def __init__(self, sequences: List[Tuple[np.ndarray, str]]):
        X = [torch.tensor(x, dtype=torch.float32) for (x, _) in sequences]
        y = [TARGET_MAP[y_str] for (_, y_str) in sequences]
        self.X = torch.stack(X, dim=0)             # (N, T, F)
        self.y = torch.tensor(y, dtype=torch.long) # (N,)

    def __len__(self): return self.y.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
