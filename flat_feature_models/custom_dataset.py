import torch
from torch.utils.data import Dataset
from typing import List

class BatteryDataset(Dataset):
    def __init__(self, df, feature_cols: List[str], target_col: str):

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return x, y