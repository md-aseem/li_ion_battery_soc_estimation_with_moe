from torch.utils.data import Dataset

class SequenceBatteryDataSet(Dataset):
    def __init__(self, features, target, device='cuda'):
        super().__init__()

        self.features = features
        self.target = target
        self.device = device

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]