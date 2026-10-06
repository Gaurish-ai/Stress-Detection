"""
PyTorch Dataset definition for Multi-Branch 1D CNN.
"""
import torch
from torch.utils.data import Dataset


class RawSensorDataset(Dataset):
    def __init__(self, acc, eda, temp, hr, y, subject_ids=None):
        """
        Args:
            acc:  (N, 3, 960) numpy array
            eda:  (N, 1, 120) numpy array
            temp: (N, 1, 120) numpy array
            hr:   (N, 1, 30)  numpy array
            y:    (N,) numpy array binary labels
            subject_ids: (N,) optional subject ID array
        """
        self.acc = torch.tensor(acc, dtype=torch.float32)
        self.eda = torch.tensor(eda, dtype=torch.float32)
        self.temp = torch.tensor(temp, dtype=torch.float32)
        self.hr = torch.tensor(hr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = subject_ids if subject_ids is not None else [None] * len(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.acc[idx],
            self.eda[idx],
            self.temp[idx],
            self.hr[idx],
            self.y[idx],
        )
