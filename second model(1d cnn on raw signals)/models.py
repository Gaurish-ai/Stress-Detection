"""
Deep Learning Architecture for Approach 2: Multi-Branch 1D CNN.

Processes each raw physiological sensor waveform (ACC at 32 Hz, EDA & TEMP at 4 Hz,
HR at 1 Hz) in its own specialized 1D convolutional branch, then fuses the temporal
embeddings for binary stress classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DROPOUT_RATE


class MultiBranch1DCNN(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super(MultiBranch1DCNN, self).__init__()

        # --- Branch 1: Accelerometer (3 channels, 960 samples @ 32 Hz) ---
        self.acc_branch = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # shape: (B, 32, 240)
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # shape: (B, 64, 60)
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # shape: (B, 128, 1)
        )

        # --- Branch 2: EDA + TEMP (2 channels, 120 samples @ 4 Hz) ---
        self.eda_temp_branch = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # shape: (B, 32, 60)
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # shape: (B, 64, 1)
        )

        # --- Branch 3: Heart Rate / Pulse (1 channel, 30 samples @ 1 Hz) ---
        self.hr_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # shape: (B, 16, 15)
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # shape: (B, 32, 1)
        )

        # --- Fusion Classifier Head (128 + 64 + 32 = 224 dims) ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(224, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 2)  # 2 classes: Not Stressed (0) vs Stressed (1)
        )

    def forward(self, acc, eda, temp, hr):
        """
        Forward pass through multi-branch 1D CNN.

        Args:
            acc:  (B, 3, 960) float32 tensor
            eda:  (B, 1, 120) float32 tensor
            temp: (B, 1, 120) float32 tensor
            hr:   (B, 1, 30)  float32 tensor

        Returns:
            logits: (B, 2) float32 tensor
        """
        acc_feat = self.acc_branch(acc).squeeze(-1)              # (B, 128)
        
        # Concatenate EDA and TEMP along channel axis -> (B, 2, 120)
        eda_temp = torch.cat([eda, temp], dim=1)
        eda_temp_feat = self.eda_temp_branch(eda_temp).squeeze(-1) # (B, 64)
        
        hr_feat = self.hr_branch(hr).squeeze(-1)                 # (B, 32)

        # Concatenate all branch feature embeddings
        fused = torch.cat([acc_feat, eda_temp_feat, hr_feat], dim=1) # (B, 224)
        logits = self.classifier(fused)
        return logits
