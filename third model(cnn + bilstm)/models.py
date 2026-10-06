"""
Architecture for Approach 3: Multi-Branch 1D CNN + BiLSTM + Temporal Self-Attention.

Preserves exact temporal sequence ordering across the 30-second window, models
trajectory accumulation with a Bidirectional LSTM, and dynamically focuses on acute
inflection moments using Temporal Self-Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DROPOUT_RATE, LSTM_HIDDEN_DIM


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism that assigns dynamic importance weights across all sequence time steps.
    Focuses the network on exact timestamps where physiological spikes occur.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        """
        Args:
            lstm_out: (Batch, Seq_Len, hidden_dim) float32 tensor
        Returns:
            context: (Batch, hidden_dim) weighted sum vector
            attn_probs: (Batch, Seq_Len, 1) attention probabilities
        """
        scores = self.attn_weights(lstm_out)           # (Batch, Seq_Len, 1)
        attn_probs = F.softmax(scores, dim=1)          # (Batch, Seq_Len, 1)
        context = torch.sum(lstm_out * attn_probs, dim=1)  # (Batch, hidden_dim)
        return context, attn_probs


class MultiBranch1DCNN_BiLSTM_Attention(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE, lstm_hidden=LSTM_HIDDEN_DIM):
        super(MultiBranch1DCNN_BiLSTM_Attention, self).__init__()

        # --- Branch 1: Accelerometer (3 channels, 960 samples -> 60 steps) ---
        self.acc_branch = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # shape: (B, 32, 240)
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # shape: (B, 64, 60)
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()                               # shape: (B, 64, 60)
        )

        # --- Branch 2: EDA + TEMP (2 channels, 120 samples -> 60 steps) ---
        self.eda_temp_branch = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # shape: (B, 32, 60)
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()                               # shape: (B, 64, 60)
        )

        # --- Branch 3: Heart Rate (1 channel, 30 samples -> upsampled to 60 steps) ---
        self.hr_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()                               # shape: (B, 32, 60)
        )

        # --- Bidirectional LSTM (Input: 64 + 64 + 32 = 160 channels) ---
        self.bilstm = nn.LSTM(
            input_size=160,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- Temporal Self-Attention ---
        self.attention = TemporalAttention(hidden_dim=lstm_hidden * 2)

        # --- Classifier Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 2)
        )

    def forward(self, acc, eda, temp, hr):
        """
        Forward pass through CNN + BiLSTM + Attention sequence model.

        Args:
            acc:  (B, 3, 960) float32 tensor
            eda:  (B, 1, 120) float32 tensor
            temp: (B, 1, 120) float32 tensor
            hr:   (B, 1, 30)  float32 tensor

        Returns:
            logits: (B, 2) float32 tensor
        """
        acc_seq = self.acc_branch(acc)                             # (B, 64, 60)
        
        eda_temp = torch.cat([eda, temp], dim=1)                   # (B, 2, 120)
        eda_temp_seq = self.eda_temp_branch(eda_temp)              # (B, 64, 60)
        
        hr_interp = F.interpolate(hr, size=60, mode="linear", align_corners=False)
        hr_seq = self.hr_branch(hr_interp)                         # (B, 32, 60)

        # Concatenate multi-sensor channels across all 60 time steps -> (B, 160, 60)
        fused = torch.cat([acc_seq, eda_temp_seq, hr_seq], dim=1)
        
        # Transpose to (Batch, Seq_Len=60, Channels=160) for BiLSTM
        fused_seq = fused.transpose(1, 2)
        
        lstm_out, _ = self.bilstm(fused_seq)                       # (B, 60, 128)
        
        context, attn_probs = self.attention(lstm_out)             # (B, 128)
        
        logits = self.classifier(context)                          # (B, 2)
        return logits
