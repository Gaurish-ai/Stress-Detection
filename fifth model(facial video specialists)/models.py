from __future__ import annotations
import torch
from torch import nn


class FacialBiLSTMAttention(nn.Module):
    """
    Model 5B: Bidirectional LSTM with Temporal Attention over 60s Rolling Visual Sequences.
    Tracks behavioral trajectories (leaning forward, gaze shifts, facial rigidity) across 12 consecutive windows.
    """
    def __init__(self, num_features: int = 11, hidden_dim: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_len=12, num_features=11)
        lstm_out, _ = self.lstm(x)  # (batch_size, sequence_len, hidden_dim * 2)
        
        # Calculate temporal attention scores across the 12 windows
        scores = self.attn_weights(lstm_out)  # (batch_size, sequence_len, 1)
        attn_dist = torch.softmax(scores, dim=1)
        
        # Weighted sum across time
        pooled = torch.sum(lstm_out * attn_dist, dim=1)  # (batch_size, hidden_dim * 2)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, pooled


class FacialVideoTransformer(nn.Module):
    """
    Model 5C: Multi-Head Self-Attention Transformer over 60s Rolling Visual Sequences.
    Correlates non-contiguous windows, isolating rapid motion bursts from sustained facial agitation.
    """
    def __init__(
        self,
        num_features: int = 11,
        embed_dim: int = 64,
        sequence_len: int = 12,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_len, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Target-window attention query (attends over historical sequence using latest window at t)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_len=12, num_features=11)
        tokens = self.feature_projection(x) + self.positional_encoding
        encoded = self.transformer(tokens)  # (batch_size, sequence_len, embed_dim)
        
        # Query using latest window (end of buffer)
        query = self.query_proj(encoded[:, -1:, :])  # (batch_size, 1, embed_dim)
        keys = self.key_proj(encoded)                # (batch_size, sequence_len, embed_dim)
        
        # Attention scores between latest window and all historical windows
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / (keys.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        pooled = torch.bmm(attn_weights, encoded).squeeze(1)  # (batch_size, embed_dim)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, pooled
