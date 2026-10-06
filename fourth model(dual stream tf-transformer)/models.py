from __future__ import annotations
import torch
from torch import nn


class ConvBranch1D(nn.Module):
    """Multi-scale 1D Convolutional branch for localized raw sensor waveform extraction (Stream 1)."""
    def __init__(self, input_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size * buffer_windows, channels, seq_len)
        features = self.net(x).squeeze(-1)
        return self.projection(features)


class SpectralEmbeddingBranch(nn.Module):
    """Dense embedding branch for EEMD + HHT Instantaneous Frequency & Marginal Spectrum features (Stream 2)."""
    def __init__(self, num_spectral_features: int = 21, embed_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_spectral_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, spectral_feats: torch.Tensor) -> torch.Tensor:
        # spectral_feats shape: (batch_size * buffer_windows, num_spectral_features)
        return self.net(spectral_feats)


class DualStreamTFTransformer(nn.Module):
    """
    Dual-Stream Temporal-Frequency Transformer (TF-Transformer) Architecture.
    
    Stream 1: Multi-scale 1D CNN processes raw waveforms (EDA, BVP, ACC, TEMP) -> 64-dim token.
    Stream 2: Dense layers process EEMD + HHT spectral features (IF, IA, MS) -> 32-dim token.
    Fusion: Fused 96-dim representation across 12 consecutive windows (60 seconds) is modeled by a 2-layer Transformer Encoder.
    """
    def __init__(
        self,
        cnn_embed_dim: int = 64,
        spectral_embed_dim: int = 32,
        num_spectral_features: int = 21,
        buffer_windows: int = 12,
        heads: int = 4,
        layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.buffer_windows = buffer_windows
        self.fusion_dim = cnn_embed_dim + spectral_embed_dim
        
        # Stream 1: Raw Waveform Conv Branches
        self.eda_branch = ConvBranch1D(input_channels=1, embed_dim=cnn_embed_dim)
        self.bvp_branch = ConvBranch1D(input_channels=1, embed_dim=cnn_embed_dim)
        self.acc_branch = ConvBranch1D(input_channels=4, embed_dim=cnn_embed_dim)
        self.temp_branch = ConvBranch1D(input_channels=1, embed_dim=cnn_embed_dim)
        self.sensor_attention_weights = nn.Parameter(torch.ones(1, 4))
        self.modality_projection = nn.Linear(4 * cnn_embed_dim, cnn_embed_dim)
        
        # Stream 2: Spectral Features (IF, IA, MS from EEMD + HHT)
        self.spectral_branch = SpectralEmbeddingBranch(
            num_spectral_features=num_spectral_features,
            embed_dim=spectral_embed_dim,
            dropout=dropout,
        )
        
        # Positional Encoding for 12 windows (60s temporal sequence)
        self.positional_encoding = nn.Parameter(torch.randn(1, buffer_windows, self.fusion_dim) * 0.02)
        
        # Temporal Sequence Fusion (2-Layer Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Classification Head (Outputs stress logit for right now / latest window in buffer)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        raw_batch: dict[str, torch.Tensor],
        spectral_batch: torch.Tensor,
        sqi_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        raw_batch: Dict of tensors with shapes (batch_size, buffer_windows, channels, seq_len)
        spectral_batch: Tensor of shape (batch_size, buffer_windows, num_spectral_features)
        sqi_mask: Optional boolean mask (batch_size, buffer_windows) indicating clean vs corrupted BVP/EDA windows.
        """
        batch_size, buffer_windows, _ = spectral_batch.shape
        
        # Collapse batch and temporal dimension for Stream 1 & 2 local processing
        eda_flat = raw_batch["eda"].view(batch_size * buffer_windows, 1, -1)
        bvp_flat = raw_batch["bvp"].view(batch_size * buffer_windows, 1, -1)
        acc_flat = raw_batch["acc"].view(batch_size * buffer_windows, 4, -1)
        temp_flat = raw_batch["temp"].view(batch_size * buffer_windows, 1, -1)
        
        # Stream 1: Extract CNN embeddings for each sensor
        e_eda = self.eda_branch(eda_flat)
        e_bvp = self.bvp_branch(bvp_flat)
        e_acc = self.acc_branch(acc_flat)
        e_temp = self.temp_branch(temp_flat)
        
        # Optional SQI masking for motion corrupted BVP
        if sqi_mask is not None:
            mask_flat = sqi_mask.view(-1, 1)
            e_bvp = e_bvp * mask_flat.float()
            
        # Weighted fusion of sensor modalities
        modality_concat = torch.cat([e_eda, e_bvp, e_acc, e_temp], dim=-1)
        cnn_token = self.modality_projection(modality_concat)  # (batch_size * buffer_windows, cnn_embed_dim)
        
        # Stream 2: Extract EEMD+HHT spectral embeddings
        spectral_flat = spectral_batch.view(batch_size * buffer_windows, -1)
        spectral_token = self.spectral_branch(spectral_flat)    # (batch_size * buffer_windows, spectral_embed_dim)
        
        # Merge Stream 1 and Stream 2
        fused_token = torch.cat([cnn_token, spectral_token], dim=-1)  # (batch_size * buffer_windows, 96)
        
        # Reshape back to temporal sequence: (batch_size, buffer_windows=12, fusion_dim=96)
        sequence_tokens = fused_token.view(batch_size, buffer_windows, self.fusion_dim)
        sequence_tokens = sequence_tokens + self.positional_encoding
        
        # Temporal sequence modeling via Transformer Encoder across 60 seconds
        temporal_features = self.transformer(sequence_tokens)
        
        # Predict stress based on the temporal representation (mean pooled or final window focus)
        pooled_context = temporal_features.mean(dim=1)
        logits = self.classifier(pooled_context).squeeze(-1)
        return logits
