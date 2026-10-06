from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBranch(nn.Module):
    """1D CNN extractor for a single sensor window sequence (e.g. EDA @ 4Hz over 5s = 20 samples)."""
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        return self.pool(h).squeeze(-1)


class RawPhysioTFTransformer(nn.Module):
    """
    Model 4 V2: E4 Physiological TF-Transformer (with per-subject resting baseline normalized inputs).
    Processes rolling 12-window sequences through multi-scale Conv1D branches & Temporal Transformer.
    """
    def __init__(self, embed_dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.eda = ConvBranch(1, embed_dim)
        self.bvp = ConvBranch(1, embed_dim)
        self.acc = ConvBranch(4, embed_dim)
        self.temp = ConvBranch(1, embed_dim)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, 12, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, sensors_seq: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sensors_seq: dict where each value is (batch_size, 12, window_samples)
        Returns:
            logits: (batch_size,)
            physio_tokens: (batch_size, embed_dim)
        """
        batch_size, seq_len = sensors_seq["eda"].shape[0], sensors_seq["eda"].shape[1]
        
        eda_flat = sensors_seq["eda"].reshape(batch_size * seq_len, *sensors_seq["eda"].shape[2:])
        bvp_flat = sensors_seq["bvp"].reshape(batch_size * seq_len, *sensors_seq["bvp"].shape[2:])
        acc_flat = sensors_seq["acc"].reshape(batch_size * seq_len, *sensors_seq["acc"].shape[2:])
        temp_flat = sensors_seq["temp"].reshape(batch_size * seq_len, *sensors_seq["temp"].shape[2:])
        
        eda_tok = self.eda(eda_flat).reshape(batch_size, seq_len, -1)
        bvp_tok = self.bvp(bvp_flat).reshape(batch_size, seq_len, -1)
        acc_tok = self.acc(acc_flat).reshape(batch_size, seq_len, -1)
        temp_tok = self.temp(temp_flat).reshape(batch_size, seq_len, -1)
        
        combined_windows = (eda_tok + bvp_tok + acc_tok + temp_tok) / 4.0
        encoded = self.transformer(combined_windows + self.pos_embedding)
        
        physio_token = encoded[:, -1, :]
        logits = self.classifier(physio_token).squeeze(-1)
        return logits, physio_token


class IntermediateCrossAttentionFuser(nn.Module):
    """
    Model 6B: Intermediate Token Cross-Attention Fusion.
    Fuses the 64-dim physiological token from RawPhysioTFTransformer with the 77-dim visual XGBoost features.
    """
    def __init__(self, physio_dim: int = 64, visual_dim: int = 77, embed_dim: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.physio_proj = nn.LayerNorm(physio_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(physio_dim + embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, physio_token: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        p_norm = self.physio_proj(physio_token)
        v_tok = self.visual_proj(visual_feats)
        fused = torch.cat([p_norm, v_tok], dim=-1)
        return self.fusion_mlp(fused).squeeze(-1)


class EarlySynchronizedFuser(nn.Module):
    """
    Model 7 / Model 6C: Synchronized Early Feature Fusion (Step 2).
    Fuses window-level visual features and E4 sensor features right at Layer 0 (before temporal modeling),
    and passes the unified multi-stream sequence through a shared Temporal Transformer Encoder.
    """
    def __init__(self, physio_dim: int = 64, visual_dim: int = 11, embed_dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.eda = ConvBranch(1, physio_dim)
        self.bvp = ConvBranch(1, physio_dim)
        self.acc = ConvBranch(4, physio_dim)
        self.temp = ConvBranch(1, physio_dim)
        
        self.video_proj = nn.Sequential(
            nn.Linear(visual_dim, physio_dim),
            nn.LayerNorm(physio_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Early Fusion projection right at Layer 0 combining physical & visual window features
        self.early_fusion_proj = nn.Sequential(
            nn.Linear(physio_dim + physio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, 12, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, sensors_seq: dict[str, torch.Tensor], video_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sensors_seq: dict of (batch_size, 12, window_samples)
        video_seq: (batch_size, 12, visual_dim=11)
        Returns:
            logits: (batch_size,)
            fused_token: (batch_size, embed_dim)
        """
        batch_size, seq_len = sensors_seq["eda"].shape[0], sensors_seq["eda"].shape[1]
        
        eda_flat = sensors_seq["eda"].reshape(batch_size * seq_len, *sensors_seq["eda"].shape[2:])
        bvp_flat = sensors_seq["bvp"].reshape(batch_size * seq_len, *sensors_seq["bvp"].shape[2:])
        acc_flat = sensors_seq["acc"].reshape(batch_size * seq_len, *sensors_seq["acc"].shape[2:])
        temp_flat = sensors_seq["temp"].reshape(batch_size * seq_len, *sensors_seq["temp"].shape[2:])
        
        eda_tok = self.eda(eda_flat).reshape(batch_size, seq_len, -1)
        bvp_tok = self.bvp(bvp_flat).reshape(batch_size, seq_len, -1)
        acc_tok = self.acc(acc_flat).reshape(batch_size, seq_len, -1)
        temp_tok = self.temp(temp_flat).reshape(batch_size, seq_len, -1)
        
        physio_windows = (eda_tok + bvp_tok + acc_tok + temp_tok) / 4.0
        visual_windows = self.video_proj(video_seq)
        
        # --- EARLY FUSION AT LAYER 0 across time sequence ---
        joint_windows = torch.cat([physio_windows, visual_windows], dim=-1)
        early_tokens = self.early_fusion_proj(joint_windows)
        # ---------------------------------------------------
        
        encoded = self.transformer(early_tokens + self.pos_embedding)
        fused_token = encoded[:, -1, :]
        logits = self.classifier(fused_token).squeeze(-1)
        
        return logits, fused_token


def optimize_late_fusion_weight(y_true: np.ndarray, probs_phys: np.ndarray, probs_vis: np.ndarray) -> float:
    """Finds optimal linear weight alpha for Late Probability Fusion on the training fold."""
    best_alpha = 0.5
    best_acc = -1.0
    for alpha in np.linspace(0.0, 1.0, 101):
        p_comb = alpha * probs_phys + (1.0 - alpha) * probs_vis
        pred = (p_comb >= 0.5).astype(int)
        acc = float(np.mean(pred == y_true))
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)
    return best_alpha
