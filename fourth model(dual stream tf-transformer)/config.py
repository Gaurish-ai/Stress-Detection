from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TFTransformerConfig:
    """Configuration for Dual-Stream Temporal-Frequency Transformer (TF-Transformer)."""
    dataset_dir: Path = Path("DATASET")
    annotation_source: str = "self_annotations"
    output_dir: Path = Path("gemini model/fourth model(dual stream tf-transformer)/outputs")
    
    # Window and Buffer Settings
    window_seconds: int = 5
    buffer_windows: int = 12  # 12 consecutive 5s windows = 60s temporal context
    
    # Physical Sensor Sampling Rates and Samples per 5s Window
    window_samples: dict[str, int] = field(
        default_factory=lambda: {
            "eda": 20,       # GSR sensor at 4 Hz
            "temp": 20,      # DS18B20 temperature sensor at 4 Hz
            "acc": 160,      # MPU6500 IMU (x, y, z, mag) at 32 Hz
            "bvp": 320,      # MAX30102 pulse sensor (BVP) at 64 Hz
        }
    )
    
    channels: dict[str, list[str]] = field(
        default_factory=lambda: {
            "eda": ["value"],
            "temp": ["value"],
            "bvp": ["value"],
            "acc": ["x", "y", "z", "mag"],
        }
    )
    
    # EEMD + HHT Frequency Bands of Interest (Selvam et al., 2025)
    spectral_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hrv_lf": (0.04, 0.15),       # Sympathetic activity
            "hrv_hf": (0.15, 0.40),       # Parasympathetic activity
            "eda_symp": (0.045, 0.25),    # EDASYMP sympathetic activity band
        }
    )
    
    # Model Architecture Dimensions
    cnn_embed_dim: int = 64
    spectral_embed_dim: int = 32
    fusion_dim: int = 96  # cnn_embed_dim + spectral_embed_dim
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_dim: int = 256
    dropout: float = 0.25
    
    # Quality Screening Thresholds
    acc_motion_threshold: float = 2.5     # Acceleration magnitude variance threshold for SQI mask
    eda_disconnect_threshold: float = 0.03  # Minimum valid EDA conductance (uS)
    
    seed: int = 42
