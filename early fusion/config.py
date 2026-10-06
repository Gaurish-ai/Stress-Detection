from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EarlyFusionConfig:
    """Configuration for Early Fusion & Per-Subject Normalization (Model 4 V2, Model 6A/6B, and Model 7 Early Fusion)."""
    # Paths
    project_dir: Path = Path("d:/Sensor Stress")
    cached_video_features_csv: Path = Path("d:/Sensor Stress/multimodal_stress_pipeline/outputs_smoke/video_features_self_annotations_2frames.csv")
    output_dir: Path = Path("d:/Sensor Stress/gemini model/early fusion/outputs")
    
    # 18 Common Subjects between E4 Sensors and Debate Video Recordings
    common_subjects: list[int] = field(default_factory=lambda: [
        4, 5, 8, 9, 10, 13, 15, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31
    ])
    
    # Sequence and Buffer Settings
    window_seconds: int = 5
    buffer_windows: int = 12  # 12 consecutive 5s windows = 60s temporal context
    
    # Step 1: Per-Subject Resting Baseline Normalization (Crucial upgrade for E4 sensors!)
    use_baseline_subtraction: bool = True
    
    # The 11 Visual/Facial Feature Columns
    video_feature_cols: list[str] = field(default_factory=lambda: [
        "face_count_mean",
        "face_count_max",
        "face_area_ratio_mean",
        "face_area_ratio_max",
        "face_center_x_mean",
        "face_center_y_mean",
        "brightness_mean",
        "brightness_std",
        "contrast_mean",
        "sharpness_mean",
        "motion_mean",
    ])
    
    # Neural Network Dimensions & Training Settings
    cnn_embed_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff_dim: int = 128
    dropout: float = 0.25
    
    # Training Hyperparameters
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    seed: int = 42
