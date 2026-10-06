from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Pop local config from sys.modules if train.py already imported it
_local_config = sys.modules.pop("config", None)

# 2. Setup paths for modular_stress_pipeline and project root
PROJECT_DIR = Path("d:/Sensor Stress")
MODULAR_DIR = PROJECT_DIR / "modular_stress_pipeline"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

# 3. Load modular config and bind to sys.modules['config'] so internal imports inside dataset.py succeed
import modular_stress_pipeline.config as _mod_config_module
sys.modules["config"] = _mod_config_module
from modular_stress_pipeline.dataset import build_window_dataset
from modular_stress_pipeline.config import PipelineConfig as ModularConfig

# 4. Restore paths and local sys.modules['config']
if str(MODULAR_DIR) in sys.path:
    sys.path.remove(str(MODULAR_DIR))
if _local_config is not None:
    sys.modules["config"] = _local_config
else:
    sys.modules.pop("config", None)

from config import FacialSpecialistsConfig


def load_and_align_video_windows(config: FacialSpecialistsConfig) -> list[dict]:
    """
    Loads all synchronized 5-second windows for the 18 common subjects, merging ground-truth
    stress labels with the 11 continuous visual/facial features. Constructs 12-window (60s)
    rolling circular sequences for our Facial Models (XGBoost, BiLSTM, and Transformer).
    """
    if not config.cached_video_features_csv.exists():
        raise FileNotFoundError(f"Cached video features not found at {config.cached_video_features_csv}")

    print("Step 1: Loading base annotations and matching with cached 11 video features...")
    modular_cfg = ModularConfig(
        dataset_dir=config.project_dir / "DATASET",
        annotation_source="self_annotations",
        output_dir=config.output_dir
    )
    sensors_raw, y_all, groups_all, rows = build_window_dataset(modular_cfg)
    
    # Filter for the 18 common subjects
    keep_mask = rows["pid"].isin(config.common_subjects).to_numpy()
    rows = rows[keep_mask].reset_index(drop=True)
    y_filtered = y_all[keep_mask]
    
    # Load cached video table and merge on pid and seconds
    video_df = pd.read_csv(config.cached_video_features_csv)
    rows["y"] = y_filtered
    merged = rows.merge(video_df, on=["pid", "seconds"], how="inner").sort_values(["pid", "seconds"])
    
    subjects_data = []
    buf = config.buffer_windows
    
    for pid in sorted(config.common_subjects):
        subj_rows = merged[merged["pid"] == pid].copy()
        if len(subj_rows) < buf:
            print(f"Skipping P{pid:02d}: only {len(subj_rows)} windows (need at least {buf})")
            continue
            
        # Extract 11 visual features matrix: (N_windows, 11)
        video_feats = subj_rows[config.video_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        y_vals = subj_rows["y"].to_numpy(dtype=np.float32)
        
        # Build 12-window (60s) rolling circular sequences
        n_win = len(y_vals)
        seq_count = n_win - buf + 1
        
        video_seqs = np.zeros((seq_count, buf, len(config.video_feature_cols)), dtype=np.float32)
        y_seqs = np.zeros((seq_count,), dtype=np.float32)
        
        for idx in range(seq_count):
            end_idx = idx + buf
            video_seqs[idx] = video_feats[idx:end_idx]
            # Target is the current / latest window in the rolling buffer
            y_seqs[idx] = y_vals[end_idx - 1]
            
        # Also compute handcrafted 77 statistical summary features for Model 5A (XGBoost)
        # For each of the 11 features across the 12 windows: mean, std, min, max, range, slope, delta
        xgb_feats = np.zeros((seq_count, len(config.video_feature_cols) * 7), dtype=np.float32)
        for idx in range(seq_count):
            window_slice = video_seqs[idx]  # shape: (12, 11)
            f_mean = np.mean(window_slice, axis=0)
            f_std = np.std(window_slice, axis=0)
            f_min = np.min(window_slice, axis=0)
            f_max = np.max(window_slice, axis=0)
            f_range = f_max - f_min
            f_delta = window_slice[-1] - window_slice[0]
            # Slope approximation across 12 windows using linear regression or diff mean
            f_slope = np.mean(np.diff(window_slice, axis=0), axis=0)
            
            summary_vec = np.concatenate([f_mean, f_std, f_min, f_max, f_range, f_delta, f_slope])
            xgb_feats[idx] = summary_vec
            
        subjects_data.append({
            "sid": pid,
            "video_seqs": video_seqs,   # (seq_count, 12, 11) for BiLSTM / Transformer
            "xgb_feats": xgb_feats,     # (seq_count, 77) for XGBoost
            "y": y_seqs                 # (seq_count,)
        })
        print(f"Subject P{pid:02d} loaded -> {seq_count} rolling 60s visual sequences")
        
    return subjects_data


def get_loso_splits(subjects_data: list[dict], test_sid: int) -> tuple[dict, dict]:
    """Splits subjects_data into pooled training arrays vs held-out test subject arrays."""
    train_seqs, train_xgb, train_y = [], [], []
    test_seqs, test_xgb, test_y = None, None, None
    
    for subj in subjects_data:
        if subj["sid"] == test_sid:
            test_seqs = subj["video_seqs"]
            test_xgb = subj["xgb_feats"]
            test_y = subj["y"]
        else:
            train_seqs.append(subj["video_seqs"])
            train_xgb.append(subj["xgb_feats"])
            train_y.append(subj["y"])
            
    train_dict = {
        "video_seqs": np.concatenate(train_seqs, axis=0) if train_seqs else np.array([]),
        "xgb_feats": np.concatenate(train_xgb, axis=0) if train_xgb else np.array([]),
        "y": np.concatenate(train_y, axis=0) if train_y else np.array([])
    }
    test_dict = {
        "video_seqs": test_seqs if test_seqs is not None else np.array([]),
        "xgb_feats": test_xgb if test_xgb is not None else np.array([]),
        "y": test_y if test_y is not None else np.array([])
    }
    return train_dict, test_dict
