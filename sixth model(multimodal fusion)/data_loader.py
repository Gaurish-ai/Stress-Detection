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
from modular_stress_pipeline.normalization import fit_train_only_normalizers, apply_normalizers
from modular_stress_pipeline.config import PipelineConfig as ModularConfig

# 4. Restore paths and local sys.modules['config']
if str(MODULAR_DIR) in sys.path:
    sys.path.remove(str(MODULAR_DIR))
if _local_config is not None:
    sys.modules["config"] = _local_config
else:
    sys.modules.pop("config", None)

from config import MultimodalFusionConfig


def load_and_align_multimodal_windows(config: MultimodalFusionConfig) -> list[dict]:
    """
    Loads E4 physiological sensor windows (EDA, BVP, ACC, TEMP) and merges them with
    the 11 continuous visual features from video_features_self_annotations_2frames.csv across
    the 18 common subjects. Constructs 12-window (60s) rolling circular sequences.
    """
    if not config.cached_video_features_csv.exists():
        raise FileNotFoundError(f"Cached video features not found at {config.cached_video_features_csv}")

    print("Step 1: Loading E4 sensor windows and matching with cached video features...")
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
    sensors_filtered = {sensor: values[keep_mask] for sensor, values in sensors_raw.items()}
    
    # Load cached video table and merge on pid and seconds
    video_df = pd.read_csv(config.cached_video_features_csv)
    rows["y"] = y_filtered
    rows["orig_idx"] = np.arange(len(rows))
    
    merged = rows.merge(video_df, on=["pid", "seconds"], how="inner").sort_values(["pid", "seconds"])
    
    subjects_data = []
    buf = config.buffer_windows
    
    for pid in sorted(config.common_subjects):
        subj_rows = merged[merged["pid"] == pid].copy()
        if len(subj_rows) < buf:
            print(f"Skipping P{pid:02d}: only {len(subj_rows)} windows (need at least {buf})")
            continue
            
        orig_indices = subj_rows["orig_idx"].to_numpy(dtype=int)
        subj_sensors = {s: sensors_filtered[s][orig_indices] for s in sensors_filtered}
        video_feats = subj_rows[config.video_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        y_vals = subj_rows["y"].to_numpy(dtype=np.float32)
        
        n_win = len(y_vals)
        seq_count = n_win - buf + 1
        
        # Pre-allocate sequence buffers with full channel/sample shapes
        sensors_seqs = {
            s: np.zeros((seq_count, buf, *subj_sensors[s].shape[1:]), dtype=np.float32)
            for s in subj_sensors
        }
        video_seqs = np.zeros((seq_count, buf, len(config.video_feature_cols)), dtype=np.float32)
        y_seqs = np.zeros((seq_count,), dtype=np.float32)
        
        for idx in range(seq_count):
            end_idx = idx + buf
            for s in sensors_seqs:
                sensors_seqs[s][idx] = subj_sensors[s][idx:end_idx]
            video_seqs[idx] = video_feats[idx:end_idx]
            y_seqs[idx] = y_vals[end_idx - 1]
            
        # Compute 77 summary features for XGBoost across each 12-window slice
        xgb_feats = np.zeros((seq_count, len(config.video_feature_cols) * 7), dtype=np.float32)
        for idx in range(seq_count):
            window_slice = video_seqs[idx]
            f_mean = np.mean(window_slice, axis=0)
            f_std = np.std(window_slice, axis=0)
            f_min = np.min(window_slice, axis=0)
            f_max = np.max(window_slice, axis=0)
            f_range = f_max - f_min
            f_delta = window_slice[-1] - window_slice[0]
            f_slope = np.mean(np.diff(window_slice, axis=0), axis=0)
            
            xgb_feats[idx] = np.concatenate([f_mean, f_std, f_min, f_max, f_range, f_delta, f_slope])
            
        subjects_data.append({
            "sid": pid,
            "sensors_seqs": sensors_seqs, # dict of (seq_count, 12, window_samples)
            "xgb_feats": xgb_feats,       # (seq_count, 77)
            "y": y_seqs                   # (seq_count,)
        })
        print(f"Subject P{pid:02d} loaded -> {seq_count} rolling 60s multimodal sequences")
        
    return subjects_data


def get_normalized_loso_splits(subjects_data: list[dict], test_sid: int) -> tuple[dict, dict]:
    """
    Splits subjects_data into pooled train and held-out test dictionaries, applying
    train-only normalization to the 4 E4 sensor sequences.
    """
    train_sensors = {s: [] for s in ["eda", "bvp", "acc", "temp"]}
    train_xgb, train_y = [], []
    test_sensors = None
    test_xgb, test_y = None, None
    
    for subj in subjects_data:
        if subj["sid"] == test_sid:
            test_sensors = subj["sensors_seqs"]
            test_xgb = subj["xgb_feats"]
            test_y = subj["y"]
        else:
            for s in train_sensors:
                train_sensors[s].append(subj["sensors_seqs"][s])
            train_xgb.append(subj["xgb_feats"])
            train_y.append(subj["y"])
            
    train_dict = {
        "sensors": {s: np.concatenate(train_sensors[s], axis=0) for s in train_sensors},
        "xgb_feats": np.concatenate(train_xgb, axis=0),
        "y": np.concatenate(train_y, axis=0)
    }
    test_dict = {
        "sensors": test_sensors,
        "xgb_feats": test_xgb,
        "y": test_y
    }
    
    # Compute normalization across training sequences (reshape to 2D for mean/std then reshape back)
    for s in train_dict["sensors"]:
        train_arr = train_dict["sensors"][s]
        test_arr = test_dict["sensors"][s]
        shape_t = train_arr.shape
        shape_e = test_arr.shape
        
        flat_t = train_arr.reshape(-1, shape_t[-1])
        flat_e = test_arr.reshape(-1, shape_e[-1])
        
        mean_s = np.mean(flat_t, axis=0, keepdims=True)
        std_s = np.std(flat_t, axis=0, keepdims=True)
        std_s[std_s < 1e-6] = 1.0
        
        norm_t = (flat_t - mean_s) / std_s
        norm_e = (flat_e - mean_s) / std_s
        
        train_dict["sensors"][s] = norm_t.reshape(shape_t).astype(np.float32)
        test_dict["sensors"][s] = norm_e.reshape(shape_e).astype(np.float32)
        
    return train_dict, test_dict
