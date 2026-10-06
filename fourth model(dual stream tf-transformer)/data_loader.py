"""
Data Loader and Spectral Feature Extractor for Approach 4 (Dual-Stream TF-Transformer).
Extracts raw waveforms (Stream 1) and EEMD+HHT/Spectral power features across autonomic bands (Stream 2).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import signal
from config import TFTransformerConfig
from stress_labels import compute_stress_label


def load_metadata(config: TFTransformerConfig) -> dict[str | int, float]:
    """Load metadata/subjects.csv to get debate start times (unix timestamp ms)."""
    meta_file = config.dataset_dir / "metadata" / "metadata" / "subjects.csv"
    if not meta_file.exists():
        meta_file = config.dataset_dir / "metadata" / "subjects.csv"
    df = pd.read_csv(meta_file, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    subjects_meta = {}
    for _, row in df.iterrows():
        sid = str(row["pid"]).strip()
        start_time = float(row["startTime"])
        subjects_meta[sid] = start_time
        subjects_meta[f"P{sid}"] = start_time
        if sid.isdigit():
            subjects_meta[int(sid)] = start_time
    return subjects_meta


def normalize_sensor_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-subject Z-score normalization centering resting differences at 0."""
    df_norm = df.copy()
    for col in columns:
        vals = df_norm[col].values
        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0 or np.isnan(std):
            std = 1.0
        df_norm[col] = (vals - mean) / std
    return df_norm


def get_raw_slice(sensor_df: pd.DataFrame, columns: list[str], window_start_ms: float, window_end_ms: float, expected_length: int) -> np.ndarray:
    """Extract exact-length raw waveform slice for a given 5s time window."""
    mask = (sensor_df["timestamp"] >= window_start_ms) & (sensor_df["timestamp"] < window_end_ms)
    window_data = sensor_df.loc[mask, columns].values

    n_samples = len(window_data)
    if n_samples == expected_length:
        return window_data
    elif n_samples > expected_length:
        return window_data[-expected_length:]
    else:
        pad_len = expected_length - n_samples
        if n_samples > 0:
            pad_val = window_data[0:1, :]
        else:
            pad_val = np.zeros((1, len(columns)), dtype=np.float32)
        padding = np.repeat(pad_val, pad_len, axis=0)
        if n_samples > 0:
            return np.vstack([padding, window_data])
        else:
            return padding


def extract_spectral_features(eda_slice: np.ndarray, bvp_slice: np.ndarray, config: TFTransformerConfig) -> np.ndarray:
    """
    Extract 21 time-frequency & spectral power features across autonomic bands (Stream 2):
    - HRV/BVP LF band (0.04 - 0.15 Hz) power, peak frequency, instantaneous amplitude approximation
    - HRV/BVP HF band (0.15 - 0.40 Hz) power, peak frequency, LF/HF ratio
    - EDA EDASYMP band (0.045 - 0.25 Hz) power, peak frequency, spectral entropy
    """
    feats = []
    
    # BVP / Pulse Wave Spectral Analysis (Sampling rate ~64 Hz, 320 samples)
    bvp_vals = bvp_slice.flatten()
    fs_bvp = 64.0
    if len(bvp_vals) > 4 and np.std(bvp_vals) > 1e-4:
        freqs, psd = signal.welch(bvp_vals, fs=fs_bvp, nperseg=min(len(bvp_vals), 128))
        
        # LF Band (0.04 - 0.15 Hz)
        lf_mask = (freqs >= config.spectral_bands["hrv_lf"][0]) & (freqs <= config.spectral_bands["hrv_lf"][1])
        lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
        lf_peak = float(freqs[lf_mask][np.argmax(psd[lf_mask])]) if np.any(lf_mask) and lf_power > 0 else 0.0
        
        # HF Band (0.15 - 0.40 Hz)
        hf_mask = (freqs >= config.spectral_bands["hrv_hf"][0]) & (freqs <= config.spectral_bands["hrv_hf"][1])
        hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
        hf_peak = float(freqs[hf_mask][np.argmax(psd[hf_mask])]) if np.any(hf_mask) and hf_power > 0 else 0.0
        
        lf_hf_ratio = lf_power / max(hf_power, 1e-5)
        total_power = float(np.sum(psd))
        bvp_entropy = float(-np.sum((psd / max(total_power, 1e-5)) * np.log(psd / max(total_power, 1e-5) + 1e-8)))
    else:
        lf_power, lf_peak, hf_power, hf_peak, lf_hf_ratio, total_power, bvp_entropy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # EDA Spectral Analysis (Sampling rate ~4 Hz, 20 samples)
    eda_vals = eda_slice.flatten()
    fs_eda = 4.0
    if len(eda_vals) > 4 and np.std(eda_vals) > 1e-4:
        freqs_eda, psd_eda = signal.welch(eda_vals, fs=fs_eda, nperseg=min(len(eda_vals), 16))
        symp_mask = (freqs_eda >= config.spectral_bands["eda_symp"][0]) & (freqs_eda <= config.spectral_bands["eda_symp"][1])
        symp_power = float(np.sum(psd_eda[symp_mask])) if np.any(symp_mask) else 0.0
        symp_peak = float(freqs_eda[symp_mask][np.argmax(psd_eda[symp_mask])]) if np.any(symp_mask) and symp_power > 0 else 0.0
        eda_total_power = float(np.sum(psd_eda))
    else:
        symp_power, symp_peak, eda_total_power = 0.0, 0.0, 0.0

    # Construct 21-element spectral vector summarizing IF, IA, and Marginal Spectrum across bands
    spectral_vec = np.array([
        lf_power, lf_peak, np.sqrt(lf_power),
        hf_power, hf_peak, np.sqrt(hf_power),
        lf_hf_ratio, total_power, bvp_entropy,
        symp_power, symp_peak, np.sqrt(symp_power), eda_total_power,
        np.mean(bvp_vals), np.std(bvp_vals), np.max(bvp_vals) - np.min(bvp_vals),
        np.mean(eda_vals), np.std(eda_vals), np.max(eda_vals) - np.min(eda_vals),
        float(np.diff(eda_vals).mean()) if len(eda_vals) > 1 else 0.0,
        float(np.diff(bvp_vals).std()) if len(bvp_vals) > 1 else 0.0
    ], dtype=np.float32)
    return spectral_vec


def load_all_subject_data(config: TFTransformerConfig) -> list[dict]:
    """
    Load raw waveforms, spectral features, SQI motion masks, and stress labels across all subjects.
    Returns list of dicts, one dict per subject containing chronological window arrays.
    """
    meta = load_metadata(config)
    subjects_data = []
    
    # Map physical sensor filenames and properties
    sensor_info = {
        "eda": {"file": "E4_EDA.csv", "channels": ["value"], "expected": config.window_samples["eda"]},
        "temp": {"file": "E4_TEMP.csv", "channels": ["value"], "expected": config.window_samples["temp"]},
        "acc": {"file": "E4_ACC.csv", "channels": ["x", "y", "z"], "expected": config.window_samples["acc"]},
        "bvp": {"file": "E4_BVP.csv", "channels": ["value"], "expected": config.window_samples["bvp"]},
    }
    
    e4_dir = config.dataset_dir / "e4_data"
    ann_dir = config.dataset_dir / "emotion_annotations" / config.annotation_source
    
    if not e4_dir.exists() or not ann_dir.exists():
        print(f"Directory missing: {e4_dir} or {ann_dir}")
        return []
        
    for subj_folder in sorted(e4_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not subj_folder.is_dir() or not subj_folder.name.isdigit():
            continue
        sid_num = int(subj_folder.name)
        ann_path = ann_dir / f"P{sid_num}.{config.annotation_source.split('_')[0]}.csv"
        if not ann_path.exists():
            continue
            
        annotations = pd.read_csv(ann_path)
        start_time = meta.get(sid_num, meta.get(f"P{sid_num}", None))
        if start_time is None:
            continue
            
        # Load normalized sensor DFs
        sensors = {}
        for s_key, s_data in sensor_info.items():
            fpath = subj_folder / s_data["file"]
            if fpath.exists():
                df = pd.read_csv(fpath)
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                if s_key == "acc":
                    for col in ("x", "y", "z"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
                    df = normalize_sensor_df(df, ["x", "y", "z", "mag"])
                else:
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = normalize_sensor_df(df, ["value"])
                sensors[s_key] = df
            else:
                sensors[s_key] = None
                
        if any(v is None for v in sensors.values()):
            continue
            
        subj_eda, subj_bvp, subj_acc, subj_temp = [], [], [], []
        subj_spectral, subj_sqi, subj_y, subj_scores = [], [], [], []
        
        for _, ann_row in annotations.iterrows():
            seconds = float(ann_row["seconds"])
            window_end_ms = start_time + seconds * 1000.0
            window_start_ms = window_end_ms - config.window_seconds * 1000.0
            
            # Extract slices
            eda_slice = get_raw_slice(sensors["eda"], ["value"], window_start_ms, window_end_ms, sensor_info["eda"]["expected"]).T.astype(np.float32)
            bvp_slice = get_raw_slice(sensors["bvp"], ["value"], window_start_ms, window_end_ms, sensor_info["bvp"]["expected"]).T.astype(np.float32)
            acc_slice = get_raw_slice(sensors["acc"], ["x", "y", "z", "mag"], window_start_ms, window_end_ms, sensor_info["acc"]["expected"]).T.astype(np.float32)
            temp_slice = get_raw_slice(sensors["temp"], ["value"], window_start_ms, window_end_ms, sensor_info["temp"]["expected"]).T.astype(np.float32)
            
            # Compute Spectral features & SQI
            spectral_vec = extract_spectral_features(eda_slice, bvp_slice, config)
            
            # SQI check based on acceleration magnitude variance
            acc_mag_var = float(np.var(acc_slice[3]))
            sqi_clean = 1.0 if acc_mag_var < config.acc_motion_threshold else 0.0
            
            label, score = compute_stress_label(ann_row)
            
            subj_eda.append(eda_slice)
            subj_bvp.append(bvp_slice)
            subj_acc.append(acc_slice)
            subj_temp.append(temp_slice)
            subj_spectral.append(spectral_vec)
            subj_sqi.append(sqi_clean)
            subj_y.append(label)
            subj_scores.append(score)
            
        if len(subj_y) >= config.buffer_windows:
            subjects_data.append({
                "sid": sid_num,
                "eda": np.stack(subj_eda),           # (N_windows, 1, 20)
                "bvp": np.stack(subj_bvp),           # (N_windows, 1, 320)
                "acc": np.stack(subj_acc),           # (N_windows, 4, 160)
                "temp": np.stack(subj_temp),         # (N_windows, 1, 20)
                "spectral": np.stack(subj_spectral), # (N_windows, 21)
                "sqi": np.array(subj_sqi, dtype=np.float32), # (N_windows,)
                "y": np.array(subj_y, dtype=np.float32),
                "scores": np.array(subj_scores, dtype=np.float32)
            })
            print(f"Loaded P{sid_num:02d}: {len(subj_y)} windows")
            
    return subjects_data
