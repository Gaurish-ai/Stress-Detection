"""
Data Loader and Raw Waveform Slicer for Approach 3 (1D CNN + BiLSTM + Attention).
"""
import os
import numpy as np
import pandas as pd
from config import (
    E4_DATA_DIR,
    ANNOTATIONS_DIR,
    METADATA_FILE,
    SENSORS,
    WINDOW_SIZE_SEC,
    VALID_SUBJECTS,
    ANNOTATION_SUFFIX,
)
from stress_labels import compute_stress_label


def load_metadata():
    """Load metadata/subjects.csv to get debate start times (unix timestamp ms)."""
    df = pd.read_csv(METADATA_FILE, skipinitialspace=True)
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


def normalize_sensor_df(df, columns):
    """
    Apply per-subject Z-score normalization across each channel of a sensor DataFrame.
    Prevents inter-subject baseline differences from hurting cross-subject generalization.
    """
    df_norm = df.copy()
    for col in columns:
        vals = df_norm[col].values
        mean = np.mean(vals)
        std = np.std(vals)
        if std == 0 or np.isnan(std):
            std = 1.0
        df_norm[col] = (vals - mean) / std
    return df_norm


def get_raw_slice(sensor_df, columns, window_start_ms, window_end_ms, expected_length):
    """
    Extract exact-length raw waveform slice for a given time window.
    """
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


def load_all_raw_waveforms():
    """
    Load and process all subjects, returning exact-length waveform matrices suitable for sequence training.
    """
    meta = load_metadata()

    all_acc = []
    all_eda = []
    all_temp = []
    all_hr = []
    all_y = []
    all_scores = []
    all_sids = []

    print(f"Loading raw waveforms across {len(VALID_SUBJECTS)} subjects with {WINDOW_SIZE_SEC}s history windows...")

    for sid_num in VALID_SUBJECTS:
        sid_str = str(sid_num)
        ann_path = os.path.join(ANNOTATIONS_DIR, f"P{sid_num}{ANNOTATION_SUFFIX}")
        if not os.path.exists(ann_path):
            continue

        annotations = pd.read_csv(ann_path)
        start_time = meta.get(sid_num, meta.get(f"P{sid_num}", meta.get(sid_str, None)))
        if start_time is None:
            continue

        # Load and normalize sensor data frames
        sensors = {}
        for s_key, s_info in SENSORS.items():
            fpath = os.path.join(E4_DATA_DIR, sid_str, s_info["file"])
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df = normalize_sensor_df(df, s_info["channels"])
                sensors[s_key] = df
            else:
                sensors[s_key] = None

        if any(v is None for v in sensors.values()):
            continue

        n_win = len(annotations)
        print(f"  P{sid_num:2d}: extracting {n_win} windows...", end="")

        subj_acc, subj_eda, subj_temp, subj_hr = [], [], [], []
        subj_y, subj_scores, subj_sids = [], [], []

        for _, ann_row in annotations.iterrows():
            seconds = int(ann_row["seconds"])
            window_end_ms = start_time + seconds * 1000
            window_start_ms = window_end_ms - WINDOW_SIZE_SEC * 1000

            acc_slice = get_raw_slice(sensors["ACC"], SENSORS["ACC"]["channels"], window_start_ms, window_end_ms, SENSORS["ACC"]["expected_length"])
            acc_slice = acc_slice.T.astype(np.float32)  # (3, 960)

            eda_slice = get_raw_slice(sensors["EDA"], SENSORS["EDA"]["channels"], window_start_ms, window_end_ms, SENSORS["EDA"]["expected_length"])
            eda_slice = eda_slice.T.astype(np.float32)  # (1, 120)

            temp_slice = get_raw_slice(sensors["TEMP"], SENSORS["TEMP"]["channels"], window_start_ms, window_end_ms, SENSORS["TEMP"]["expected_length"])
            temp_slice = temp_slice.T.astype(np.float32)  # (1, 120)

            hr_slice = get_raw_slice(sensors["HR"], SENSORS["HR"]["channels"], window_start_ms, window_end_ms, SENSORS["HR"]["expected_length"])
            hr_slice = hr_slice.T.astype(np.float32)  # (1, 30)

            label, score = compute_stress_label(ann_row)

            subj_acc.append(acc_slice)
            subj_eda.append(eda_slice)
            subj_temp.append(temp_slice)
            subj_hr.append(hr_slice)
            subj_y.append(label)
            subj_scores.append(score)
            subj_sids.append(sid_num)

        all_acc.extend(subj_acc)
        all_eda.extend(subj_eda)
        all_temp.extend(subj_temp)
        all_hr.extend(subj_hr)
        all_y.extend(subj_y)
        all_scores.extend(subj_scores)
        all_sids.extend(subj_sids)

        print(f" [OK]")

    acc_arr = np.array(all_acc, dtype=np.float32)
    eda_arr = np.array(all_eda, dtype=np.float32)
    temp_arr = np.array(all_temp, dtype=np.float32)
    hr_arr = np.array(all_hr, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.int64)
    scores_arr = np.array(all_scores, dtype=np.float32)
    sids_arr = np.array(all_sids, dtype=np.int64)

    print(f"\nRaw Waveform Dataset Loaded:")
    print(f"  ACC shape :  {acc_arr.shape}  (Channels: 3, Length: {acc_arr.shape[2]})")
    print(f"  EDA shape :  {eda_arr.shape}  (Channels: 1, Length: {eda_arr.shape[2]})")
    print(f"  TEMP shape:  {temp_arr.shape} (Channels: 1, Length: {temp_arr.shape[2]})")
    print(f"  HR shape  :  {hr_arr.shape}   (Channels: 1, Length: {hr_arr.shape[2]})")
    print(f"  Total samples: {len(y_arr)} | Stressed: {y_arr.sum()} ({y_arr.sum()/len(y_arr):.1%})")

    return {
        "acc": acc_arr,
        "eda": eda_arr,
        "temp": temp_arr,
        "hr": hr_arr,
        "y": y_arr,
        "scores": scores_arr,
        "subject_ids": sids_arr,
    }
