"""
Feature Engineering -- Extract handcrafted physiological features per window.

For each 5-second window aligned to an annotation timestamp, we extract
statistical and signal-processing features from 4 sensor modalities.

Total features: 34
    EDA  (GSR)       : 10 features
    HR   (MAX30102)  :  6 features
    ACC  (MPU6500)   : 12 features
    TEMP (DS18B20)   :  6 features
"""
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

from config import SENSORS, WINDOW_SIZE_SEC


# ============================================================
# EDA (GSR) Features -- 10 features
# ============================================================
_EDA_KEYS = [
    "eda_mean", "eda_std", "eda_min", "eda_max", "eda_range",
    "eda_slope", "eda_num_peaks", "eda_auc", "eda_kurtosis", "eda_skewness",
]


def extract_eda_features(values):
    """
    Extract EDA features from a 5-second window (~20 samples at 4 Hz).

    Features:
        mean, std, min, max, range  -- basic statistics
        slope                       -- linear trend (stress causes rising EDA)
        num_peaks                   -- SCR count (phasic responses)
        auc                         -- area under curve (total activation)
        kurtosis, skewness          -- distribution shape
    """
    if len(values) < 2:
        return {k: 0.0 for k in _EDA_KEYS}

    features = {
        "eda_mean": np.mean(values),
        "eda_std": np.std(values),
        "eda_min": np.min(values),
        "eda_max": np.max(values),
        "eda_range": np.ptp(values),
    }

    # Linear trend
    slope, *_ = stats.linregress(np.arange(len(values)), values)
    features["eda_slope"] = slope

    # Skin conductance response peaks
    peaks, _ = find_peaks(values, distance=2) if len(values) >= 3 else ([], {})
    features["eda_num_peaks"] = float(len(peaks))

    # Area under curve
    features["eda_auc"] = float(np.trapz(values))

    # Higher-order stats
    features["eda_kurtosis"] = float(stats.kurtosis(values)) if len(values) > 3 else 0.0
    features["eda_skewness"] = float(stats.skew(values)) if len(values) > 2 else 0.0

    return features


# ============================================================
# HR (MAX30102) Features -- 6 features
# ============================================================
_HR_KEYS = ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_range", "hr_delta"]


def extract_hr_features(values):
    """
    Extract HR features from a 5-second window (~5 samples at 1 Hz).

    Features:
        mean, std, min, max, range -- basic statistics
        delta                      -- last - first (HR change direction)
    """
    if len(values) == 0:
        return {k: 0.0 for k in _HR_KEYS}

    return {
        "hr_mean": np.mean(values),
        "hr_std": np.std(values),
        "hr_min": np.min(values),
        "hr_max": np.max(values),
        "hr_range": np.ptp(values),
        "hr_delta": float(values[-1] - values[0]) if len(values) > 1 else 0.0,
    }


# ============================================================
# ACC (MPU6500) Features -- 12 features
# ============================================================
_ACC_KEYS = [
    "acc_x_mean", "acc_x_std", "acc_y_mean", "acc_y_std",
    "acc_z_mean", "acc_z_std",
    "acc_mag_mean", "acc_mag_std", "acc_mag_range", "acc_mag_max",
    "acc_jerk_mean", "acc_sma",
]


def extract_acc_features(x_vals, y_vals, z_vals):
    """
    Extract accelerometer features from a 5-second window (~160 samples at 32 Hz).

    Features:
        Per-axis mean, std          -- activity level per direction
        Magnitude mean, std, range  -- overall motion intensity
        mag_max                     -- peak motion
        jerk_mean                   -- mean absolute derivative of magnitude (abruptness)
        sma                        -- Signal Magnitude Area (total activity)
    """
    if len(x_vals) == 0:
        return {k: 0.0 for k in _ACC_KEYS}

    magnitude = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)

    features = {
        "acc_x_mean": np.mean(x_vals),
        "acc_x_std": np.std(x_vals),
        "acc_y_mean": np.mean(y_vals),
        "acc_y_std": np.std(y_vals),
        "acc_z_mean": np.mean(z_vals),
        "acc_z_std": np.std(z_vals),
        "acc_mag_mean": np.mean(magnitude),
        "acc_mag_std": np.std(magnitude),
        "acc_mag_range": np.ptp(magnitude),
        "acc_mag_max": np.max(magnitude),
    }

    # Jerk -- abruptness of motion
    if len(magnitude) > 1:
        features["acc_jerk_mean"] = float(np.mean(np.abs(np.diff(magnitude))))
    else:
        features["acc_jerk_mean"] = 0.0

    # Signal Magnitude Area
    features["acc_sma"] = float(
        (np.sum(np.abs(x_vals)) + np.sum(np.abs(y_vals)) + np.sum(np.abs(z_vals)))
        / len(x_vals)
    )

    return features


# ============================================================
# TEMP (DS18B20) Features -- 6 features
# ============================================================
_TEMP_KEYS = ["temp_mean", "temp_std", "temp_min", "temp_max", "temp_slope", "temp_range"]


def extract_temp_features(values):
    """
    Extract skin temperature features from a 5-second window (~20 samples at 4 Hz).

    Features:
        mean, std, min, max -- basic statistics
        slope               -- warming/cooling trend (stress -> vasoconstriction -> cooling)
        range               -- temperature variability
    """
    if len(values) < 2:
        return {k: 0.0 for k in _TEMP_KEYS}

    features = {
        "temp_mean": np.mean(values),
        "temp_std": np.std(values),
        "temp_min": np.min(values),
        "temp_max": np.max(values),
        "temp_range": np.ptp(values),
    }

    slope, *_ = stats.linregress(np.arange(len(values)), values)
    features["temp_slope"] = slope

    return features


# ============================================================
# Combined Feature Extraction
# ============================================================
def extract_window_features(sensor_windows):
    """
    Extract all features from all sensors for a single 5-second window.

    Args:
        sensor_windows: dict of {sensor_name: DataFrame of samples in window}

    Returns:
        dict of {feature_name: float}  (34 features total)
    """
    features = {}

    # --- EDA ---
    eda_df = sensor_windows.get("EDA")
    if eda_df is not None and len(eda_df) > 0:
        features.update(extract_eda_features(eda_df["value"].values.astype(float)))
    else:
        features.update({k: 0.0 for k in _EDA_KEYS})

    # --- HR ---
    hr_df = sensor_windows.get("HR")
    if hr_df is not None and len(hr_df) > 0:
        features.update(extract_hr_features(hr_df["value"].values.astype(float)))
    else:
        features.update({k: 0.0 for k in _HR_KEYS})

    # --- ACC ---
    acc_df = sensor_windows.get("ACC")
    if acc_df is not None and len(acc_df) > 0:
        features.update(
            extract_acc_features(
                acc_df["x"].values.astype(float),
                acc_df["y"].values.astype(float),
                acc_df["z"].values.astype(float),
            )
        )
    else:
        features.update({k: 0.0 for k in _ACC_KEYS})

    # --- TEMP ---
    temp_df = sensor_windows.get("TEMP")
    if temp_df is not None and len(temp_df) > 0:
        features.update(extract_temp_features(temp_df["value"].values.astype(float)))
    else:
        features.update({k: 0.0 for k in _TEMP_KEYS})

    return features


def build_feature_matrix(all_subject_data):
    """
    Build the complete feature matrix and label vector from all subjects.

    For each annotation row (every 5 seconds), we:
        1. Compute the time window [startTime + (sec-5)*1000, startTime + sec*1000)
        2. Extract sensor samples in that window
        3. Compute 34 handcrafted features
        4. Add 34 temporal delta features (change from previous window)
        5. Compute composite stress label

    Total features: 68 (34 base + 34 delta)

    Args:
        all_subject_data: dict of {subject_id: subject_data_dict}

    Returns:
        X             -- np.ndarray (n_samples, 68)
        y             -- np.ndarray (n_samples,) binary stress labels
        subject_ids   -- np.ndarray (n_samples,) subject ID per sample
        feature_names -- list of str
        stress_scores -- np.ndarray (n_samples,) raw 0-1 scores
    """
    from stress_labels import compute_stress_label
    from data_loader import get_sensor_window

    all_features = []
    all_labels = []
    all_scores = []
    all_sids = []

    for sid, data in all_subject_data.items():
        annotations = data["annotations"]
        sensors = data["sensors"]
        start_time = data["start_time"]

        n_windows = len(annotations)
        print(f"    P{sid}: {n_windows} windows", end="")

        # Collect features for this subject first (for temporal deltas)
        subject_features = []
        subject_labels = []
        subject_scores = []

        for _, ann_row in annotations.iterrows():
            seconds = int(ann_row["seconds"])

            # Annotation at second N covers the window [N-5, N) relative to debate start
            window_end_ms = start_time + seconds * 1000
            window_start_ms = window_end_ms - WINDOW_SIZE_SEC * 1000

            # Extract sensor samples in this window
            sensor_windows = {}
            for sensor_name, sensor_df in sensors.items():
                if sensor_df is not None:
                    sensor_windows[sensor_name] = get_sensor_window(
                        sensor_df, window_start_ms, window_end_ms
                    )
                else:
                    sensor_windows[sensor_name] = None

            # Extract base features (34)
            features = extract_window_features(sensor_windows)
            subject_features.append(features)

            # Compute stress label
            label, score = compute_stress_label(ann_row)
            subject_labels.append(label)
            subject_scores.append(score)

        # Add temporal delta features for this subject
        base_keys = list(subject_features[0].keys())
        for i in range(len(subject_features)):
            if i == 0:
                # First window: delta = 0
                for k in base_keys:
                    subject_features[i][f"delta_{k}"] = 0.0
            else:
                # Delta = current - previous window
                for k in base_keys:
                    subject_features[i][f"delta_{k}"] = (
                        subject_features[i][k] - subject_features[i - 1][k]
                    )

        # Append to global lists
        all_features.extend(subject_features)
        all_labels.extend(subject_labels)
        all_scores.extend(subject_scores)
        all_sids.extend([sid] * n_windows)

        print(f"  ->  {n_windows} extracted")

    # Convert to numpy arrays
    feature_names = list(all_features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in all_features], dtype=np.float64)
    y = np.array(all_labels, dtype=np.int32)
    subject_ids = np.array(all_sids, dtype=np.int32)
    stress_scores = np.array(all_scores, dtype=np.float64)

    n_stressed = int(y.sum())
    n_total = len(y)
    print(f"\n  Feature matrix shape : {X.shape}")
    print(f"  Total windows        : {n_total}")
    print(f"  Stressed             : {n_stressed} ({n_stressed/n_total:.1%})")
    print(f"  Not-stressed         : {n_total - n_stressed} ({(n_total-n_stressed)/n_total:.1%})")

    return X, y, subject_ids, feature_names, stress_scores


def normalize_per_subject(X, subject_ids):
    """
    Z-score normalize features within each subject independently.

    This removes inter-subject baseline differences (e.g., person A has
    baseline HR=65 while person B has HR=90). After normalization, features
    represent relative deviations from each individual's own baseline.

    Args:
        X: (n_samples, n_features) feature matrix
        subject_ids: (n_samples,) subject ID per sample

    Returns:
        X_normalized: (n_samples, n_features) normalized feature matrix
    """
    X_norm = X.copy()
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        subj_data = X_norm[mask]
        mean = subj_data.mean(axis=0)
        std = subj_data.std(axis=0)
        std[std == 0] = 1.0  # prevent division by zero for constant features
        X_norm[mask] = (subj_data - mean) / std
    return X_norm

