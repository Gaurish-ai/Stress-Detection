"""
Configuration for the Baseline Stress Detection Model.
All hyperparameters, paths, and sensor settings in one place.

Physical Sensor Mapping:
    GSR       ->  EDA  (Electrodermal Activity)
    MAX30102  ->  HR   (Heart Rate)
    MPU6500   ->  ACC  (Accelerometer x, y, z)
    DS18B20   ->  TEMP (Skin Temperature)
"""
import os

# ============================================================
# PATHS
# ============================================================
# Navigate from this file up to the project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # d:\Sensor Stress

DATASET_DIR = os.path.join(_PROJECT_ROOT, "DATASET")
E4_DATA_DIR = os.path.join(DATASET_DIR, "e4_data")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "emotion_annotations")
METADATA_DIR = os.path.join(DATASET_DIR, "metadata", "metadata")
OUTPUT_DIR = os.path.join(_THIS_DIR, "outputs")

# ============================================================
# ANNOTATION SOURCE
# ============================================================
ANNOTATION_SOURCE = "aggregated_external_annotations"
ANNOTATION_SUFFIX = ".external.csv"

# ============================================================
# SENSOR CONFIGURATION
# ============================================================
# Only sensors that map to physical hardware:
#   GSR -> EDA | MAX30102 -> HR | MPU6500 -> ACC | DS18B20 -> TEMP
SENSORS = {
    "EDA": {
        "file": "E4_EDA.csv",
        "rate_hz": 4,
        "columns": ["value"],
        "hardware": "GSR",
    },
    "HR": {
        "file": "E4_HR.csv",
        "rate_hz": 1,
        "columns": ["value"],
        "hardware": "MAX30102",
    },
    "ACC": {
        "file": "E4_ACC.csv",
        "rate_hz": 32,
        "columns": ["x", "y", "z"],
        "hardware": "MPU6500",
    },
    "TEMP": {
        "file": "E4_TEMP.csv",
        "rate_hz": 4,
        "columns": ["value"],
        "hardware": "DS18B20",
    },
}

# ============================================================
# WINDOWING
# ============================================================
WINDOW_SIZE_SEC = 5  # Matches annotation resolution (every 5 seconds)

# ============================================================
# STRESS SCORE -- Option A: Composite Score
# ============================================================
# stress = arousal * 0.35 + (6 - valence) * 0.35 + nervous * 0.20 + angry * 0.10
# Raw range: [1.0, 5.0]  ->  Normalized: [0.0, 1.0]
STRESS_WEIGHTS = {
    "arousal": 0.35,
    "inv_valence": 0.35,   # applied to (6 - valence)
    "nervous": 0.20,
    "angry": 0.10,
}

# --- Binary classification ---
# Threshold 0.35 gives ~18% stressed (median-split), much more balanced than 0.5 (2.4%)
STRESS_THRESHOLD = 0.35

# --- 3-class classification ---
# Low: score <= 0.2625  |  Medium: 0.2625 < score <= 0.40  |  High: score > 0.40
STRESS_THRESHOLDS_3CLASS = (0.2625, 0.40)
CLASS_NAMES_3 = ["Low Stress", "Medium Stress", "High Stress"]

# --- Mode ---
# "binary" or "3class"
CLASSIFICATION_MODE = "binary"

# ============================================================
# XGBOOST HYPERPARAMETERS
# ============================================================
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# ============================================================
# SUBJECTS WITH VALID E4 DATA
# ============================================================
# P2, P3, P6, P7 have no E4 wristband data
VALID_SUBJECTS = [
    1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32,
]
