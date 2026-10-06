"""
Configuration for Approach 2: 1D CNN on Raw Multi-Rate Sensor Waveforms.
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATASET"))

E4_DATA_DIR = os.path.join(DATASET_ROOT, "e4_data")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "emotion_annotations", "aggregated_external_annotations")
ANNOTATION_SUFFIX = ".external.csv"
METADATA_FILE = os.path.join(DATASET_ROOT, "metadata", "metadata", "subjects.csv")

VALID_SUBJECTS = [
    1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32,
]

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ============================================================
# SENSOR MAPPING & RAW WAVEFORM SHAPES
# ============================================================
# We preserve the native sampling rates of each physical hardware sensor using a multi-branch 1D CNN.
# WINDOW_SIZE_SEC: duration of raw waveform history preceding each 5-second annotation.
# 30 seconds captures the physiological lag of EDA and skin temperature responses.
WINDOW_SIZE_SEC = 30
ANNOTATION_STEP_SEC = 5

SENSORS = {
    "ACC": {
        "file": "E4_ACC.csv",
        "rate_hz": 32,
        "channels": ["x", "y", "z"],
        "hardware": "MPU6500",
        "expected_length": 32 * WINDOW_SIZE_SEC,  # 960 samples
    },
    "EDA": {
        "file": "E4_EDA.csv",
        "rate_hz": 4,
        "channels": ["value"],
        "hardware": "GSR",
        "expected_length": 4 * WINDOW_SIZE_SEC,   # 120 samples
    },
    "TEMP": {
        "file": "E4_TEMP.csv",
        "rate_hz": 4,
        "channels": ["value"],
        "hardware": "DS18B20",
        "expected_length": 4 * WINDOW_SIZE_SEC,   # 120 samples
    },
    "HR": {
        "file": "E4_HR.csv",
        "rate_hz": 1,
        "channels": ["value"],
        "hardware": "MAX30102",
        "expected_length": 1 * WINDOW_SIZE_SEC,   # 30 samples
    },
}

# ============================================================
# STRESS LABELING -- Option A: Composite Score
# ============================================================
# same parameters as Phase 1 baseline for fair, apples-to-apples comparison
STRESS_WEIGHTS = {
    "arousal": 0.35,
    "inv_valence": 0.35,   # applied to (6 - valence)
    "nervous": 0.20,
    "angry": 0.10,
}
STRESS_THRESHOLD = 0.35

# ============================================================
# DEEP LEARNING & TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 64
EPOCHS_PER_FOLD = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4

# Use WeightedRandomSampler during training to handle class imbalance dynamically
USE_WEIGHTED_SAMPLER = True
