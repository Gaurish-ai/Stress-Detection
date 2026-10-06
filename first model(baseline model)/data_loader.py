"""
Data Loader -- Load and align E4 sensor data with emotion annotations.

Handles:
    - Loading subjects.csv for timing metadata (startTime, endTime)
    - Loading sensor CSVs (EDA, HR, ACC, TEMP)
    - Loading annotation CSVs (aggregated external)
    - Temporal alignment of sensor windows to annotation timestamps
"""
import os
import pandas as pd
import numpy as np

from config import (
    E4_DATA_DIR, ANNOTATIONS_DIR, METADATA_DIR,
    SENSORS, ANNOTATION_SOURCE, ANNOTATION_SUFFIX,
    VALID_SUBJECTS,
)


def load_subjects_metadata():
    """
    Load subjects.csv with debate timing information.

    Returns:
        DataFrame with columns: pid, initTime, startTime, endTime
    """
    path = os.path.join(METADATA_DIR, "subjects.csv")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def load_annotation(subject_id):
    """
    Load annotation file for a given subject.

    Args:
        subject_id: int, participant ID

    Returns:
        DataFrame with annotation columns (seconds, arousal, valence, etc.)
    """
    filename = f"P{subject_id}{ANNOTATION_SUFFIX}"
    path = os.path.join(ANNOTATIONS_DIR, ANNOTATION_SOURCE, filename)
    df = pd.read_csv(path)
    return df


def load_sensor(subject_id, sensor_name):
    """
    Load a single sensor CSV for a subject.

    Args:
        subject_id: int, participant ID
        sensor_name: str, one of 'EDA', 'HR', 'ACC', 'TEMP'

    Returns:
        DataFrame with sensor data (timestamp, value columns, metadata)
    """
    sensor_cfg = SENSORS[sensor_name]
    path = os.path.join(E4_DATA_DIR, str(subject_id), sensor_cfg["file"])
    df = pd.read_csv(path)
    return df


def get_sensor_window(sensor_df, window_start_ms, window_end_ms):
    """
    Extract sensor samples that fall within a time window.

    Args:
        sensor_df: DataFrame with 'timestamp' column (Unix ms)
        window_start_ms: float, window start in Unix milliseconds
        window_end_ms: float, window end in Unix milliseconds

    Returns:
        DataFrame subset of rows within [window_start_ms, window_end_ms)
    """
    mask = (sensor_df["timestamp"] >= window_start_ms) & (
        sensor_df["timestamp"] < window_end_ms
    )
    return sensor_df.loc[mask]


def load_subject_data(subject_id, subjects_meta):
    """
    Load all sensor data and annotations for one subject.

    Args:
        subject_id: int
        subjects_meta: DataFrame from load_subjects_metadata()

    Returns:
        dict with keys:
            subject_id  -- int
            annotations -- DataFrame
            sensors     -- dict of {sensor_name: DataFrame or None}
            start_time  -- int (Unix ms, debate start)
            end_time    -- int (Unix ms, debate end)
    """
    subj_row = subjects_meta[subjects_meta["pid"] == subject_id].iloc[0]
    start_time = int(subj_row["startTime"])
    end_time = int(subj_row["endTime"])

    annotations = load_annotation(subject_id)

    sensors = {}
    for sensor_name in SENSORS:
        try:
            df = load_sensor(subject_id, sensor_name)
            sensors[sensor_name] = df
        except FileNotFoundError:
            print(f"  [WARN] {sensor_name} not found for subject {subject_id}")
            sensors[sensor_name] = None

    return {
        "subject_id": subject_id,
        "annotations": annotations,
        "sensors": sensors,
        "start_time": start_time,
        "end_time": end_time,
    }


def load_all_subjects():
    """
    Load sensor data and annotations for all valid subjects.

    Returns:
        dict of {subject_id: subject_data_dict}
    """
    subjects_meta = load_subjects_metadata()
    all_data = {}

    for sid in VALID_SUBJECTS:
        print(f"  Loading subject P{sid}...")
        all_data[sid] = load_subject_data(sid, subjects_meta)

    print(f"\n  [OK] Loaded {len(all_data)} subjects successfully.")
    return all_data
