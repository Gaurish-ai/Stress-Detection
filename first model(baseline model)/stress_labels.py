"""
Stress Label Computation -- Option A: Composite Score.

Derives a single stress score from multiple annotation dimensions:
    score = arousal x 0.35 + (6 - valence) x 0.35 + nervous x 0.20 + angry x 0.10

The raw score ranges from 1.0 (minimum stress) to 5.0 (maximum stress).
Normalized to [0, 1], then thresholded at 0.5 for binary classification.
"""
from config import STRESS_WEIGHTS, STRESS_THRESHOLD


def compute_stress_score(annotation_row):
    """
    Compute normalized composite stress score from an annotation row.

    Args:
        annotation_row: pandas Series with 'arousal', 'valence', 'nervous', 'angry'

    Returns:
        float: Normalized stress score in [0, 1]
    """
    arousal = float(annotation_row["arousal"])
    valence = float(annotation_row["valence"])
    nervous = float(annotation_row["nervous"])
    angry = float(annotation_row["angry"])

    raw_score = (
        arousal * STRESS_WEIGHTS["arousal"]
        + (6.0 - valence) * STRESS_WEIGHTS["inv_valence"]
        + nervous * STRESS_WEIGHTS["nervous"]
        + angry * STRESS_WEIGHTS["angry"]
    )

    # Normalize from [1.0, 5.0] to [0.0, 1.0]
    normalized = (raw_score - 1.0) / 4.0
    return normalized


def compute_stress_label(annotation_row):
    """
    Compute binary stress label and raw score.

    Args:
        annotation_row: pandas Series with annotation dimensions

    Returns:
        tuple: (label: int 0 or 1, score: float in [0, 1])
    """
    score = compute_stress_score(annotation_row)
    label = 1 if score > STRESS_THRESHOLD else 0
    return label, score
