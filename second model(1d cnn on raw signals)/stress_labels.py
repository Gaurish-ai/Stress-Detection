"""
Stress Label Computation -- Option A: Composite Score.

Derives binary stress labels and normalized scores from 4-dimensional external consensus ratings:
    score = arousal x 0.35 + (6 - valence) x 0.35 + nervous x 0.20 + angry x 0.10
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
