"""
Stress Label Computation -- Option A: Composite Score.
"""
from config import STRESS_WEIGHTS, STRESS_THRESHOLD


def compute_stress_score(annotation_row):
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

    normalized = (raw_score - 1.0) / 4.0
    return normalized


def compute_stress_label(annotation_row):
    score = compute_stress_score(annotation_row)
    label = 1 if score > STRESS_THRESHOLD else 0
    return label, score
