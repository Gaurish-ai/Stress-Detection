"""
Stress Label Computation -- Composite Score from Affective & Emotion Annotations.
"""
from __future__ import annotations

STRESS_WEIGHTS = {
    "arousal": 0.35,
    "inv_valence": 0.35,
    "nervous": 0.20,
    "angry": 0.10,
}
STRESS_THRESHOLD = 0.5


def compute_stress_score(annotation_row) -> float:
    arousal = float(annotation_row["arousal"])
    valence = float(annotation_row["valence"])
    nervous = float(annotation_row["nervous"]) if "nervous" in annotation_row else 1.0
    angry = float(annotation_row["angry"]) if "angry" in annotation_row else 1.0

    raw_score = (
        arousal * STRESS_WEIGHTS["arousal"]
        + (6.0 - valence) * STRESS_WEIGHTS["inv_valence"]
        + nervous * STRESS_WEIGHTS["nervous"]
        + angry * STRESS_WEIGHTS["angry"]
    )

    normalized = (raw_score - 1.0) / 4.0
    return float(normalized)


def compute_stress_label(annotation_row) -> tuple[int, float]:
    score = compute_stress_score(annotation_row)
    label = 1 if score > STRESS_THRESHOLD else 0
    return label, score
