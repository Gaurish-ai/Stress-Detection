# Sensor-Based Stress Detection — Implementation Plan

## Project Goal
Build a stress detection system using physiological sensor data from Empatica E4 wristband,
mapped to physical sensors: MAX30102 (HR), DS18B20 (TEMP), GSR (EDA), MPU6500 (ACC).

## Dataset
- **28 subjects** with E4 wristband data (debate recordings, ~10-15 min each)
- **4 sensors used**: EDA (4Hz), HR (1Hz), ACC-xyz (32Hz), TEMP (4Hz)
- **Annotations**: Every 5 seconds — arousal, valence, nervous, angry, etc. (1-5 scale)
- **Annotation source**: Aggregated external annotations (consensus of 5 raters)

## Stress Label: Composite Score (Option A)
```
stress_score = arousal × 0.35 + (6 - valence) × 0.35 + nervous × 0.20 + angry × 0.10
normalized = (score - 1) / 4    →  range [0, 1]
stress = 1 if normalized > 0.5 else 0
```

## Physical Sensor Mapping
| Hardware Sensor | Dataset Signal | Sampling Rate |
|----------------|---------------|---------------|
| GSR            | EDA           | 4 Hz          |
| MAX30102       | HR            | 1 Hz          |
| MPU6500        | ACC (x,y,z)   | 32 Hz         |
| DS18B20        | TEMP          | 4 Hz          |

## Model Approach: Phase 1 — Baseline (XGBoost)
1. Load sensor data + annotations
2. Align to 5-second windows using timestamps from subjects.csv
3. Extract handcrafted features per window (34 features total):
   - EDA: mean, std, min, max, range, slope, peaks, AUC, kurtosis, skewness
   - HR: mean, std, min, max, range, delta
   - ACC: per-axis mean/std, magnitude stats, jerk, SMA
   - TEMP: mean, std, min, max, slope, range
4. Train XGBoost with Leave-One-Subject-Out (LOSO) cross-validation
5. Evaluate: accuracy, F1, precision, recall, ROC-AUC

## Future Phases
- Phase 2: Deep Learning (CNN + BiLSTM)
- Phase 3: Multimodal (sensor + facial video fusion)
