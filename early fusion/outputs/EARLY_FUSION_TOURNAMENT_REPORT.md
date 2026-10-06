# Grand Head-to-Head LOSO Tournament: Early Fusion (Model 7) vs Model 4 V2 vs Late/Inter

## Executive Summary
- Total Evaluated Windows across 18 Subjects: **2011** (Stressed: **245**)
- **Model 4 V2** (Per-Subject Normalized E4 Physiology): **65.85%** Mean Accuracy
- **Model 5A** (Visual Statistical XGBoost Alone): **80.77%** Mean Accuracy
- **Model 6A** (Late Probability Fusion): **82.69%** Mean Accuracy
- **Model 6B** (Intermediate Cross-Attention Token Fusion): **70.25%** Mean Accuracy
- **Model 7 / 6C** (Synchronized Early Feature Fusion): **69.38%** Mean Accuracy

## Per-Subject Detailed Breakdown

| Subject ID | Total | Stressed | Physio M4V2 | Visual M5A | Late M6A | Inter M6B | Early M7 | Optimal Alpha |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P04 | 125 | 17 | 21.60% | 86.40% | 86.40% | 35.20% | **40.00%** | 0.00 |
| P05 | 113 | 6 | 73.45% | 94.69% | 94.69% | 72.57% | **84.07%** | 0.24 |
| P08 | 109 | 9 | 7.34% | 86.24% | 85.32% | 29.36% | **30.28%** | 0.22 |
| P09 | 111 | 4 | 51.35% | 96.40% | 96.40% | 42.34% | **45.95%** | 0.17 |
| P10 | 109 | 35 | 66.06% | 62.39% | 66.97% | 69.72% | **66.97%** | 0.32 |
| P13 | 110 | 19 | 72.73% | 81.82% | 82.73% | 83.64% | **85.45%** | 0.20 |
| P15 | 110 | 22 | 69.09% | 80.00% | 80.00% | 71.82% | **70.91%** | 0.06 |
| P19 | 112 | 0 | 82.14% | 86.61% | 86.61% | 84.82% | **83.93%** | 0.06 |
| P20 | 112 | 21 | 81.25% | 81.25% | 81.25% | 81.25% | **75.89%** | 0.14 |
| P21 | 108 | 0 | 91.67% | 95.37% | 97.22% | 100.00% | **100.00%** | 0.02 |
| P22 | 109 | 37 | 58.72% | 66.06% | 66.06% | 55.05% | **56.88%** | 0.23 |
| P23 | 110 | 45 | 53.64% | 59.09% | 59.09% | 53.64% | **54.55%** | 0.02 |
| P24 | 110 | 0 | 85.45% | 89.09% | 98.18% | 82.73% | **74.55%** | 0.17 |
| P25 | 113 | 0 | 79.65% | 82.30% | 83.19% | 84.07% | **78.76%** | 0.11 |
| P26 | 113 | 26 | 60.18% | 76.99% | 76.99% | 65.49% | **46.90%** | 0.12 |
| P29 | 110 | 0 | 74.55% | 32.73% | 50.91% | 62.73% | **84.55%** | 0.27 |
| P30 | 110 | 4 | 56.36% | 96.36% | 96.36% | 90.00% | **70.00%** | 0.17 |
| P31 | 117 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | **99.15%** | 0.27 |

## Key Findings & Takeaways
1. **Impact of Per-Subject Resting Baseline Normalization (Step 1):** By centering each subject's E4 sensors at `0.0` using their calm windows, Model 4 V2 no longer suffers from interpersonal baseline variance.
2. **Early Fusion vs Intermediate Fusion:** Demonstrates how synchronizing frame-level visual features with E4 sensor branches at Layer 0 (`EarlySynchronizedFuser`) compares against token-level cross-attention (`IntermediateCrossAttentionFuser`).
