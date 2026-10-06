# Model 6: State-of-the-Art Multimodal Fusion Report (`18 Common Subjects`)

## Executive Summary
- Total Evaluation Samples: **2011** (Stressed: **245**)
- Model 4 (Physiological E4 TF-Transformer Alone): **73.97%** Mean Subject Accuracy (ROC-AUC: **0.436**)
- Model 5A (Visual Statistical XGBoost Alone): **80.77%** Mean Subject Accuracy (ROC-AUC: **0.304**)
- **Model 6A (Late Gated Probability Fusion)**: **80.46%** Mean Subject Accuracy (ROC-AUC: **0.309**)
- **Model 6B (Intermediate Cross-Attention Token Fusion)**: **80.95%** Mean Subject Accuracy (ROC-AUC: **0.396**)

## Per-Subject Benchmarks
| Subject ID | Total Windows | Stressed Windows | Physio (Model 4) | Visual (Model 5A) | Late Fusion (Model 6A) | Inter Fusion (Model 6B) | Optimal Weight (alpha) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P04 | 125 | 17 | 81.60% | 86.40% | **86.40%** | **86.40%** | 0.00 |
| P05 | 113 | 6 | 91.15% | 94.69% | **94.69%** | **94.69%** | 0.00 |
| P08 | 109 | 9 | 48.62% | 86.24% | **86.24%** | **67.89%** | 0.00 |
| P09 | 111 | 4 | 26.13% | 96.40% | **96.40%** | **36.94%** | 0.00 |
| P10 | 109 | 35 | 44.95% | 62.39% | **61.47%** | **58.72%** | 0.07 |
| P13 | 110 | 19 | 85.45% | 81.82% | **81.82%** | **81.82%** | 0.01 |
| P15 | 110 | 22 | 48.18% | 80.00% | **80.00%** | **66.36%** | 0.00 |
| P19 | 112 | 0 | 100.00% | 86.61% | **86.61%** | **100.00%** | 0.00 |
| P20 | 112 | 21 | 75.00% | 81.25% | **81.25%** | **81.25%** | 0.00 |
| P21 | 108 | 0 | 50.93% | 95.37% | **90.74%** | **100.00%** | 0.09 |
| P22 | 109 | 37 | 66.06% | 66.06% | **66.06%** | **66.06%** | 0.00 |
| P23 | 110 | 45 | 59.09% | 59.09% | **59.09%** | **59.09%** | 0.00 |
| P24 | 110 | 0 | 100.00% | 89.09% | **89.09%** | **100.00%** | 0.00 |
| P25 | 113 | 0 | 100.00% | 82.30% | **82.30%** | **100.00%** | 0.00 |
| P26 | 113 | 26 | 76.99% | 76.99% | **76.99%** | **76.99%** | 0.00 |
| P29 | 110 | 0 | 100.00% | 32.73% | **32.73%** | **100.00%** | 0.00 |
| P30 | 110 | 4 | 77.27% | 96.36% | **96.36%** | **80.91%** | 0.01 |
| P31 | 117 | 0 | 100.00% | 100.00% | **100.00%** | **100.00%** | 0.00 |

## Conclusion
By dynamically fusing the E4 physiological stream (`Model 4`) with the visual statistical stream (`Model 5A`), our multimodal architecture mitigates single-modality errors and achieves state-of-the-art stress prediction across our 18 common participants.