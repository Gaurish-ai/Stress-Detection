# Approach 5: Facial & Video Specialists (`fifth model(facial video specialists)/`)
## Pure Visual Behavioral & Facial Dynamics for Stress Detection

This folder contains our **Fifth Model Package: Facial & Video Specialists (`Model 5`)**. Before fusing video features with our E4 physiological sensors (`Model 4 TF-Transformer`), we perform a rigorous, isolated exploration of **pure facial and video dynamics** across our **18 common participants** (`P04, P05, P08, P09, P10, P13, P15, P19, P20, P21, P22, P23, P24, P25, P26, P29, P30, P31`) who have synchronized debate video recordings.

---

## 1. Why Isolate Video Features First?

In multimodal representation learning, **one-shot end-to-end concatenation often suffers from modality collapse**. If physiological sensor streams (`EDA, BVP`) converge faster or carry larger initial gradient updates, the neural network ignores subtle visual dynamics. 

By building dedicated **Facial Specialists (`Model 5`)** and evaluating them with strict **Leave-One-Subject-Out (LOSO)** cross-validation on our 18 common subjects:
1. We empirically benchmark the exact standalone stress prediction accuracy of pure visual behavior (`P(Stress | Video Alone)`).
2. We identify which temporal modeling architecture (Tabular Trees vs. LSTMs vs. Self-Attention) best captures human facial expressions, posture, and gaze shifts over time.
3. We extract clean, pre-trained visual embeddings that can be fused with our physiological `TF-Transformer` in Stage 2 (`Model 6: Multimodal Fusion`) without gradient interference.

---

## 2. The 11 Continuous Video Features

We utilize the 11 mathematical visual/facial features extracted from each 5-second window using OpenCV & Haar Cascades (cached in `multimodal_stress_pipeline/outputs_smoke/video_features_self_annotations_2frames.csv`):

| Feature Category | Features Included | What It Captures Biologically & Behaviorally |
| :--- | :--- | :--- |
| **Facial Presence & Posture** | `face_count_mean`, `face_count_max`, `face_area_ratio_mean`, `face_area_ratio_max` | Leaning forward/backward toward the screen, postural rigidity, and head tracking stability. |
| **Gaze & Head Orientation** | `face_center_x_mean`, `face_center_y_mean` | Gaze direction, head rotation, nodding, and nervous posture shifts during debates. |
| **Micro-Tension & Lighting** | `brightness_mean`, `brightness_std`, `contrast_mean`, `sharpness_mean` | Localized pixel texture variance (`Laplacian variance`), lighting changes from micro-movements, and facial tension. |
| **Behavioral Motion** | `motion_mean` | Frame-to-frame absolute pixel difference across the window (`cv2.absdiff`) separating vigorous speaking/gesturing from sedentary tension. |

---

## 3. The 3 Facial Specialist Architectures Explored Head-to-Head

We implement and evaluate **3 distinct modeling approaches** operating across a **60-second rolling circular buffer (`12 consecutive 5-second windows`)**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │     Rolling 60-Second Video Buffer (12 Windows)        │
                       │   (12 x 11 continuous visual features = 132 values)    │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
            ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
            │        MODEL 5A:         │  │        MODEL 5B:         │  │        MODEL 5C:         │
            │   Statistical XGBoost    │  │  BiLSTM + Attention      │  │  Temporal Transformer    │
            └────────────┬─────────────┘  └────────────┬─────────────┘  └────────────┬─────────────┘
                         │ • Extracts 77 statistical   │ • Processes 12x11 sequence │ • Projects 11d to 64d token
                         │   summary features across   │   via Bidirectional LSTM   │ • 2-Layer Multi-Head Self-
                         │   60s (mean, std, range,    │ • Temporal attention maps  │   Attention Encoder across
                         │   slope, delta, min, max)   │   historical trajectory    │   60s historical context
                         ▼                         ▼                         ▼
            ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
            │  Tabular Classification  │  │   Attention Pool Logit   │  │  Target-Window Logit     │
            │    (XGBoost + SMOTE)     │  │       (PyTorch)          │  │       (PyTorch)          │
            └──────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘
```

### Model 5A: Handcrafted Statistical XGBoost (`Tabular Baseline`)
- **Input:** For each 60-second rolling sequence (`12 windows x 11 features`), we compute 7 statistical summary features per feature (`mean`, `std`, `min`, `max`, `range`, `slope/trend`, and `delta` between latest window $t$ and first window $t-11$). Total: $11 \times 7 = 77$ handcrafted features.
- **Why:** Gradient Boosted Trees excel at sharp, non-linear tabular cutoffs (e.g., classifying stress when `motion_mean` drops below a threshold while `sharpness_std` spikes due to facial rigidity).

### Model 5B: Temporal BiLSTM with Attention (`Sequential Behavioral Trajectory`)
- **Input:** The raw sequential matrix of 12 consecutive windows (`Shape: [Batch, Sequence=12, Features=11]`).
- **Why:** Human expressions and posture evolve continuously during stress. A Bidirectional LSTM with temporal attention tracks behavioral trajectories over time (such as a subject slowly leaning closer while reducing head nodding across a 1-minute horizon).

### Model 5C: Multi-Head Facial Video Transformer (`Self-Attention over Visual Dynamics`)
- **Input:** The 12 consecutive windows projected to 64-dimensional embeddings (`Shape: [Batch, Sequence=12, Embed=64]`) plus positional encoding.
- **Why:** Multi-Head Self-Attention correlates non-contiguous windows across the 60-second horizon. It distinguishes whether a sudden motion spike at $t-5$ was just an isolated hand gesture (when surrounding windows show calm posture) or the beginning of sustained agitation.

---

## 4. Evaluation & Next Steps

All 3 models are evaluated simultaneously under **Leave-One-Subject-Out (LOSO) cross-validation across our 18 common subjects**. 
Once complete, `train.py` outputs a direct comparison (`results.json`, `video_models_comparison.png`), identifying the winning visual architecture and generating pre-trained embeddings for Stage 2 (`Model 6: Dual-Stream Multimodal Fusion`).
