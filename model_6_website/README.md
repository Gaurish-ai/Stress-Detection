# AuraSense AI 2.0 | Model 6 Multimodal Stress Telemetry Dashboard

A state-of-the-art, cybernetic glassmorphic web application specifically designed for **Approach 6: Multimodal Fusion (`Model 6A` & `Model 6B`)**. This dashboard unites **real-time ESP32 E4 physiological telemetry** with **continuous 11-feature facial/video dynamics** and delivers an official, definitive diagnostic assessment in **exactly 60 seconds**.

---

## 🏛️ Folder Structure

```text
model_6_website/
├── index.html       # Multimodal Dashboard (Physio Charts + Live Video HUD + 60s Timer Hub)
├── index.css        # Design system and responsive layout
├── app.js           # BLE GATT, real-time waveforms, webcam extractor, Model 6A/6B scoring
└── README.md        # Documentation & setup guide
```

---

## 🌟 Key Capabilities & Features

### 1. ⏱️ The 60-Second Official Stress Assessment Routine
* Click **🚀 Start 60s Stress Diagnostic** to initiate the official evaluation countdown (`0s` to `60s`).
* During the countdown, the system fills the **rolling 12-window circular buffer** (`12 × 5-second windows = 60s temporal context`).
* At exactly **60 seconds**, the circular progress gauge locks in and pops open the **Official Diagnostic Reveal Modal**, displaying:
  * **Overall Classification:** `CALM BASELINE`, `ELEVATED COGNITIVE AROUSAL`, or `ACUTE COGNITIVE STRESS`.
  * **Physiological Stream (Model 4):** Rolling Z-score and HRV contribution score (`P_phys`).
  * **Visual Stream (Model 5A):** 11-feature summary tree probability (`P_vis`).
  * **Model 6A & 6B Scores:** Precise late gating probability and intermediate cross-attention output.

### 2. 📡 Stream 1: E4 Physiological Telemetry (BLE + Simulator)
* Real-time scrolling canvas charts rendering at **60 frames per second** with auto-scaling bounding boxes:
  * **Electrodermal Activity (`EDA / GSR`)** in µS with live Z-score (`(x - µ_rest) / σ_rest`).
  * **Blood Volume Pulse (`PPG / BVP`)** with real-time `HR` (BPM) and `HRV SDNN` calculation.
  * **Peripheral Skin Temperature (`TEMP`)** in °C.
  * **3-Axis Accelerometer ($X, Y, Z$)** with **Somatic Agitation Filtering** to separate physical exercise from emotional stress.

### 3. 👁️ Stream 2: Facial & Video Dynamics (Model 5 Specialist)
* **Live Webcam Mode (`MediaDevices.getUserMedia`):** Extracts continuous visual metrics directly from your camera frame-by-frame:
  * **Laplacian Sharpness (`sharpness_mean`):** Calculates 3x3 Laplacian variance across the facial region to detect facial rigidity and tension.
  * **Behavioral Motion (`motion_mean`):** Calculates frame-to-frame pixel absolute differences (`cv2.absdiff` equivalent) to track physical agitation.
  * **Postural Leaning (`face_area_ratio`):** Tracks forward/backward leaning relative to the screen during intense tasks.
* **Debate Video Simulator Mode:** Generates realistic synthetic debate trajectories matching the exact distribution of your `video_features_self_annotations_2frames.csv` dataset when a webcam is disabled.

### 4. 🧠 Model 6A & 6B Real-Time Fusion Comparison
* **Model 6A (Late Probability Gating):** `P_fusion = α · P_phys + (1 - α) · P_vis`. Includes an **interactive $\alpha$-gating slider** (`0.0` to `1.0`, default `0.50`) so you can test shifting dominance between physical sensors vs. facial expressions.
* **Model 6B (Intermediate Cross-Attention Token Fuser):** Simulates attention allocation between the 64d physiological token and 77d visual trajectory, showing live attention balance percentages (`Physio 52% | Visual 48%`).

### 5. 📥 15-Column Telemetry Recording & CSV Export
* Record synchronized multimodal streams into memory and export clean, publication-ready datasets (`aurasense_model6_telemetry.csv`):
  `timestamp, eda, bvp, hr, temp, acc_mag, face_count, face_area_ratio, brightness, sharpness, motion, p_phys, p_vis, p_fusion_6a, p_fusion_6b`

---

## 🚀 How to Run the Website Locally

Since the application uses pure HTML5, CSS, and Vanilla JavaScript, launch it immediately with zero build step dependencies:

### Option 1: VS Code Live Server / Python HTTP Server (Recommended)
Open a terminal inside this directory and run:
```bash
python -m http.server 8000
```
Then open `http://localhost:8000` in **Google Chrome** or **Microsoft Edge**.

### Option 2: Direct File Open
Double-click `index.html` in this folder.
*(Note: Live webcam and Web Bluetooth require `localhost` or HTTPS for hardware/camera access, so the Python HTTP server or GitHub Pages deployment is preferred.)*
