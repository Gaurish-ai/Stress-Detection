# Multimodal Stress Detection

This repository contains a six-stage research project for multimodal stress detection using physiological sensor data, facial/video features, and a static browser dashboard for a 60-second stress assessment.

## Public Website

The working website lives in:

```text
model_6_website/
```

It is a static HTML/CSS/JavaScript site, so it does not need a build step.

Run it locally:

```bash
cd model_6_website
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

The site is designed to run on GitHub Pages through the workflow in `.github/workflows/deploy-pages.yml`. After pushing this repository to GitHub, enable Pages with the **GitHub Actions** source in the repository settings.

## Repository Structure

```text
first model(baseline model)/                  Baseline tabular XGBoost pipeline
second model(1d cnn on raw signals)/          Raw waveform 1D CNN pipeline
third model(cnn + bilstm)/                    CNN + BiLSTM + attention pipeline
fourth model(dual stream tf-transformer)/     Dual-stream temporal-frequency transformer
fifth model(facial video specialists)/        Facial/video specialist models
sixth model(multimodal fusion)/               Model 6A/6B multimodal fusion experiments
early fusion/                                 Early fusion experiments
model_6_website/                              Public static website/dashboard
*.pdf, *.md                                   Project reports and analysis summaries
```

## Python Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch installation can depend on your CPU/GPU/CUDA setup, so if `pip install torch` does not match your machine, install the correct PyTorch build from the official PyTorch selector.

## Data Notice

The training scripts expect external research data that is not included in this repository, including paths such as:

```text
D:\DATASET
D:\Sensor Stress\multimodal_stress_pipeline\outputs_smoke\video_features_self_annotations_2frames.csv
```

Those paths should be updated or documented before other people can reproduce the model training end to end. The public website does not require the raw dataset to load.

## What Should Be Uploaded

Upload the project source code, reports, generated result summaries, website files, `.gitignore`, `requirements.txt`, this README, and the GitHub Pages workflow.

Do not upload raw private datasets, `.env` files, Python cache folders, or future trained model binaries unless you are sure they are public and small enough for GitHub.
