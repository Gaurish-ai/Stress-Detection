"""
Training Pipeline -- XGBoost Baseline with LOSO Cross-Validation.

Usage:
    python train.py

Pipeline steps:
    1. Load all sensor data + annotations for 28 subjects
    2. Extract 34 handcrafted features per 5-second window
    3. Train XGBoost with Leave-One-Subject-Out cross-validation
    4. Evaluate and save results (metrics, plots, model artifacts)
"""
import os
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import XGBOOST_PARAMS, OUTPUT_DIR
from data_loader import load_all_subjects
from feature_engineering import build_feature_matrix, normalize_per_subject
from evaluate import (
    print_overall_results,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_per_subject_results,
    plot_roc_curve,
    plot_stress_distribution,
    save_results,
)


def loso_cross_validation(X, y, subject_ids, feature_names):
    """
    Leave-One-Subject-Out cross-validation with XGBoost.

    For each of the 28 subjects:
        - Train on the other 27 subjects
        - Test on the held-out subject
        - Collect predictions and feature importances

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) binary labels
        subject_ids: (n_samples,) subject ID per sample
        feature_names: list of feature names

    Returns:
        dict with per-subject results, predictions, and feature importances
    """
    unique_subjects = np.unique(subject_ids)

    all_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    feature_importances = np.zeros(len(feature_names))
    n_folds_used = 0

    print(f"\n{'=' * 60}")
    print(f"  LOSO Cross-Validation  --  {len(unique_subjects)} folds")
    print(f"{'=' * 60}\n")

    for i, test_subject in enumerate(unique_subjects):
        # ---- Split ----
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_test) == 0:
            print(f"  Fold {i+1:2d}: Subject P{test_subject:2d} -- SKIPPED (no test samples)")
            continue

        # ---- Feature scaling ----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Replace NaN / Inf from edge cases
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- SMOTE Oversampling ----
        n_pos = int(y_train.sum())
        n_neg = int((1 - y_train).sum())
        if n_pos > 1 and n_neg > 0:
            k_neighbors = min(5, n_pos - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_fit, y_train_fit = smote.fit_resample(X_train_s, y_train)
            sp_weight = 1.0  # Already balanced by SMOTE
        else:
            X_train_fit, y_train_fit = X_train_s, y_train
            sp_weight = n_neg / max(n_pos, 1)

        # ---- Train ----
        params = XGBOOST_PARAMS.copy()
        model = XGBClassifier(**params, scale_pos_weight=sp_weight)
        model.fit(X_train_fit, y_train_fit, verbose=False)

        # ---- Optimal Threshold Tuning on Training Fold ----
        y_train_prob = model.predict_proba(X_train_s)[:, 1]
        best_thresh = 0.5
        best_f1 = -1.0
        for thresh in np.arange(0.15, 0.85, 0.02):
            score = f1_score(y_train, (y_train_prob > thresh).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        # ---- Predict ----
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob > best_thresh).astype(int)

        # ---- Metrics ----
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        result = {
            "subject": int(test_subject),
            "n_test": int(len(y_test)),
            "n_stressed": int(y_test.sum()),
            "n_not_stressed": int((1 - y_test).sum()),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        }
        all_results.append(result)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        feature_importances += model.feature_importances_
        n_folds_used += 1

        print(
            f"  Fold {i+1:2d}: P{test_subject:2d}  |  "
            f"Acc {acc:.3f}  F1 {f1:.3f}  "
            f"Prec {prec:.3f}  Rec {rec:.3f}  |  "
            f"{len(y_test)} samples ({y_test.sum():.0f} stressed)"
        )

    # Average feature importance across folds
    if n_folds_used > 0:
        feature_importances /= n_folds_used

    return {
        "per_subject": all_results,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_prob": np.array(all_y_prob),
        "feature_importances": feature_importances,
        "feature_names": feature_names,
    }


# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("  SENSOR STRESS DETECTION -- BASELINE MODEL (XGBoost)")
    print("=" * 60)

    t0 = time.time()

    # ---- Step 1: Load data ----
    print("\n[1/4] Loading sensor data and annotations...")
    all_data = load_all_subjects()

    # ---- Step 2: Feature extraction ----
    print("\n[2/5] Extracting features (34 base + 34 delta = 68 per window)...")
    X, y, subject_ids, feature_names, stress_scores = build_feature_matrix(all_data)

    # ---- Step 3: Per-subject normalization ----
    print("\n[3/5] Normalizing features per-subject (z-score)...")
    X = normalize_per_subject(X, subject_ids)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("  [OK] Per-subject normalization applied.")

    # ---- Step 4: LOSO training ----
    print("\n[4/5] Training with LOSO cross-validation...")
    results = loso_cross_validation(X, y, subject_ids, feature_names)

    # ---- Step 5: Evaluation & output ----
    print("\n[5/5] Evaluating and saving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    overall = print_overall_results(results)
    plot_confusion_matrix(results, OUTPUT_DIR)
    plot_feature_importance(results, OUTPUT_DIR)
    plot_per_subject_results(results, OUTPUT_DIR)
    plot_roc_curve(results, OUTPUT_DIR)
    plot_stress_distribution(stress_scores, y, OUTPUT_DIR)
    save_results(results, OUTPUT_DIR, overall)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  [OK] Pipeline completed in {elapsed:.1f} seconds")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
