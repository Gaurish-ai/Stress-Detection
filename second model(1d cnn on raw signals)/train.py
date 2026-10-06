"""
Training Pipeline for Approach 2: Multi-Branch 1D CNN with LOSO Cross-Validation.

Usage:
    python train.py
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    BATCH_SIZE,
    EPOCHS_PER_FOLD,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    USE_WEIGHTED_SAMPLER,
    OUTPUT_DIR,
)
from data_loader import load_all_raw_waveforms
from dataset import RawSensorDataset
from models import MultiBranch1DCNN
from evaluate import evaluate_predictions


def train_fold(model, train_loader, optimizer, criterion, device, epochs):
    """Train the 1D CNN model for a single LOSO fold."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for acc, eda, temp, hr, y in train_loader:
            acc, eda, temp, hr, y = acc.to(device), eda.to(device), temp.to(device), hr.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(acc, eda, temp, hr)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def predict_dataset(model, data_loader, device):
    """Get predicted probabilities and ground truth labels for a dataset."""
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for acc, eda, temp, hr, y in data_loader:
            acc, eda, temp, hr = acc.to(device), eda.to(device), temp.to(device), hr.to(device)
            logits = model(acc, eda, temp, hr)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_y.extend(y.numpy().tolist())
    return np.array(all_y), np.array(all_probs)


def main():
    print("=" * 65)
    print("  APPROACH 2: MULTI-BRANCH 1D CNN ON RAW SENSOR WAVEFORMS")
    print("=" * 65)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Step 1: Load Raw Waveform Dataset ----
    print("[1/3] Loading multi-rate raw waveforms and applying Z-score normalization...")
    data = load_all_raw_waveforms()

    acc_all = data["acc"]
    eda_all = data["eda"]
    temp_all = data["temp"]
    hr_all = data["hr"]
    y_all = data["y"]
    sids_all = data["subject_ids"]

    unique_subjects = np.unique(sids_all)
    print(f"\n[2/3] Starting Leave-One-Subject-Out (LOSO) Cross-Validation across {len(unique_subjects)} subjects...")
    print("-" * 65)

    all_results = []

    for i, test_sid in enumerate(unique_subjects):
        test_mask = sids_all == test_sid
        train_mask = ~test_mask

        # Check train/test samples
        n_test = test_mask.sum()
        if n_test == 0:
            continue

        # Prepare train and test PyTorch datasets
        train_dataset = RawSensorDataset(
            acc_all[train_mask], eda_all[train_mask], temp_all[train_mask], hr_all[train_mask], y_all[train_mask]
        )
        test_dataset = RawSensorDataset(
            acc_all[test_mask], eda_all[test_mask], temp_all[test_mask], hr_all[test_mask], y_all[test_mask]
        )

        # Handle class imbalance during training with WeightedRandomSampler
        if USE_WEIGHTED_SAMPLER:
            y_tr = y_all[train_mask]
            class_counts = np.bincount(y_tr)
            if len(class_counts) == 2 and class_counts[1] > 0 and class_counts[0] > 0:
                class_weights = {0: 1.0 / class_counts[0], 1: 1.0 / class_counts[1]}
                sample_weights = [class_weights[cls] for cls in y_tr]
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            else:
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Non-shuffled loaders for evaluation
        train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Instantiate Model and Optimizer
        model = MultiBranch1DCNN(dropout_rate=DROPOUT_RATE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        # Train model for this fold
        train_loss = train_fold(model, train_loader, optimizer, criterion, device, epochs=EPOCHS_PER_FOLD)

        # Threshold optimization on training fold predictions
        y_tr_true, y_tr_prob = predict_dataset(model, train_eval_loader, device)
        best_thresh = 0.5
        best_f1 = -1.0
        for thresh in np.arange(0.15, 0.85, 0.02):
            score = f1_score(y_tr_true, (y_tr_prob > thresh).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        # Evaluate on held-out subject test fold
        y_te_true, y_te_prob = predict_dataset(model, test_loader, device)
        y_te_pred = (y_te_prob > best_thresh).astype(int)

        acc = accuracy_score(y_te_true, y_te_pred)
        f1 = f1_score(y_te_true, y_te_pred, zero_division=0)
        prec = precision_score(y_te_true, y_te_pred, zero_division=0)
        rec = recall_score(y_te_true, y_te_pred, zero_division=0)

        all_results.append({
            "subject": int(test_sid),
            "n_test": int(n_test),
            "n_stressed": int(y_te_true.sum()),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "best_thresh": float(best_thresh),
            "y_true": y_te_true.tolist(),
            "y_pred": y_te_pred.tolist(),
            "y_prob": y_te_prob.tolist(),
        })

        print(
            f"  Fold {i+1:2d} (P{test_sid:2d}): Loss {train_loss:.3f} | "
            f"Acc {acc:.3f}  F1 {f1:.3f}  Prec {prec:.3f}  Rec {rec:.3f} | "
            f"Thresh {best_thresh:.2f} ({y_te_true.sum()} stressed samples)"
        )

    print("-" * 65)
    print("\n[3/3] Computing global evaluation metrics and saving artifacts...")
    overall = evaluate_predictions(all_results, OUTPUT_DIR)

    elapsed = time.time() - t0
    print(f"\n[OK] Approach 2 pipeline completed successfully in {elapsed:.1f} seconds ({elapsed/60:.1f} min).")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
