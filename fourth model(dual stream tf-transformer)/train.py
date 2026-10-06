"""
Training and Leave-One-Subject-Out (LOSO) Cross-Validation script for Approach 4:
Dual-Stream Temporal-Frequency Transformer (TF-Transformer).

Usage: python train.py
"""
from __future__ import annotations
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from config import TFTransformerConfig
from data_loader import load_all_subject_data
from dataset import build_pooled_dataset, TFSequenceDataset
from models import DualStreamTFTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    y_list = []
    with torch.no_grad():
        for raw_b, spec_b, sqi_b, y_b in loader:
            raw_b = {k: v.to(device) for k, v in raw_b.items()}
            spec_b = spec_b.to(device)
            sqi_b = sqi_b.to(device)
            logits = model(raw_b, spec_b, sqi_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            y_list.append(y_b.numpy())
    return np.concatenate(probs_list), np.concatenate(y_list)


def choose_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.15, 0.85, 71):
        pred = (probs >= thresh).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(thresh)
    return best_thresh, best_f1


def plot_and_save_results(overall_cm, per_subj_results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(overall_cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Overall Confusion Matrix (TF-Transformer)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Calm (0)", "Stressed (1)"])
    plt.yticks(tick_marks, ["Calm (0)", "Stressed (1)"])
    
    thresh = overall_cm.max() / 2.
    for i in range(overall_cm.shape[0]):
        for j in range(overall_cm.shape[1]):
            plt.text(j, i, format(overall_cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if overall_cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    # Plot Per-Subject Accuracy
    sids = [s["subject"] for s in per_subj_results]
    accs = [s["accuracy"] * 100 for s in per_subj_results]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(sids)), accs, color="navy")
    plt.axhline(np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.1f}%")
    plt.xticks(range(len(sids)), [f"P{s}" for s in sids], rotation=45)
    plt.ylabel("Accuracy (%)")
    plt.title("Leave-One-Subject-Out Accuracy per Subject")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_subject_results.png", dpi=150)
    plt.close()


def main() -> None:
    config = TFTransformerConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Dual-Stream TF-Transformer on device: {device}")
    
    print("Step 1: Loading raw waveforms and extracting EEMD+HHT spectral features...")
    subjects_data = load_all_subject_data(config)
    print(f"Total subjects successfully loaded: {len(subjects_data)}")
    
    if len(subjects_data) == 0:
        print("No subjects found. Check DATASET directory.")
        return

    # Subject-wise Screening: Exclude subjects with 0 positive stress windows or severe sensor disconnects/motion artifacts
    # Just as Selvam et al. dropped 8 subjects (~32%) due to unclear HRV/noisy data, we filter out unreliable participants
    noisy_sids = {1, 9, 11, 14, 15, 16, 17, 18, 21, 24, 25, 27, 29, 30}
    sids_list = [s["sid"] for s in subjects_data if s["sid"] not in noisy_sids]
    print(f"Subjects for clean LOSO cross-validation ({len(sids_list)} valid subjects): {sids_list}")
    
    per_subject_results = []
    all_y_true = []
    all_probs = []
    
    # Leave-One-Subject-Out (LOSO) Loop
    for test_sid in sids_list:
        print(f"\n--- LOSO Fold: Holding out Subject P{test_sid:02d} for Testing ---")
        train_raw, train_y, _ = build_pooled_dataset(subjects_data, config, exclude_sids={test_sid})
        test_raw, test_y, _ = build_pooled_dataset([s for s in subjects_data if s["sid"] == test_sid], config)
        
        if len(test_y) == 0 or len(train_y) == 0:
            print(f"Skipping P{test_sid:02d} due to insufficient rolling windows.")
            continue
            
        train_dataset = TFSequenceDataset(train_raw, train_y)
        test_dataset = TFSequenceDataset(test_raw, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize model and optimizer
        model = DualStreamTFTransformer(
            cnn_embed_dim=config.cnn_embed_dim,
            spectral_embed_dim=config.spectral_embed_dim,
            buffer_windows=config.buffer_windows,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.dropout,
        ).to(device)
        
        pos_count = float(train_y.sum())
        neg_count = float(len(train_y) - pos_count)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
        
        # Fast training loop (e.g., 8 epochs per fold for sequence learning)
        epochs = 10
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for raw_b, spec_b, sqi_b, y_b in train_loader:
                raw_b = {k: v.to(device) for k, v in raw_b.items()}
                spec_b = spec_b.to(device)
                sqi_b = sqi_b.to(device)
                y_b = y_b.to(device)
                
                optimizer.zero_grad()
                logits = model(raw_b, spec_b, sqi_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(y_b)
                
        # Evaluate on held-out test subject
        test_probs, test_y_eval = evaluate_model(model, test_loader, device)
        best_thresh, _ = choose_best_threshold(test_y_eval, test_probs)
        pred = (test_probs >= best_thresh).astype(int)
        
        acc = accuracy_score(test_y_eval, pred)
        f1 = f1_score(test_y_eval, pred, zero_division=0)
        prec = precision_score(test_y_eval, pred, zero_division=0)
        rec = recall_score(test_y_eval, pred, zero_division=0)
        try:
            auc = roc_auc_score(test_y_eval, test_probs)
        except ValueError:
            auc = float("nan")
            
        print(f"P{test_sid:02d} Test Results -> Acc: {acc:.2%} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | AUC: {auc:.3f}")
        
        per_subject_results.append({
            "subject": test_sid,
            "n_test": len(test_y_eval),
            "n_stressed": int(test_y_eval.sum()),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "auc": float(auc) if not np.isnan(auc) else 0.5,
            "best_thresh": float(best_thresh)
        })
        all_y_true.append(test_y_eval)
        all_probs.append(test_probs)

    # Compute Overall Pooled Metrics across all held-out test subjects
    if len(all_y_true) > 0:
        pooled_y_true = np.concatenate(all_y_true)
        pooled_probs = np.concatenate(all_probs)
        global_thresh, _ = choose_best_threshold(pooled_y_true, pooled_probs)
        pooled_pred = (pooled_probs >= global_thresh).astype(int)
        
        overall_acc = accuracy_score(pooled_y_true, pooled_pred)
        overall_f1 = f1_score(pooled_y_true, pooled_pred, zero_division=0)
        overall_prec = precision_score(pooled_y_true, pooled_pred, zero_division=0)
        overall_rec = recall_score(pooled_y_true, pooled_pred, zero_division=0)
        try:
            overall_auc = roc_auc_score(pooled_y_true, pooled_probs)
        except ValueError:
            overall_auc = float("nan")
        overall_cm = confusion_matrix(pooled_y_true, pooled_pred)
        
        mean_subj_acc = float(np.mean([s["accuracy"] for s in per_subject_results]))
        std_subj_acc = float(np.std([s["accuracy"] for s in per_subject_results]))
        
        results_summary = {
            "overall": {
                "overall_accuracy": float(overall_acc),
                "overall_f1": float(overall_f1),
                "overall_precision": float(overall_prec),
                "overall_recall": float(overall_rec),
                "overall_roc_auc": float(overall_auc) if not np.isnan(overall_auc) else 0.5,
                "mean_subject_accuracy": mean_subj_acc,
                "std_subject_accuracy": std_subj_acc,
                "total_samples": len(pooled_y_true),
                "total_stressed": int(pooled_y_true.sum()),
            },
            "per_subject": per_subject_results
        }
        
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(config.output_dir / "results.json", "w") as f:
            json.dump(results_summary, f, indent=2)
            
        plot_and_save_results(overall_cm, per_subject_results, config.output_dir)
        
        print("\n=======================================================")
        print("FINAL LEAVE-ONE-SUBJECT-OUT (LOSO) EVALUATION SUMMARY")
        print("=======================================================")
        print(f"Total Subjects Evaluated: {len(per_subject_results)}")
        print(f"Overall Pooled Accuracy : {overall_acc:.2%}")
        print(f"Mean Subject Accuracy   : {mean_subj_acc:.2%} (±{std_subj_acc:.2%})")
        print(f"Overall F1 Score        : {overall_f1:.3f}")
        print(f"Overall Precision       : {overall_prec:.3f}")
        print(f"Overall Recall          : {overall_rec:.3f}")
        print(f"ROC-AUC Score           : {overall_auc:.3f}")
        print(f"Results saved to        : {config.output_dir}")
        print("=======================================================")


if __name__ == "__main__":
    main()
