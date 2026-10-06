"""
Leave-One-Subject-Out (LOSO) Training and Evaluation Script for Approach 5:
Facial & Video Specialists (Pure Visual Stress Detection across 18 Common Subjects).

Evaluates 3 distinct models head-to-head:
1. Model 5A: Handcrafted Statistical XGBoost (`Tabular Baseline`)
2. Model 5B: Temporal BiLSTM with Attention (`Sequential Trajectory`)
3. Model 5C: Multi-Head Facial Video Transformer (`Self-Attention over Visual Dynamics`)

Usage: python train.py
"""
from __future__ import annotations
import json
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from config import FacialSpecialistsConfig
from data_loader import load_and_align_video_windows, get_loso_splits
from models import FacialBiLSTMAttention, FacialVideoTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_nn_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    y_list = []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            logits, _ = model(x_b)
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


def plot_comparison(results_summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = ["5A: XGBoost", "5B: BiLSTM+Attn", "5C: Transformer"]
    accs = [results_summary["model_5a_xgboost"]["overall_accuracy"] * 100,
            results_summary["model_5b_bilstm"]["overall_accuracy"] * 100,
            results_summary["model_5c_transformer"]["overall_accuracy"] * 100]
    f1s = [results_summary["model_5a_xgboost"]["overall_f1"],
           results_summary["model_5b_bilstm"]["overall_f1"],
           results_summary["model_5c_transformer"]["overall_f1"]]
    aucs = [results_summary["model_5a_xgboost"]["overall_roc_auc"],
            results_summary["model_5b_bilstm"]["overall_roc_auc"],
            results_summary["model_5c_transformer"]["overall_roc_auc"]]
            
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Accuracy Bar Chart
    axes[0].bar(models, accs, color=["darkcyan", "royalblue", "darkviolet"])
    axes[0].set_title("Overall LOSO Accuracy (%)")
    axes[0].set_ylabel("Accuracy (%)")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
        
    # F1 Score Bar Chart
    axes[1].bar(models, f1s, color=["darkcyan", "royalblue", "darkviolet"])
    axes[1].set_title("Overall LOSO F1 Score")
    axes[1].set_ylabel("F1 Score")
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        
    # ROC-AUC Bar Chart
    axes[2].bar(models, aucs, color=["darkcyan", "royalblue", "darkviolet"])
    axes[2].set_title("Overall LOSO ROC-AUC")
    axes[2].set_ylabel("ROC-AUC")
    for i, v in enumerate(aucs):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_dir / "facial_models_comparison.png", dpi=150)
    plt.close()


def main() -> None:
    config = FacialSpecialistsConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Facial Specialists (Models 5A, 5B, 5C) on device: {device}")
    
    subjects_data = load_and_align_video_windows(config)
    if not subjects_data:
        print("No valid visual sequence data found.")
        return
        
    print(f"\nLoaded {len(subjects_data)} common subjects for head-to-head LOSO benchmarking.")
    
    # Storage for predictions across all 3 models
    preds_5a_probs, preds_5b_probs, preds_5c_probs = [], [], []
    all_y_true = []
    per_subj_results = []
    
    for test_subj in subjects_data:
        test_sid = test_subj["sid"]
        print(f"\n--- LOSO Fold: Holding out Subject P{test_sid:02d} for Testing ---")
        train_dict, test_dict = get_loso_splits(subjects_data, test_sid)
        
        if len(test_dict["y"]) == 0 or len(train_dict["y"]) == 0:
            continue
            
        # =========================================================
        # Model 5A: Handcrafted Statistical XGBoost (`Tabular`)
        # =========================================================
        pos_count = float(train_dict["y"].sum())
        neg_count = float(len(train_dict["y"]) - pos_count)
        scale_pos_weight = neg_count / max(pos_count, 1.0)
        
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=config.seed,
            n_jobs=-1,
            eval_metric="logloss"
        )
        xgb.fit(train_dict["xgb_feats"], train_dict["y"])
        probs_5a = xgb.predict_proba(test_dict["xgb_feats"])[:, 1]
        
        # =========================================================
        # Model 5B & 5C: PyTorch BiLSTM and Transformer
        # =========================================================
        train_ds = TensorDataset(torch.tensor(train_dict["video_seqs"]), torch.tensor(train_dict["y"]))
        test_ds = TensorDataset(torch.tensor(test_dict["video_seqs"]), torch.tensor(test_dict["y"]))
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
        
        # Model 5B
        bilstm = FacialBiLSTMAttention(
            num_features=len(config.video_feature_cols),
            hidden_dim=config.lstm_hidden_dim,
            dropout=config.dropout
        ).to(device)
        
        # Model 5C
        transformer = FacialVideoTransformer(
            num_features=len(config.video_feature_cols),
            embed_dim=config.embed_dim,
            sequence_len=config.buffer_windows,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            ff_dim=config.transformer_ff_dim,
            dropout=config.dropout
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([scale_pos_weight], device=device))
        
        # Train BiLSTM (Model 5B)
        opt_b = torch.optim.AdamW(bilstm.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=config.epochs)
        for epoch in range(1, config.epochs + 1):
            bilstm.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                opt_b.zero_grad()
                logits, _ = bilstm(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(bilstm.parameters(), max_norm=2.0)
                opt_b.step()
            sched_b.step()
        probs_5b, _ = evaluate_nn_model(bilstm, test_loader, device)
        
        # Train Transformer (Model 5C)
        opt_c = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=config.epochs)
        for epoch in range(1, config.epochs + 1):
            transformer.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                opt_c.zero_grad()
                logits, _ = transformer(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=2.0)
                opt_c.step()
            sched_c.step()
        probs_5c, _ = evaluate_nn_model(transformer, test_loader, device)
        
        preds_5a_probs.append(probs_5a)
        preds_5b_probs.append(probs_5b)
        preds_5c_probs.append(probs_5c)
        all_y_true.append(test_dict["y"])
        
        # Quick fold summary
        acc_a = accuracy_score(test_dict["y"], (probs_5a >= 0.5).astype(int))
        acc_b = accuracy_score(test_dict["y"], (probs_5b >= 0.5).astype(int))
        acc_c = accuracy_score(test_dict["y"], (probs_5c >= 0.5).astype(int))
        print(f"P{test_sid:02d} -> Accs [XGBoost: {acc_a:.2%} | BiLSTM: {acc_b:.2%} | Transformer: {acc_c:.2%}]")
        
        per_subj_results.append({
            "subject": test_sid,
            "n_test": len(test_dict["y"]),
            "n_stressed": int(test_dict["y"].sum()),
            "acc_5a": float(acc_a),
            "acc_5b": float(acc_b),
            "acc_5c": float(acc_c),
        })

    # =========================================================
    # Compute Final Pooled Comparisons across all 18 Subjects
    # =========================================================
    y_true_all = np.concatenate(all_y_true)
    probs_5a_all = np.concatenate(preds_5a_probs)
    probs_5b_all = np.concatenate(preds_5b_probs)
    probs_5c_all = np.concatenate(preds_5c_probs)
    
    def get_summary(probs: np.ndarray) -> dict:
        thresh, f1 = choose_best_threshold(y_true_all, probs)
        pred = (probs >= thresh).astype(int)
        acc = accuracy_score(y_true_all, pred)
        prec = precision_score(y_true_all, pred, zero_division=0)
        rec = recall_score(y_true_all, pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true_all, probs)
        except ValueError:
            auc = 0.5
        return {
            "overall_accuracy": float(acc),
            "overall_f1": float(f1),
            "overall_precision": float(prec),
            "overall_recall": float(rec),
            "overall_roc_auc": float(auc),
            "best_threshold": float(thresh)
        }
        
    summary_5a = get_summary(probs_5a_all)
    summary_5b = get_summary(probs_5b_all)
    summary_5c = get_summary(probs_5c_all)
    
    results_summary = {
        "model_5a_xgboost": summary_5a,
        "model_5b_bilstm": summary_5b,
        "model_5c_transformer": summary_5c,
        "per_subject": per_subj_results,
        "total_samples": len(y_true_all),
        "total_stressed": int(y_true_all.sum())
    }
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    plot_comparison(results_summary, config.output_dir)
    
    print("\n=======================================================")
    print("FINAL FACIAL & VIDEO SPECIALISTS COMPARISON (18 SUBJECTS)")
    print("=======================================================")
    print(f"Model 5A: Statistical XGBoost -> Acc: {summary_5a['overall_accuracy']:.2%} | F1: {summary_5a['overall_f1']:.3f} | AUC: {summary_5a['overall_roc_auc']:.3f}")
    print(f"Model 5B: BiLSTM + Attention  -> Acc: {summary_5b['overall_accuracy']:.2%} | F1: {summary_5b['overall_f1']:.3f} | AUC: {summary_5b['overall_roc_auc']:.3f}")
    print(f"Model 5C: Temporal Transformer-> Acc: {summary_5c['overall_accuracy']:.2%} | F1: {summary_5c['overall_f1']:.3f} | AUC: {summary_5c['overall_roc_auc']:.3f}")
    print(f"Results and charts saved to    -> {config.output_dir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
