from __future__ import annotations
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

from config import MultimodalFusionConfig
from data_loader import load_and_align_multimodal_windows, get_normalized_loso_splits
from models import RawPhysioTFTransformer, IntermediateCrossAttentionFuser, optimize_late_fusion_weight


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalFoldDataset(Dataset):
    def __init__(self, sensors_dict: dict[str, np.ndarray], xgb_feats: np.ndarray, y: np.ndarray) -> None:
        self.sensors = {s: torch.tensor(sensors_dict[s], dtype=torch.float32) for s in sensors_dict}
        self.xgb_feats = torch.tensor(xgb_feats, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        return {s: self.sensors[s][idx] for s in self.sensors}, self.xgb_feats[idx], self.y[idx]


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, thresh: float = 0.5) -> dict[str, float]:
    pred = (probs >= thresh).astype(int)
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.5
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "roc_auc": auc}


def run_loso_multimodal_fusion() -> None:
    config = MultimodalFusionConfig()
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Approach 6: Multimodal Fusion (Raw Model 4 + Model 5A) on device: {device}", flush=True)
    
    subjects_data = load_and_align_multimodal_windows(config)
    print(f"\nLoaded {len(subjects_data)} common subjects for head-to-head LOSO benchmarking.", flush=True)
    
    per_subject_results = []
    
    # Storage for pooled predictions
    all_y_true = []
    all_probs_phys = []
    all_probs_vis = []
    all_probs_late = []
    all_probs_inter = []
    
    for subj in subjects_data:
        test_sid = subj["sid"]
        print(f"\n--- LOSO Fold: Holding out Subject P{test_sid:02d} for Testing ---", flush=True)
        train_dict, test_dict = get_normalized_loso_splits(subjects_data, test_sid)
        
        # 1. Fit Model 5A (Statistical Visual XGBoost)
        n_pos = np.sum(train_dict["y"] == 1)
        n_neg = np.sum(train_dict["y"] == 0)
        scale_pos = float(n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=config.seed,
            n_jobs=-1,
            eval_metric="logloss"
        )
        xgb_model.fit(train_dict["xgb_feats"], train_dict["y"])
        
        train_probs_vis = xgb_model.predict_proba(train_dict["xgb_feats"])[:, 1]
        test_probs_vis = xgb_model.predict_proba(test_dict["xgb_feats"])[:, 1]
        
        # 2. Train Model 4 (RawPhysioTFTransformer)
        train_ds = MultimodalFoldDataset(train_dict["sensors"], train_dict["xgb_feats"], train_dict["y"])
        test_ds = MultimodalFoldDataset(test_dict["sensors"], test_dict["xgb_feats"], test_dict["y"])
        
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
        
        phys_model = RawPhysioTFTransformer(
            embed_dim=config.cnn_embed_dim,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.dropout
        ).to(device)
        
        pos_weight = torch.tensor([scale_pos], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(phys_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        phys_model.train()
        for epoch in range(config.epochs):
            for batch_s, _, batch_y in train_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                logits, _ = phys_model(batch_s)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
        # Extract train probabilities and physio tokens for fusion optimization
        phys_model.eval()
        train_probs_phys_list, train_physio_tokens_list = [], []
        with torch.no_grad():
            # Evaluate train set in batches without shuffle
            train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
            for batch_s, _, _ in train_eval_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                logits, tokens = phys_model(batch_s)
                train_probs_phys_list.append(torch.sigmoid(logits).cpu().numpy())
                train_physio_tokens_list.append(tokens.cpu())
                
        train_probs_phys = np.concatenate(train_probs_phys_list, axis=0)
        train_physio_tokens = torch.cat(train_physio_tokens_list, dim=0)
        
        # Evaluate test set
        test_probs_phys_list, test_physio_tokens_list = [], []
        with torch.no_grad():
            for batch_s, _, _ in test_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                logits, tokens = phys_model(batch_s)
                test_probs_phys_list.append(torch.sigmoid(logits).cpu().numpy())
                test_physio_tokens_list.append(tokens.cpu())
                
        test_probs_phys = np.concatenate(test_probs_phys_list, axis=0)
        test_physio_tokens = torch.cat(test_physio_tokens_list, dim=0)
        
        # 3. Model 6A: Late Probability Gating / Ensemble
        best_alpha = optimize_late_fusion_weight(train_dict["y"], train_probs_phys, train_probs_vis)
        test_probs_late = best_alpha * test_probs_phys + (1.0 - best_alpha) * test_probs_vis
        
        # 4. Model 6B: Intermediate Cross-Attention Token Fuser
        inter_fuser = IntermediateCrossAttentionFuser(
            physio_dim=config.cnn_embed_dim,
            visual_dim=train_dict["xgb_feats"].shape[1],
            embed_dim=config.cnn_embed_dim,
            dropout=config.dropout
        ).to(device)
        
        # Train intermediate fuser on frozen/extracted tokens
        fuser_optimizer = torch.optim.AdamW(inter_fuser.parameters(), lr=1e-3, weight_decay=1e-4)
        train_xgb_tensor = torch.tensor(train_dict["xgb_feats"], dtype=torch.float32)
        train_y_tensor = torch.tensor(train_dict["y"], dtype=torch.float32)
        
        inter_fuser.train()
        for epoch in range(15):
            perm = torch.randperm(len(train_y_tensor))
            for i in range(0, len(train_y_tensor), config.batch_size):
                idx_b = perm[i:i + config.batch_size]
                p_tok_b = train_physio_tokens[idx_b].to(device)
                v_feat_b = train_xgb_tensor[idx_b].to(device)
                y_b = train_y_tensor[idx_b].to(device)
                
                fuser_optimizer.zero_grad()
                out_logits = inter_fuser(p_tok_b, v_feat_b)
                loss_f = criterion(out_logits, y_b)
                loss_f.backward()
                fuser_optimizer.step()
                
        inter_fuser.eval()
        with torch.no_grad():
            test_xgb_tensor = torch.tensor(test_dict["xgb_feats"], dtype=torch.float32, device=device)
            test_p_tokens = test_physio_tokens.to(device)
            inter_logits = inter_fuser(test_p_tokens, test_xgb_tensor)
            test_probs_inter = torch.sigmoid(inter_logits).cpu().numpy()
            
        # Compute and record per-subject accuracies
        acc_phys = float(np.mean((test_probs_phys >= 0.5).astype(int) == test_dict["y"]))
        acc_vis = float(np.mean((test_probs_vis >= 0.5).astype(int) == test_dict["y"]))
        acc_late = float(np.mean((test_probs_late >= 0.5).astype(int) == test_dict["y"]))
        acc_inter = float(np.mean((test_probs_inter >= 0.5).astype(int) == test_dict["y"]))
        
        print(
            f"P{test_sid:02d} -> Accs [Physio(M4): {acc_phys:.2%} | Visual(M5A): {acc_vis:.2%} | "
            f"Late Fusion(M6A): {acc_late:.2%} | Inter Fusion(M6B): {acc_inter:.2%}] (alpha={best_alpha:.2f})",
            flush=True
        )
        
        per_subject_results.append({
            "subject": test_sid,
            "n_test": len(test_dict["y"]),
            "n_stressed": int(np.sum(test_dict["y"] == 1)),
            "acc_physio_m4": acc_phys,
            "acc_visual_m5a": acc_vis,
            "acc_late_m6a": acc_late,
            "acc_inter_m6b": acc_inter,
            "best_alpha": best_alpha
        })
        
        all_y_true.extend(test_dict["y"].tolist())
        all_probs_phys.extend(test_probs_phys.tolist())
        all_probs_vis.extend(test_probs_vis.tolist())
        all_probs_late.extend(test_probs_late.tolist())
        all_probs_inter.extend(test_probs_inter.tolist())
        
    # Pool final summary
    y_true_all = np.array(all_y_true)
    summary_phys = compute_metrics(y_true_all, np.array(all_probs_phys))
    summary_vis = compute_metrics(y_true_all, np.array(all_probs_vis))
    summary_late = compute_metrics(y_true_all, np.array(all_probs_late))
    summary_inter = compute_metrics(y_true_all, np.array(all_probs_inter))
    
    mean_phys = float(np.mean([r["acc_physio_m4"] for r in per_subject_results]))
    mean_vis = float(np.mean([r["acc_visual_m5a"] for r in per_subject_results]))
    mean_late = float(np.mean([r["acc_late_m6a"] for r in per_subject_results]))
    mean_inter = float(np.mean([r["acc_inter_m6b"] for r in per_subject_results]))
    
    print("\n=======================================================", flush=True)
    print("FINAL MULTIMODAL FUSION COMPARISON (18 SUBJECTS)", flush=True)
    print("=======================================================", flush=True)
    print(f"Model 4 (Physio Alone)   -> Mean Subj Acc: {mean_phys:.2%} | Pooled AUC: {summary_phys['roc_auc']:.3f}", flush=True)
    print(f"Model 5A (Visual Alone)  -> Mean Subj Acc: {mean_vis:.2%} | Pooled AUC: {summary_vis['roc_auc']:.3f}", flush=True)
    print(f"Model 6A (Late Fusion)   -> Mean Subj Acc: {mean_late:.2%} | Pooled AUC: {summary_late['roc_auc']:.3f}", flush=True)
    print(f"Model 6B (Inter Fusion)  -> Mean Subj Acc: {mean_inter:.2%} | Pooled AUC: {summary_inter['roc_auc']:.3f}", flush=True)
    print("=======================================================", flush=True)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model_4_physio": summary_phys,
        "model_5a_visual": summary_vis,
        "model_6a_late_fusion": summary_late,
        "model_6b_inter_fusion": summary_inter,
        "mean_subject_accuracies": {
            "physio_m4": mean_phys,
            "visual_m5a": mean_vis,
            "late_fusion_m6a": mean_late,
            "inter_fusion_m6b": mean_inter
        },
        "per_subject": per_subject_results,
        "total_samples": len(y_true_all),
        "total_stressed": int(np.sum(y_true_all == 1))
    }
    
    with open(config.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ["Model 4 (Physio)", "Model 5A (Visual)", "Model 6A (Late Fusion)", "Model 6B (Inter Fusion)"]
    accs = [mean_phys * 100, mean_vis * 100, mean_late * 100, mean_inter * 100]
    aucs = [summary_phys["roc_auc"] * 100, summary_vis["roc_auc"] * 100, summary_late["roc_auc"] * 100, summary_inter["roc_auc"] * 100]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, accs, width, label="Mean Subject Accuracy (%)", color="#3b82f6")
    ax.bar(x + width/2, aucs, width, label="Pooled ROC-AUC (%)", color="#10b981")
    
    ax.set_ylabel("Score (%)")
    ax.set_title("Multimodal Fusion vs. Single Modality Benchmarks (18 Common Subjects)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 105)
    
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax.text(i - width/2, a + 1, f"{a:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.text(i + width/2, u + 1, f"{u:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(config.output_dir / "fusion_comparison.png", dpi=300)
    plt.close()
    print(f"Results saved to -> {config.output_dir}", flush=True)


if __name__ == "__main__":
    run_loso_multimodal_fusion()
