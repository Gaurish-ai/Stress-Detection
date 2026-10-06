from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb

from config import EarlyFusionConfig
from data_loader import load_and_align_multimodal_windows, get_loso_splits
from models import (
    RawPhysioTFTransformer,
    IntermediateCrossAttentionFuser,
    EarlySynchronizedFuser,
    optimize_late_fusion_weight
)


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalFoldDataset(Dataset):
    def __init__(self, sensors_dict: dict[str, np.ndarray], video_seqs: np.ndarray, xgb_feats: np.ndarray, y: np.ndarray) -> None:
        self.sensors = {s: torch.tensor(sensors_dict[s], dtype=torch.float32) for s in sensors_dict}
        self.video_seqs = torch.tensor(video_seqs, dtype=torch.float32)
        self.xgb_feats = torch.tensor(xgb_feats, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        return {s: self.sensors[s][idx] for s in self.sensors}, self.video_seqs[idx], self.xgb_feats[idx], self.y[idx]


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


def run_loso_early_fusion_tournament() -> None:
    config = EarlyFusionConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================================================================", flush=True)
    print(f"Grand Head-to-Head LOSO Tournament: Early Fusion (Step 2) vs Model 4 V2 (Step 1) vs Late/Inter", flush=True)
    print(f"Running on device: {device} | Per-Subject Normalization Enabled: {config.use_baseline_subtraction}", flush=True)
    print(f"=========================================================================================\n", flush=True)
    
    subjects_data = load_and_align_multimodal_windows(config)
    print(f"\nLoaded {len(subjects_data)} common subjects for comprehensive evaluation.", flush=True)
    
    per_subject_results = []
    
    # Storage for pooled predictions across all folds
    all_y_true = []
    all_probs = {
        "physio_m4v2": [],
        "visual_m5a": [],
        "late_m6a": [],
        "inter_m6b": [],
        "early_m7": []
    }
    
    for subj in subjects_data:
        test_sid = subj["sid"]
        print(f"\n--- LOSO Fold: Holding out Subject P{test_sid:02d} (Test Windows: {len(subj['y'])}) ---", flush=True)
        train_dict, test_dict = get_loso_splits(subjects_data, test_sid, config)
        
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
        
        # 2. Train Model 4 V2 (Per-Subject Normalized E4 TF-Transformer)
        train_ds = MultimodalFoldDataset(train_dict["sensors"], train_dict["video_seqs"], train_dict["xgb_feats"], train_dict["y"])
        test_ds = MultimodalFoldDataset(test_dict["sensors"], test_dict["video_seqs"], test_dict["xgb_feats"], test_dict["y"])
        
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
            for batch_s, _, _, batch_y in train_loader:
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
            train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
            for batch_s, _, _, _ in train_eval_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                logits, tokens = phys_model(batch_s)
                train_probs_phys_list.append(torch.sigmoid(logits).cpu().numpy())
                train_physio_tokens_list.append(tokens.cpu())
                
        train_probs_phys = np.concatenate(train_probs_phys_list, axis=0)
        train_physio_tokens = torch.cat(train_physio_tokens_list, dim=0)
        
        # Evaluate Model 4 V2 on test set
        test_probs_phys_list, test_physio_tokens_list = [], []
        with torch.no_grad():
            for batch_s, _, _, _ in test_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                logits, tokens = phys_model(batch_s)
                test_probs_phys_list.append(torch.sigmoid(logits).cpu().numpy())
                test_physio_tokens_list.append(tokens.cpu())
                
        test_probs_phys = np.concatenate(test_probs_phys_list, axis=0)
        test_physio_tokens = torch.cat(test_physio_tokens_list, dim=0)
        
        # 3. Model 6A: Late Probability Gating
        best_alpha = optimize_late_fusion_weight(train_dict["y"], train_probs_phys, train_probs_vis)
        test_probs_late = best_alpha * test_probs_phys + (1.0 - best_alpha) * test_probs_vis
        
        # 4. Model 6B: Intermediate Cross-Attention Token Fuser
        inter_fuser = IntermediateCrossAttentionFuser(
            physio_dim=config.cnn_embed_dim,
            visual_dim=train_dict["xgb_feats"].shape[1],
            embed_dim=config.cnn_embed_dim,
            dropout=config.dropout
        ).to(device)
        
        fuser_optimizer = torch.optim.AdamW(inter_fuser.parameters(), lr=1e-3, weight_decay=1e-4)
        train_xgb_tensor = torch.tensor(train_dict["xgb_feats"], dtype=torch.float32)
        train_y_tensor = torch.tensor(train_dict["y"], dtype=torch.float32)
        
        inter_fuser.train()
        for epoch in range(config.epochs):
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
            
        # 5. Model 7 / Model 6C: Synchronized Early Feature Fusion (Step 2)
        early_model = EarlySynchronizedFuser(
            physio_dim=config.cnn_embed_dim,
            visual_dim=train_dict["video_seqs"].shape[-1],
            embed_dim=config.cnn_embed_dim,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.dropout
        ).to(device)
        
        early_optimizer = torch.optim.AdamW(early_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        early_model.train()
        for epoch in range(config.epochs):
            for batch_s, batch_v, _, batch_y in train_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                batch_v = batch_v.to(device)
                batch_y = batch_y.to(device)
                
                early_optimizer.zero_grad()
                logits_early, _ = early_model(batch_s, batch_v)
                loss_e = criterion(logits_early, batch_y)
                loss_e.backward()
                early_optimizer.step()
                
        early_model.eval()
        test_probs_early_list = []
        with torch.no_grad():
            for batch_s, batch_v, _, _ in test_loader:
                batch_s = {s: batch_s[s].to(device) for s in batch_s}
                batch_v = batch_v.to(device)
                logits_early, _ = early_model(batch_s, batch_v)
                test_probs_early_list.append(torch.sigmoid(logits_early).cpu().numpy())
                
        test_probs_early = np.concatenate(test_probs_early_list, axis=0)
        
        # Record accuracies across the 5 models
        y_t = test_dict["y"]
        acc_m4 = float(np.mean((test_probs_phys >= 0.5).astype(int) == y_t))
        acc_m5 = float(np.mean((test_probs_vis >= 0.5).astype(int) == y_t))
        acc_m6a = float(np.mean((test_probs_late >= 0.5).astype(int) == y_t))
        acc_m6b = float(np.mean((test_probs_inter >= 0.5).astype(int) == y_t))
        acc_m7 = float(np.mean((test_probs_early >= 0.5).astype(int) == y_t))
        
        print(
            f"P{test_sid:02d} -> Accs [Physio M4V2: {acc_m4:.2%} | Visual M5A: {acc_m5:.2%} | "
            f"Late M6A: {acc_m6a:.2%} (alpha={best_alpha:.2f}) | Inter M6B: {acc_m6b:.2%} | Early M7: {acc_m7:.2%}]",
            flush=True
        )
        
        per_subject_results.append({
            "subject": test_sid,
            "n_test": len(y_t),
            "n_stressed": int(np.sum(y_t == 1)),
            "acc_physio_m4v2": acc_m4,
            "acc_visual_m5a": acc_m5,
            "acc_late_m6a": acc_m6a,
            "acc_inter_m6b": acc_m6b,
            "acc_early_m7": acc_m7,
            "best_alpha": best_alpha
        })
        
        all_y_true.append(y_t)
        all_probs["physio_m4v2"].append(test_probs_phys)
        all_probs["visual_m5a"].append(test_probs_vis)
        all_probs["late_m6a"].append(test_probs_late)
        all_probs["inter_m6b"].append(test_probs_inter)
        all_probs["early_m7"].append(test_probs_early)
        
    # Pool across all folds
    pooled_y = np.concatenate(all_y_true, axis=0)
    pooled_metrics = {}
    mean_subject_accs = {}
    
    for k in all_probs:
        p_arr = np.concatenate(all_probs[k], axis=0)
        pooled_metrics[k] = compute_metrics(pooled_y, p_arr)
        
    mean_subject_accs["physio_m4v2"] = float(np.mean([r["acc_physio_m4v2"] for r in per_subject_results]))
    mean_subject_accs["visual_m5a"] = float(np.mean([r["acc_visual_m5a"] for r in per_subject_results]))
    mean_subject_accs["late_m6a"] = float(np.mean([r["acc_late_m6a"] for r in per_subject_results]))
    mean_subject_accs["inter_m6b"] = float(np.mean([r["acc_inter_m6b"] for r in per_subject_results]))
    mean_subject_accs["early_m7"] = float(np.mean([r["acc_early_m7"] for r in per_subject_results]))
    
    print("\n=========================================================================================", flush=True)
    print("TOURNAMENT COMPLETE: MEAN SUBJECT ACCURACY OVERALL (Across 18 Subjects)", flush=True)
    print(f"  - Model 4 V2 (Per-Subject Normalized Physio): {mean_subject_accs['physio_m4v2']:.2%}")
    print(f"  - Model 5A (Visual XGBoost Alone):          {mean_subject_accs['visual_m5a']:.2%}")
    print(f"  - Model 6A (Late Probability Fusion):       {mean_subject_accs['late_m6a']:.2%}")
    print(f"  - Model 6B (Intermediate Token Fusion):     {mean_subject_accs['inter_m6b']:.2%}")
    print(f"  - Model 7 / 6C (Synchronized Early Fusion): {mean_subject_accs['early_m7']:.2%}")
    print("=========================================================================================\n", flush=True)
    
    # Save JSON results
    output_data = {
        "mean_subject_accuracies": mean_subject_accs,
        "pooled_metrics": pooled_metrics,
        "per_subject": per_subject_results,
        "total_samples": len(pooled_y),
        "total_stressed": int(np.sum(pooled_y == 1))
    }
    with open(config.output_dir / "results.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    # Generate summary markdown report
    report_path = config.output_dir / "EARLY_FUSION_TOURNAMENT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Grand Head-to-Head LOSO Tournament: Early Fusion (Model 7) vs Model 4 V2 vs Late/Inter\n\n")
        f.write("## Executive Summary\n")
        f.write(f"- Total Evaluated Windows across 18 Subjects: **{len(pooled_y)}** (Stressed: **{int(np.sum(pooled_y == 1))}**)\n")
        f.write(f"- **Model 4 V2** (Per-Subject Normalized E4 Physiology): **{mean_subject_accs['physio_m4v2']:.2%}** Mean Accuracy\n")
        f.write(f"- **Model 5A** (Visual Statistical XGBoost Alone): **{mean_subject_accs['visual_m5a']:.2%}** Mean Accuracy\n")
        f.write(f"- **Model 6A** (Late Probability Fusion): **{mean_subject_accs['late_m6a']:.2%}** Mean Accuracy\n")
        f.write(f"- **Model 6B** (Intermediate Cross-Attention Token Fusion): **{mean_subject_accs['inter_m6b']:.2%}** Mean Accuracy\n")
        f.write(f"- **Model 7 / 6C** (Synchronized Early Feature Fusion): **{mean_subject_accs['early_m7']:.2%}** Mean Accuracy\n\n")
        
        f.write("## Per-Subject Detailed Breakdown\n\n")
        f.write("| Subject ID | Total | Stressed | Physio M4V2 | Visual M5A | Late M6A | Inter M6B | Early M7 | Optimal Alpha |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in per_subject_results:
            f.write(
                f"| P{r['subject']:02d} | {r['n_test']} | {r['n_stressed']} | "
                f"{r['acc_physio_m4v2']:.2%} | {r['acc_visual_m5a']:.2%} | {r['acc_late_m6a']:.2%} | "
                f"{r['acc_inter_m6b']:.2%} | **{r['acc_early_m7']:.2%}** | {r['best_alpha']:.2f} |\n"
            )
        f.write("\n## Key Findings & Takeaways\n")
        f.write("1. **Impact of Per-Subject Resting Baseline Normalization (Step 1):** By centering each subject's E4 sensors at `0.0` using their calm windows, Model 4 V2 no longer suffers from interpersonal baseline variance.\n")
        f.write("2. **Early Fusion vs Intermediate Fusion:** Demonstrates how synchronizing frame-level visual features with E4 sensor branches at Layer 0 (`EarlySynchronizedFuser`) compares against token-level cross-attention (`IntermediateCrossAttentionFuser`).\n")
        
    print(f"Results and report successfully written to {config.output_dir}", flush=True)


if __name__ == "__main__":
    run_loso_early_fusion_tournament()
