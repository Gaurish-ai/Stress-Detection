"""
PyTorch Dataset and Rolling Sequence Builder for Approach 4 (Dual-Stream TF-Transformer).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from config import TFTransformerConfig


def create_rolling_sequences(subj_dict: dict, config: TFTransformerConfig) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Given chronological windows of a single subject, construct rolling 12-window (60s) sequences.
    For each index i >= buffer_windows - 1, the sequence is windows [i - buffer_windows + 1 : i + 1].
    """
    buf = config.buffer_windows
    n_win = len(subj_dict["y"])
    if n_win < buf:
        return {}, np.array([]), np.array([]), np.array([])
        
    seq_count = n_win - buf + 1
    
    eda_seq = np.zeros((seq_count, buf, 1, config.window_samples["eda"]), dtype=np.float32)
    bvp_seq = np.zeros((seq_count, buf, 1, config.window_samples["bvp"]), dtype=np.float32)
    acc_seq = np.zeros((seq_count, buf, 4, config.window_samples["acc"]), dtype=np.float32)
    temp_seq = np.zeros((seq_count, buf, 1, config.window_samples["temp"]), dtype=np.float32)
    spectral_seq = np.zeros((seq_count, buf, 21), dtype=np.float32)
    sqi_seq = np.zeros((seq_count, buf), dtype=np.float32)
    y_seq = np.zeros((seq_count,), dtype=np.float32)
    scores_seq = np.zeros((seq_count,), dtype=np.float32)
    
    for idx in range(seq_count):
        end_idx = idx + buf
        eda_seq[idx] = subj_dict["eda"][idx:end_idx]
        bvp_seq[idx] = subj_dict["bvp"][idx:end_idx]
        acc_seq[idx] = subj_dict["acc"][idx:end_idx]
        temp_seq[idx] = subj_dict["temp"][idx:end_idx]
        spectral_seq[idx] = subj_dict["spectral"][idx:end_idx]
        sqi_seq[idx] = subj_dict["sqi"][idx:end_idx]
        # Target label corresponds to the latest window in the buffer (right now at time t)
        y_seq[idx] = subj_dict["y"][end_idx - 1]
        scores_seq[idx] = subj_dict["scores"][end_idx - 1]
        
    raw_dict = {
        "eda": eda_seq,
        "bvp": bvp_seq,
        "acc": acc_seq,
        "temp": temp_seq,
        "spectral": spectral_seq,
        "sqi": sqi_seq,
    }
    return raw_dict, y_seq, scores_seq, np.full((seq_count,), subj_dict["sid"], dtype=np.int64)


def build_pooled_dataset(subjects_data: list[dict], config: TFTransformerConfig, exclude_sids: set[int] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Combine rolling sequences across multiple subjects into pooled arrays."""
    pooled_raw = {
        "eda": [], "bvp": [], "acc": [], "temp": [], "spectral": [], "sqi": []
    }
    pooled_y = []
    pooled_sids = []
    
    for subj in subjects_data:
        if exclude_sids and subj["sid"] in exclude_sids:
            continue
        raw_seq, y_seq, _, sids_seq = create_rolling_sequences(subj, config)
        if len(y_seq) == 0:
            continue
        for k in pooled_raw:
            pooled_raw[k].append(raw_seq[k])
        pooled_y.append(y_seq)
        pooled_sids.append(sids_seq)
        
    if len(pooled_y) == 0:
        return {}, np.array([]), np.array([])
        
    final_raw = {k: np.concatenate(v, axis=0) for k, v in pooled_raw.items()}
    final_y = np.concatenate(pooled_y, axis=0)
    final_sids = np.concatenate(pooled_sids, axis=0)
    return final_raw, final_y, final_sids


class TFSequenceDataset(Dataset):
    """PyTorch Dataset yielding (raw_batch, spectral_batch, sqi_mask, y) for TF-Transformer."""
    def __init__(self, raw_dict: dict[str, np.ndarray], y: np.ndarray) -> None:
        self.eda = torch.tensor(raw_dict["eda"], dtype=torch.float32)
        self.bvp = torch.tensor(raw_dict["bvp"], dtype=torch.float32)
        self.acc = torch.tensor(raw_dict["acc"], dtype=torch.float32)
        self.temp = torch.tensor(raw_dict["temp"], dtype=torch.float32)
        self.spectral = torch.tensor(raw_dict["spectral"], dtype=torch.float32)
        self.sqi = torch.tensor(raw_dict["sqi"], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_batch = {
            "eda": self.eda[idx],
            "bvp": self.bvp[idx],
            "acc": self.acc[idx],
            "temp": self.temp[idx],
        }
        return raw_batch, self.spectral[idx], self.sqi[idx], self.y[idx]
