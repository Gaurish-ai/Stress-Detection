"""
Evaluation -- Metrics, visualizations, and result persistence.

Generates:
    - confusion_matrix.png
    - feature_importance.png
    - per_subject_results.png
    - stress_distribution.png
    - results.json
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
)


# ============================================================
# Console Output
# ============================================================
def print_overall_results(results):
    """Print aggregated LOSO cross-validation results to console."""
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]
    per_subject = results["per_subject"]

    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS (LOSO Cross-Validation)")
    print(f"{'=' * 60}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    print(f"\n  Total samples  : {len(y_true)}")
    print(f"  Stressed       : {int(y_true.sum())} ({y_true.mean():.1%})")
    print(f"  Not-stressed   : {int((1 - y_true).sum())} ({(1 - y_true).mean():.1%})")

    # Per-subject averages
    accs = [r["accuracy"] for r in per_subject]
    f1s = [r["f1"] for r in per_subject]
    print(f"\n  Per-subject Accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Per-subject F1       : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Not Stressed', 'Stressed'], zero_division=0)}")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auc": auc}


# ============================================================
# Plots
# ============================================================
def plot_confusion_matrix(results, output_dir):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(results["y_true"], results["y_pred"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Stressed", "Stressed"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix -- LOSO Cross-Validation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [OK] Saved {path}")


def plot_feature_importance(results, output_dir, top_n=20):
    """Save top-N feature importance bar chart as PNG."""
    importances = results["feature_importances"]
    names = results["feature_names"]

    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(indices))
    ax.barh(
        y_pos,
        importances[indices][::-1],
        color="#4A90D9",
        edgecolor="white",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in indices[::-1]], fontsize=10)
    ax.set_xlabel("Feature Importance (avg across LOSO folds)", fontsize=12)
    ax.set_title(f"Top {top_n} Features -- XGBoost Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [OK] Saved {path}")


def plot_per_subject_results(results, output_dir):
    """Save per-subject accuracy and F1 grouped bar chart as PNG."""
    per_subject = results["per_subject"]
    subjects = [r["subject"] for r in per_subject]
    accs = [r["accuracy"] for r in per_subject]
    f1s = [r["f1"] for r in per_subject]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(subjects))
    w = 0.35

    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4A90D9", alpha=0.85)
    ax.bar(x + w / 2, f1s, w, label="F1-Score", color="#E8636F", alpha=0.85)

    ax.axhline(np.mean(accs), color="#4A90D9", ls="--", alpha=0.5)
    ax.axhline(np.mean(f1s), color="#E8636F", ls="--", alpha=0.5)

    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Subject Performance -- LOSO Cross-Validation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in subjects], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_subject_results.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [OK] Saved {path}")


def plot_roc_curve(results, output_dir):
    """Save ROC curve as PNG."""
    y_true = results["y_true"]
    y_prob = results["y_prob"]

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        print("    [!] Cannot plot ROC curve (single class in y_true)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#4A90D9", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve -- LOSO Cross-Validation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [OK] Saved {path}")


def plot_stress_distribution(stress_scores, y, output_dir):
    """Save histogram of stress score distribution as PNG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(stress_scores[y == 0], bins=30, alpha=0.7, color="#4A90D9", label="Not Stressed", edgecolor="white")
    ax.hist(stress_scores[y == 1], bins=30, alpha=0.7, color="#E8636F", label="Stressed", edgecolor="white")
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Threshold (0.5)")
    ax.set_xlabel("Composite Stress Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Stress Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "stress_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    [OK] Saved {path}")


# ============================================================
# Save Results
# ============================================================
def save_results(results, output_dir, overall_metrics=None):
    """Save all numerical results to JSON."""
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    output = {
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        },
        "per_subject": results["per_subject"],
        "feature_importance": {
            name: float(imp)
            for name, imp in zip(results["feature_names"], results["feature_importances"])
        },
        "dataset_stats": {
            "total_samples": len(y_true),
            "stressed": int(y_true.sum()),
            "not_stressed": int((1 - y_true).sum()),
            "stress_ratio": float(y_true.mean()),
        },
    }

    if overall_metrics:
        output["overall"].update({k: float(v) for k, v in overall_metrics.items() if v == v})  # skip NaN

    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    [OK] Saved {path}")
