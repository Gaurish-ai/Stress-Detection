"""
Evaluation and Visualization Module for Approach 3 (CNN + BiLSTM + Attention).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_predictions(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for r in all_results:
        all_y_true.extend(r["y_true"])
        all_y_pred.extend(r["y_pred"])
        all_y_prob.extend(r["y_prob"])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except Exception:
        overall_auc = 0.5

    cm = confusion_matrix(all_y_true, all_y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / max(tn + fp, 1)
    else:
        specificity = 0.0

    subj_accs = [r["accuracy"] for r in all_results]
    subj_f1s = [r["f1"] for r in all_results]

    overall_summary = {
        "overall_accuracy": float(overall_acc),
        "overall_f1": float(overall_f1),
        "overall_precision": float(overall_prec),
        "overall_recall": float(overall_rec),
        "overall_specificity": float(specificity),
        "overall_roc_auc": float(overall_auc),
        "mean_subject_accuracy": float(np.mean(subj_accs)),
        "std_subject_accuracy": float(np.std(subj_accs)),
        "mean_subject_f1": float(np.mean(subj_f1s)),
        "std_subject_f1": float(np.std(subj_f1s)),
        "total_samples": int(len(all_y_true)),
        "total_stressed": int(all_y_true.sum()),
        "total_not_stressed": int((1 - all_y_true).sum()),
    }

    print("\n" + "=" * 62)
    print("OVERALL RESULTS (LOSO Cross-Validation - CNN + BiLSTM + Attention)")
    print("=" * 62)
    print(f"  Accuracy       : {overall_acc:.4f}")
    print(f"  F1-Score       : {overall_f1:.4f}")
    print(f"  Precision      : {overall_prec:.4f}")
    print(f"  Recall (Sens.) : {overall_rec:.4f}")
    print(f"  Specificity    : {specificity:.4f}")
    print(f"  ROC-AUC Score  : {overall_auc:.4f}")
    print(f"  Per-Subject Acc: {np.mean(subj_accs):.4f} +/- {np.std(subj_accs):.4f}")
    print(f"  Per-Subject F1 : {np.mean(subj_f1s):.4f} +/- {np.std(subj_f1s):.4f}")
    print("=" * 62)

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump({"overall": overall_summary, "per_subject": all_results}, fh, indent=2)
    print(f"Saved results summary to: {json_path}")

    plot_confusion_matrix(cm, output_dir)
    plot_roc_curve(all_y_true, all_y_prob, overall_auc, output_dir)
    plot_per_subject_bars(all_results, output_dir)

    return overall_summary


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Confusion Matrix -- CNN + BiLSTM + Attention")
    plt.colorbar()
    classes = ["Not Stressed", "Stressed"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, auc_val, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkgreen", lw=2, label=f"Sequence Model ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- CNN + BiLSTM + Attention")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()


def plot_per_subject_bars(all_results, output_dir):
    subjects = [f"P{r['subject']}" for r in all_results]
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["f1"] for r in all_results]

    x = np.arange(len(subjects))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, accs, width, label="Accuracy", color="#2ca02c")
    plt.bar(x + width/2, f1s, width, label="F1-Score", color="#d62728")
    plt.axhline(np.mean(accs), color="#2ca02c", linestyle="--", alpha=0.7, label=f"Mean Acc ({np.mean(accs):.2f})")
    plt.axhline(np.mean(f1s), color="#d62728", linestyle="--", alpha=0.7, label=f"Mean F1 ({np.mean(f1s):.2f})")

    plt.xlabel("Subject ID")
    plt.ylabel("Score")
    plt.title("Per-Subject Evaluation Results (LOSO Cross-Validation - Sequence Model)")
    plt.xticks(x, subjects, rotation=45)
    plt.ylim([0, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_subject_results.png"), dpi=300)
    plt.close()
