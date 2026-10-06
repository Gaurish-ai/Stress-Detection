"""
PDF Report Generator for Approach 5: Facial & Video Specialists.
Compiles head-to-head LOSO cross-validation metrics across the 18 common subjects.
"""
from __future__ import annotations
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch

from config import FacialSpecialistsConfig


def generate_pdf_report(config: FacialSpecialistsConfig) -> None:
    results_path = config.output_dir / "results.json"
    if not results_path.exists():
        print(f"Cannot generate report: {results_path} does not exist. Run train.py first.")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    pdf_path = config.output_dir / "approach5_facial_video_specialists_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )
    
    story = []
    story.append(Paragraph("Approach 5: Facial & Video Specialists Report", title_style))
    story.append(Paragraph(
        "This report summarizes the isolated, head-to-head Leave-One-Subject-Out (LOSO) cross-validation "
        "of 3 distinct visual dynamics models on the 11 continuous video features across 18 common subjects "
        "(P04, P05, P08, P09, P10, P13, P15, P19, P20, P21, P22, P23, P24, P25, P26, P29, P30, P31).",
        body_style
    ))
    story.append(Spacer(1, 0.15 * inch))
    
    story.append(Paragraph("Head-to-Head Performance Comparison", h2_style))
    
    s_5a = results["model_5a_xgboost"]
    s_5b = results["model_5b_bilstm"]
    s_5c = results["model_5c_transformer"]
    
    data = [
        ["Model Approach", "Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"],
        [
            "Model 5A: Statistical XGBoost",
            f"{s_5a['overall_accuracy']:.2%}",
            f"{s_5a['overall_f1']:.3f}",
            f"{s_5a['overall_precision']:.3f}",
            f"{s_5a['overall_recall']:.3f}",
            f"{s_5a['overall_roc_auc']:.3f}"
        ],
        [
            "Model 5B: BiLSTM + Attention",
            f"{s_5b['overall_accuracy']:.2%}",
            f"{s_5b['overall_f1']:.3f}",
            f"{s_5b['overall_precision']:.3f}",
            f"{s_5b['overall_recall']:.3f}",
            f"{s_5b['overall_roc_auc']:.3f}"
        ],
        [
            "Model 5C: Temporal Transformer",
            f"{s_5c['overall_accuracy']:.2%}",
            f"{s_5c['overall_f1']:.3f}",
            f"{s_5c['overall_precision']:.3f}",
            f"{s_5c['overall_recall']:.3f}",
            f"{s_5c['overall_roc_auc']:.3f}"
        ],
    ]
    
    table = Table(data, colWidths=[2.2 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    
    img_path = config.output_dir / "facial_models_comparison.png"
    if img_path.exists():
        story.append(Paragraph("Visual Comparison Across Metrics", h2_style))
        story.append(Image(str(img_path), width=6.5 * inch, height=2.0 * inch))
        story.append(Spacer(1, 0.15 * inch))
        
    story.append(Paragraph("Key Takeaway for Stage 2 Fusion", h2_style))
    story.append(Paragraph(
        "By isolating and training pure facial behavior first across 12 rolling windows (60 seconds), "
        "we prevent physiological signals from dominating gradient updates. The top-performing facial model "
        "identified here will be combined with our Model 4 (Dual-Stream TF-Transformer) in Stage 2 (`Model 6`) "
        "via late logit gating and intermediate cross-attention embeddings to achieve our 85%+ accuracy target.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated PDF report: {pdf_path}")


if __name__ == "__main__":
    generate_pdf_report(FacialSpecialistsConfig())
