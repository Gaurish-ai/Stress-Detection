"""
Generate a PDF version of the implementation plan.
Usage: python generate_pdf.py
"""
from fpdf import FPDF
import os


def main():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Sensor Stress Detection", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Implementation Plan", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # --- Section: Project Goal ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Project Goal", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Build a stress detection system using physiological sensor data from "
        "the Empatica E4 wristband, mapped to physical hardware sensors: "
        "MAX30102 (HR), DS18B20 (TEMP), GSR (EDA), MPU6500 (ACC).")
    pdf.ln(4)

    # --- Section: Dataset ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Dataset", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in [
        "- 28 subjects with E4 wristband data (debate recordings, ~10-15 min each)",
        "- 4 sensors used: EDA (4Hz), HR (1Hz), ACC-xyz (32Hz), TEMP (4Hz)",
        "- Annotations every 5 seconds: arousal, valence, nervous, angry (1-5 scale)",
        "- Annotation source: Aggregated external annotations (consensus of 5 raters)",
    ]:
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: Stress Label ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Stress Label: Composite Score (Option A)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 6, "score = arousal*0.35 + (6-valence)*0.35 + nervous*0.20 + angry*0.10",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "normalized = (score - 1) / 4     [range 0 to 1]",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "stress = 1 if normalized > 0.5, else 0",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: Sensor Mapping ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "4. Physical Sensor Mapping", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_w = 55
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_w, 7, "Hardware Sensor", border=1, align="C")
    pdf.cell(col_w, 7, "Dataset Signal", border=1, align="C")
    pdf.cell(col_w, 7, "Sampling Rate", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    for hw, sig, rate in [("GSR", "EDA", "4 Hz"), ("MAX30102", "HR", "1 Hz"),
                           ("MPU6500", "ACC (x,y,z)", "32 Hz"), ("DS18B20", "TEMP", "4 Hz")]:
        pdf.cell(col_w, 7, hw, border=1, align="C")
        pdf.cell(col_w, 7, sig, border=1, align="C")
        pdf.cell(col_w, 7, rate, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: Model ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "5. Model: XGBoost Baseline with LOSO CV", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    steps = [
        "1. Load sensor data + annotations for 28 subjects",
        "2. Align to 5-second windows using startTime from subjects.csv",
        "3. Extract 34 handcrafted features per window:",
        "     EDA: mean, std, min, max, range, slope, peaks, AUC, kurtosis, skew (10)",
        "     HR:  mean, std, min, max, range, delta (6)",
        "     ACC: per-axis mean/std, magnitude stats, jerk, SMA (12)",
        "     TEMP: mean, std, min, max, slope, range (6)",
        "4. Train XGBoost with Leave-One-Subject-Out cross-validation",
        "5. Evaluate: accuracy, F1, precision, recall, ROC-AUC",
    ]
    for s in steps:
        pdf.cell(0, 6, s, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: Features Table ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "6. Feature Details (34 Total)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    feat_col = 40
    desc_col = 65
    rel_col = 60

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(feat_col, 7, "Feature", border=1, align="C")
    pdf.cell(desc_col, 7, "Description", border=1, align="C")
    pdf.cell(rel_col, 7, "Stress Relevance", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8)
    features = [
        ("eda_mean", "Avg skin conductance", "High = high arousal"),
        ("eda_std", "Variability", "High = fluctuating arousal"),
        ("eda_slope", "Linear trend", "Rising = increasing stress"),
        ("eda_num_peaks", "SCR count", "More peaks = stress responses"),
        ("eda_auc", "Area under curve", "Total activation level"),
        ("hr_mean", "Average heart rate", "High HR = stress/arousal"),
        ("hr_std", "HR variability", "Low HRV = stress"),
        ("hr_delta", "HR change in window", "Rising = acute stress"),
        ("acc_mag_mean", "Motion magnitude", "Fidgeting = nervousness"),
        ("acc_jerk_mean", "Motion abruptness", "Sudden moves = anxiety"),
        ("temp_mean", "Skin temperature", "Cooling = vasoconstriction"),
        ("temp_slope", "Temp trend", "Dropping = stress onset"),
    ]
    for feat, desc, rel in features:
        pdf.cell(feat_col, 6, feat, border=1)
        pdf.cell(desc_col, 6, desc, border=1)
        pdf.cell(rel_col, 6, rel, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: File Structure ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "7. File Structure", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", "", 9)
    structure = [
        "gemini model/",
        "  implementation_plan.md",
        "  implementation_plan.pdf",
        "  first model(baseline model)/",
        "    config.py              - Hyperparameters & paths",
        "    data_loader.py         - Load E4 CSVs + annotations",
        "    feature_engineering.py - 34 handcrafted features",
        "    stress_labels.py       - Composite stress score",
        "    train.py               - XGBoost + LOSO CV pipeline",
        "    evaluate.py            - Metrics, plots, results",
        "    outputs/               - Saved results & plots",
    ]
    for line in structure:
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Section: Future ---
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "8. Future Phases", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "- Phase 2: Deep Learning (CNN + BiLSTM)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "- Phase 3: Multimodal (sensor + facial video fusion)", new_x="LMARGIN", new_y="NEXT")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "implementation_plan.pdf")
    pdf.output(out_path)
    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    main()
