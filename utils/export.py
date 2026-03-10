import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def to_excel(hrv_df, output_filename="resultats_hrv.xlsx"):
    tmpdir = tempfile.mkdtemp()
    xlsx_path = os.path.join(tmpdir, output_filename)
    hrv_df.to_excel(xlsx_path, index=False)
    return xlsx_path


def to_pdf(hrv_df, meta: dict, signal_preview=None, rr_series=None, psd_fig=None,
              hr_mean=None, anomaly_msg=None, 
              ai_pred=None, ai_probas=None, class_mapping=None,
              output_filename="rapport_ecg.pdf"):
    """
    Génère un rapport PDF (ReportLab) :
    - PDF: Titre, métadonnées, fréquence cardiaque, anomalies, HRV, figures, diagnostic IA
    """
    print(">>> Nouvelle fonction to_pdf chargée depuis :", __file__)

    tmpdir = tempfile.mkdtemp()
    xlsx_path = os.path.join(tmpdir, "resultats_hrv.xlsx")
    pdf_path = os.path.join(tmpdir, output_filename)

    # 1) Export Excel
    hrv_df.to_excel(xlsx_path, index=False)

    # 2) Sauvegarde des figures en PNG
    sig_png = rr_png = psd_png = None

    if signal_preview is not None:
        sig, fs = signal_preview
        t = np.arange(len(sig)) / fs
        plt.figure(figsize=(5, 2))
        plt.plot(t, sig, lw=0.8)
        plt.xlabel("Temps (s)")
        plt.ylabel("mV")
        plt.title("Aperçu Lead")
        sig_png = os.path.join(tmpdir, "signal.png")
        plt.tight_layout()
        plt.savefig(sig_png, dpi=150)
        plt.close()

    if rr_series is not None:
        rr_t, rr_ms = rr_series
        plt.figure(figsize=(5, 2))
        plt.plot(rr_t, rr_ms, lw=0.8, marker='o', ms=2)
        plt.xlabel("Temps (s)")
        plt.ylabel("RR (ms)")
        plt.title("Tachogramme")
        rr_png = os.path.join(tmpdir, "rr.png")
        plt.tight_layout()
        plt.savefig(rr_png, dpi=150)
        plt.close()

    if psd_fig is not None:
        psd_png = os.path.join(tmpdir, "psd.png")
        psd_fig.savefig(psd_png, dpi=150)

    # 3) Création du PDF avec ReportLab Platypus
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Titre
    story.append(Paragraph("<b>Rapport d'analyse ECG</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    # Date
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Métadonnées patient
    if meta:
        story.append(Paragraph("<b>Informations patient</b>", styles["Heading2"]))
        for k, v in meta.items():
            story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        story.append(Spacer(1, 12))

    # Résumé rythme cardiaque
    story.append(Paragraph("<b>Résumé fréquence cardiaque</b>", styles["Heading2"]))
    if hr_mean is not None:
        story.append(Paragraph(f"Fréquence cardiaque moyenne : {hr_mean:.1f} bpm", styles["Normal"]))
    if anomaly_msg is not None:
        story.append(Paragraph(f"Analyse simple : {anomaly_msg}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Figures
    for img in [sig_png, rr_png, psd_png]:
        if img and os.path.exists(img):
            story.append(Image(img, width=400, height=150))
            story.append(Spacer(1, 12))

    # Métriques HRV sous forme de tableau
    story.append(Paragraph("<b>Métriques HRV</b>", styles["Heading2"]))
    data = [["Métrique", "Valeur"]]
    for col in hrv_df.columns:
        val = hrv_df[col].values[0]
        data.append([col, f"{val:.2f}" if isinstance(val, (int, float)) else str(val)])

    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # 🔮 Diagnostic IA
    if ai_pred is not None and class_mapping is not None:
        story.append(Paragraph("<b>Diagnostic IA</b>", styles["Heading2"]))
        pred_label = class_mapping.get(ai_pred, str(ai_pred))
        story.append(Paragraph(f"Prédiction principale : <b>{pred_label}</b>", styles["Normal"]))

        # Probabilités par classe
        if ai_probas is not None:
            data = [["Classe", "Probabilité (%)"]]
            for i, p in enumerate(ai_probas):
                label = class_mapping.get(i, f"Classe {i}")
                data.append([label, f"{p*100:.2f}"])
            table_ai = Table(data, colWidths=[250, 150])
            table_ai.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
            ]))
            story.append(table_ai)

        # Interprétation simplifiée
        interpretations = {
            "Infarctus du myocarde": "⚠️ Anomalies compatibles avec un infarctus du myocarde.",
            "Sujet sain": "✅ Signal normal, aucun signe pathologique détecté.",
            "Insuffisance cardiaque": "⚠️ Anomalies compatibles avec une insuffisance cardiaque.",
            "Bloc de branche": "⚠️ Bloc de branche détecté, conduction électrique anormale.",
            "Dysrythmie": "⚠️ Anomalies du rythme cardiaque.",
            "Autres anomalies (Myocardite / Valvulopathie / Hypertrophie / Divers)": "ℹ️ Anomalie détectée, nécessite analyse approfondie."
        }
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Interprétation : {interpretations.get(pred_label, 'Résultat à interpréter.')}", styles["Normal"]))

    # Sauvegarde finale
    doc.build(story)

    return pdf_path
