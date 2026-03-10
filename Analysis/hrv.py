import numpy as np
import pandas as pd
import neurokit2 as nk


def _coerce_rpeaks(rpeaks):
    """
    Accepte soit:
      - un np.array/list d'indices d'échantillons,
      - un dict {'ECG_R_Peaks': ...} ou {'rpeaks': ...} ou similaire.
    Retourne un dict au format attendu par NeuroKit2.
    """
    if isinstance(rpeaks, (list, np.ndarray, pd.Series)):
        peaks = np.asarray(rpeaks, dtype=int)
        return {"ECG_R_Peaks": peaks}
    if isinstance(rpeaks, dict):
        for k in ["ECG_R_Peaks", "rpeaks", "peaks", "RPeaks", "R_Peaks"]:
            if k in rpeaks and rpeaks[k] is not None:
                return {"ECG_R_Peaks": np.asarray(rpeaks[k], dtype=int)}
    raise ValueError("Format de rpeaks non reconnu. Donne un array d'indices ou un dict avec la clé 'ECG_R_Peaks'.")


def compute_hrv(rpeaks, fs):
    """
    Calcule HRV (temps, fréquence, non-linéaire) puis renvoie:
      - full_df: DataFrame complet NeuroKit2
      - views: dict {section: DataFrame nettoyé} pour l’affichage Streamlit
      - summary: DataFrame 'Résumé' avec quelques métriques clés + interprétation simple
    """
    rpk = _coerce_rpeaks(rpeaks)

    # — Calculs NeuroKit2 —
    hrv_time = nk.hrv_time(rpk, sampling_rate=fs, show=False)
    # Méthode 'welch' robuste pour fenêtres courtes, harmonise les colonnes
    hrv_freq = nk.hrv_frequency(rpk, sampling_rate=fs, show=False)
    hrv_nonlin = nk.hrv_nonlinear(rpk, sampling_rate=fs, show=False)

    full_df = pd.concat([hrv_time, hrv_freq, hrv_nonlin], axis=1)

    # — Sélections par domaine —
    cols_time = [
        "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD",
        "HRV_pNN50", "HRV_TINN", "HRV_CVNN"
    ]
    cols_freq = [
        "HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF",
        "HRV_LFn", "HRV_HFn", "HRV_LFHF", "HRV_TotalPower"
    ]
    cols_nonlin = [
        "HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_SampEn", "HRV_ApEn"
    ]

    # — Renommage FR + unités —
    rename_map = {
        # Temps
        "HRV_MeanNN": "Mean NN (ms)",
        "HRV_SDNN": "SDNN (ms)",
        "HRV_RMSSD": "RMSSD (ms)",
        "HRV_SDSD": "SDSD (ms)",
        "HRV_pNN50": "pNN50 (%)",
        "HRV_TINN": "TINN (ms)",
        "HRV_CVNN": "CVNN (%)",
        # Fréquence (puissances en ms²)
        "HRV_ULF": "ULF (ms²)",
        "HRV_VLF": "VLF (ms²)",
        "HRV_LF": "LF (ms²)",
        "HRV_HF": "HF (ms²)",
        "HRV_LFn": "LF norm. (nu)",
        "HRV_HFn": "HF norm. (nu)",
        "HRV_LFHF": "LF/HF (ratio)",
        "HRV_TotalPower": "Puissance totale (ms²)",
        # Non-linéaire
        "HRV_SD1": "SD1 (ms)",
        "HRV_SD2": "SD2 (ms)",
        "HRV_SD1SD2": "SD1/SD2 (ratio)",
        "HRV_SampEn": "Sample Entropy",
        "HRV_ApEn": "Approximate Entropy",
    }

    def _clean(df, keep_cols):
        keep = [c for c in keep_cols if c in df.columns]
        out = df[keep].rename(columns=rename_map)
        # % sur pNN50 et CVNN ; reste ms ou ratios => arrondis adaptés
        for col in out.columns:
            if "(%)" in col or "(nu)" in col:
                out[col] = out[col].astype(float).round(1)
            elif "(ratio)" in col:
                out[col] = out[col].astype(float).round(2)
            else:
                out[col] = out[col].astype(float).round(2)
        return out

    time_view = _clean(full_df, cols_time)
    freq_view = _clean(full_df, cols_freq)
    nonlin_view = _clean(full_df, cols_nonlin)

    # — Résumé (quelques KPI lisibles) —
    key_cols = ["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LFHF"]
    summary = _clean(full_df, key_cols)

    # micro-interprétation (règles simples et indicatives)
    def _comment_row(row):
        sdnn = row.get("SDNN (ms)")
        rmssd = row.get("RMSSD (ms)")
        lf_hf = row.get("LF/HF (ratio)")
        hints = []

        # NB: seuils indicatifs pour enregistrements courts (5 min) — à contextualiser
        if pd.notna(sdnn):
            if sdnn < 50:
                hints.append("Variabilité faible (SDNN<50)")
            elif sdnn > 100:
                hints.append("Variabilité élevée (SDNN>100)")
        if pd.notna(rmssd):
            if rmssd < 20:
                hints.append("Ton parasympathique faible (RMSSD<20)")
            elif rmssd > 42:
                hints.append("Bon ton parasympathique (RMSSD>42)")
        if pd.notna(lf_hf):
            if lf_hf > 2:
                hints.append("Dominance sympathique (LF/HF>2)")
            elif lf_hf < 0.5:
                hints.append("Dominance parasympathique (LF/HF<0.5)")

        return " | ".join(hints) if hints else "—"

    summary["Interprétation (indicative)"] = summary.apply(_comment_row, axis=1)

    views = {
        "time": time_view,
        "freq": freq_view,
        "nonlin": nonlin_view,
    }
    return full_df, views, summary
