# analysis/detection.py
import neurokit2 as nk
import numpy as np
from scipy.signal import medfilt

def detect_r_peaks(ecg, fs):
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return rpeaks['ECG_R_Peaks']

def compute_hr_from_rpeaks(r_peaks, fs):
    rr = np.diff(r_peaks) / fs  # intervalles en secondes
    
    # garde seulement les intervalles plausibles pour adultes (30–200 bpm)
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    
    if rr.size == 0:
        return np.nan, np.array([])
    
    hr_inst = 60.0 / rr  # bpm
    
    # lissage facultatif
    if hr_inst.size >= 5:
        hr_inst_smooth = medfilt(hr_inst, kernel_size=5)
    else:
        hr_inst_smooth = hr_inst
    
    hr_mean = float(np.nanmean(hr_inst_smooth))
    return hr_mean, hr_inst_smooth


def detect_tachy_brady(hr_mean, low=60, high=90):
    if np.isnan(hr_mean):
        return "⚠️ Impossible de calculer la FC (pics R insuffisants)"
    if hr_mean < low:
        return "⚠️ Bradycardie détectée"
    if hr_mean > high:
        return "⚠️ Tachycardie détectée"
    return "✅ Rythme cardiaque normal"
