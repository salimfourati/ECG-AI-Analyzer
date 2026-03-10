import pandas as pd
import numpy as np
import scipy.signal as signal
import wfdb

def load_signal(uploaded_file):
    df = pd.read_csv(uploaded_file)
    ecg = df.iloc[:, 1].values  # Supposé : signal en 2ᵉ colonne
    fs = 250  # Hz – à adapter si besoin
    ecg = bandpass_filter(ecg, fs)
    ecg = normalize_signal(ecg)
    return ecg, fs

def bandpass_filter(ecg, fs, lowcut=0.5, highcut=40):
    nyq = 0.5 * fs
    b, a = signal.butter(3, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, ecg)

def normalize_signal(ecg):
    return (ecg - np.mean(ecg)) / np.std(ecg)

def load_ptb_record(record_path: str, lead: str = "i"):
    record = wfdb.rdrecord(record_path)
    available_leads = [s.lower() for s in record.sig_name]

    if lead.lower() not in available_leads:
        raise ValueError(f"Lead {lead} not found. Available leads: {available_leads}")

    idx = available_leads.index(lead.lower())
    signal_mv = record.p_signal[:, idx]  # données déjà en millivolts
    fs = record.fs
    signal_mv = bandpass_filter(signal_mv, fs)

    return signal_mv, fs, record  # ✅ retour complet

def load_all_leads(record_path: str):
    """
    Charge tous les leads disponibles pour un enregistrement PTB.
    Retourne : matrice (N, n_leads), fréquence d'échantillonnage, et noms des leads
    """
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    signal_matrix = record.p_signal  # forme (N, n_leads)
    lead_names = record.sig_name
    return signal_matrix, fs, lead_names

def extract_metadata_from_hea(hea_path: str):
    metadata = {}
    with open(hea_path, "r") as f:
        for line in f:
            if line.startswith("#") and ':' in line:
                key, value = line[1:].split(":", 1)
                metadata[key.strip()] = value.strip()
    return metadata

def notch_filter(ecg, fs, freq=50, Q=30):
    # IIR notch
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(w0=freq/(fs/2), Q=Q)
    return filtfilt(b, a, ecg)

def detrend(ecg):
    from scipy.signal import detrend as sp_detrend
    return sp_detrend(ecg, type='linear')
