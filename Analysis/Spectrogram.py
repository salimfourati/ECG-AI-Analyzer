import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np

def plot_spectrogram(ecg, fs, nperseg=512, noverlap=256, fmax=40):
    """
    Trace un spectrogramme optimisé pour signaux ECG.

    - Axe X : temps en secondes
    - Axe Y : fréquence en Hz
    - Couleurs : énergie (en dB)
    - fmax par défaut = 40 Hz (zone utile pour l'ECG)
    """

    # Calcul du spectrogramme
    f, t, Sxx = spectrogram(
        ecg,
        fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude"
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    # Affichage en dB avec une colormap intuitive pour un pro de santé
    pcm = ax.pcolormesh(
        t, f, 10*np.log10(Sxx + 1e-12),
        shading='gouraud',
        cmap="RdBu_r"  # bleu = faible, rouge = fort
    )

    # Limiter aux fréquences utiles
    ax.set_ylim(0, fmax)
    ax.set_ylabel("Fréquence [Hz]")
    ax.set_xlabel("Temps [s]")
    ax.set_title(f"Spectrogramme ECG (0–{fmax} Hz)")

    # Ajouter barre de couleur
    fig.colorbar(pcm, ax=ax, label="Énergie (dB)")

    # Annotations physiologiques
    ax.axhline(0.3, color="green", linestyle="--", lw=1)
    ax.text(1, 0.35, "Respiration (~0.2–0.4 Hz)", color="green")

    ax.axhline(20, color="purple", linestyle="--", lw=1)
    ax.text(1, 21, "Bruit musculaire >20 Hz", color="purple")

    fig.tight_layout()
    return fig
