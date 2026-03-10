import streamlit as st
import os , warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ⚠️ Nettoyer les warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from Analysis import preprocessing, detection, hrv, Spectrogram
from utils import plot , export
import importlib
importlib.reload(export)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import CNN_BiLSTM_ECG
import plotly.express as px


st.set_page_config(page_title="Analyseur ECG", layout="wide")

css_path = os.path.join("Styles", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Analyse intelligente de signaux ECG (PTB Database)")


# --- Fonction de chargement modèle ---
@st.cache_resource
def load_ai_model():
    model = CNN_BiLSTM_ECG(num_classes=7) 
    MODEL_PATH = os.path.join("ML", "best_model_targeted.pth")
    print(">>> Chargement du modèle depuis :", MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

ai_model = load_ai_model()

# 📁 Base path vers tes dossiers patients
BASE_PATH = r"C:/Users/user/ecg-database1.0.0"

# 1️⃣ Choix du dossier patient
patients = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])
patient_selected = st.sidebar.selectbox("Choisir un patient", patients)

if patient_selected:
    patient_path = os.path.join(BASE_PATH, patient_selected)

    # 2️⃣ Choix du fichier .hea dans ce dossier
    hea_files = sorted([f for f in os.listdir(patient_path) if f.endswith(".hea")])
    hea_selected = st.sidebar.selectbox("Choisir un fichier .hea", hea_files)

    if "last_file" not in st.session_state:
        st.session_state.last_file = None
    if st.session_state.last_file != hea_selected:
        st.session_state.pred = None
        st.session_state.probas = None
        st.session_state.last_file = hea_selected

    if hea_selected:
        record_path = os.path.join(patient_path, hea_selected.replace(".hea", ""))

        try:
            # 🔄 Charger tous les leads disponibles
            all_signals, fs, lead_names = preprocessing.load_all_leads(record_path)

            st.subheader("Affichage de tous les leads disponibles")
            plot.plot_all_leads(all_signals, fs, lead_names)

            # 🔽 Sélection du lead
            default_idx = 0
            for i, name in enumerate(lead_names):
                if name.strip().upper() in ["II", "LEAD II", "2"]:
                    default_idx = i
                    break

            lead_selected = st.selectbox(
                "Choisir le lead pour l'analyse",
                options=range(len(lead_names)),
                format_func=lambda i: lead_names[i],
                index=default_idx
            )

            signal = all_signals[:, lead_selected]
            st.write(f"➡️ Analyse sur **Lead {lead_names[lead_selected]}**")


            # 🌟 Organisation en onglets
            tab_hrv, tab_spectro, tab_info, tab_ai, tab_export = st.tabs(
                ["📊 Analyse HRV", "📡 Spectrogramme", "ℹ️ Infos Patient", "🔮 Diagnostic IA", "📂 Export"]
            )
            # --- Onglet HRV ---
            with tab_hrv:
                with tab_hrv:
                    st.markdown("## 🧩 Qu’est-ce que la variabilité de la fréquence cardiaque (HRV) ?")
                    st.info("""
                    La **variabilité de la fréquence cardiaque (Heart Rate Variability)** correspond aux variations de durée entre deux battements successifs.
                    
                    - Un **HRV élevé** = meilleure adaptation au stress, bon équilibre sympathique/parasympathique.  
                    - Un **HRV faible** = fatigue, stress chronique, ou atteinte cardiovasculaire.  

                    ### Domaines d’analyse :
                    - **Temporel** : SDNN, RMSSD → reflètent la variabilité globale et parasympathique.  
                    - **Fréquentiel** : LF/HF → équilibre entre le système sympathique (stress) et parasympathique (repos).  
                    - **Non-linéaire** : entropie, SD1/SD2 → complexité du signal cardiaque.  

                    ⚠️ Les indices sont **indicatifs** et doivent toujours être interprétés dans un contexte clinique.
                    """)
                if st.button("Détecter les R-peaks sur le lead sélectionné"):
                    rpeaks = detection.detect_r_peaks(signal, fs)
                    st.subheader("R-peaks détectés")
                    plot.plot_with_rpeaks(signal, rpeaks, fs)

                    hr_mean, hr_inst = detection.compute_hr_from_rpeaks(rpeaks, fs)

                    if not np.isnan(hr_mean):
                        st.metric("Fréquence cardiaque moyenne", f"{hr_mean:.1f} bpm")
                    else:
                        st.warning("Impossible de calculer la fréquence cardiaque (pas assez de R-peaks).")

                    # 🚨 Détection tachycardie / bradycardie
                    msg = detection.detect_tachy_brady(hr_mean)
                    if "⚠️" in msg:
                        st.warning(msg)
                    else:
                        st.success(msg)

                    st.subheader("Analyse HRV")
                    full_df, views, summary = hrv.compute_hrv(rpeaks, fs)

                    # Sauvegarde dans session_state
                    st.session_state["summary"] = summary
                    st.session_state["hr_mean"] = hr_mean
                    st.session_state["msg"] = msg
                    st.session_state["rpeaks"] = rpeaks

                    subtab_resume, subtab_temps, subtab_freq, subtab_nonlin, subtab_legend = st.tabs(
                            ["Résumé", "Domaine temporel", "Domaine fréquentiel", "Non-linéaire", "Légende"]
                        )

                    with subtab_resume:
                            st.caption("📌 Métriques clés")
                            st.dataframe(summary, use_container_width=True)

                    with subtab_temps:
                            st.caption("🕒 Indices temporels")
                            st.dataframe(views["time"], use_container_width=True)

                    with subtab_freq:
                            st.caption("📶 Indices fréquentiels")
                            st.dataframe(views["freq"], use_container_width=True)

                    with subtab_nonlin:
                            st.caption("🧩 Indices non-linéaires")
                            st.dataframe(views["nonlin"], use_container_width=True)
                    with subtab_legend:
                            st.markdown("""**Légende**

                    Domaine temporel
                    - **Mean NN (ms)** : *Mean of Normal-to-Normal intervals* — moyenne des intervalles entre battements normaux.  
                    - **SDNN (ms)** : *Standard Deviation of NN intervals* — variabilité globale de la fréquence cardiaque.  
                    - **RMSSD (ms)** : *Root Mean Square of Successive Differences* — reflète l’activité parasympathique (court terme).  
                    - **SDSD (ms)** : *Standard Deviation of Successive Differences* — variabilité à court terme.  
                    - **pNN50 (%)** : *Percentage of NN intervals differing by >50 ms* — proportion d’intervalles très variables.  
                    - **TINN (ms)** : *Triangular Interpolation of NN Interval Histogram* — largeur de la distribution des intervalles.  
                    - **CVNN (%)** : *Coefficient of Variation of NN intervals* — variabilité relative.

                    Domaine fréquentiel
                    - **ULF (ms²)** : *Ultra Low Frequency power* — très basse fréquence (<0,003 Hz), reflète tendances longues.  
                    - **VLF (ms²)** : *Very Low Frequency power* — basse fréquence (0,003–0,04 Hz), composante lente.  
                    - **LF (ms²)** : *Low Frequency power* — basse fréquence (0,04–0,15 Hz), lié au sympathique + baroréflexe.  
                    - **HF (ms²)** : *High Frequency power* — haute fréquence (0,15–0,40 Hz), activité parasympathique (respiration).  
                    - **LFn (nu)** : *Normalized Low Frequency power* — LF rapporté à la puissance totale.  
                    - **HFn (nu)** : *Normalized High Frequency power* — HF rapporté à la puissance totale.  
                    - **LF/HF (ratio)** : *LF to HF ratio* — indicateur de l’équilibre sympatho-vagal.  
                    - **Puissance totale (ms²)** : somme ULF+VLF+LF+HF.

                    Domaine non-linéaire
                    - **SD1 (ms)** : *Poincaré SD1* — variabilité à court terme (dispersion transversale).  
                    - **SD2 (ms)** : *Poincaré SD2* — variabilité à long terme (dispersion longitudinale).  
                    - **SD1/SD2 (ratio)** : *SD1 divided by SD2* — équilibre court/long terme.  
                    - **Sample Entropy** : *Entropie de l’échantillon* — complexité et imprévisibilité de la série temporelle.  
                    - **Approximate Entropy** : *Entropie approximative* — mesure alternative de la complexité du signal.

                    ---

                    ⚠️ Remarque importantes :  
                    - Certaines valeurs (`None`) apparaissent quand la durée du signal est trop courte ou que le calcul n’est pas applicable (ex. ULF sur segments <5 minutes).  
                    - Les interprétations automatiques (SDNN bas = variabilité faible, LF/HF haut = dominance sympathique, etc.) sont **indicatives** et doivent être contextualisées.""")

                    # (optionnel) montrer le DataFrame brut pour debug
                    with st.expander("Voir la table complète"):
                        st.dataframe(full_df.round(3), use_container_width=True)


            # --- Onglet Export ---
            with tab_export:
                st.markdown("### 📂 Exporter les résultats")

                summary = st.session_state.get("summary", None)
                hr_mean = st.session_state.get("hr_mean", None)
                msg = st.session_state.get("msg", None)
                rpeaks = st.session_state.get("rpeaks", None)

                if summary is None:
                    st.warning("⚠️ Vous devez d’abord détecter les R-peaks dans l’onglet HRV avant d’exporter.")
                else:
                    # Export Excel
                    if st.button("📊 Exporter en Excel"):
                        xlsx_path = export.to_excel(summary)
                        st.success("✅ Fichier Excel généré")
                        with open(xlsx_path, "rb") as f:
                            st.download_button("⬇️ Télécharger le fichier Excel", f, file_name="resultats_hrv.xlsx")

                    # Export PDF
                    if st.button("📄 Exporter en PDF"):
                        pdf_path = export.to_pdf(
                            hrv_df=summary,
                            meta=preprocessing.extract_metadata_from_hea(record_path + ".hea"),
                            signal_preview=(signal, fs),
                            rr_series=(np.cumsum(np.diff(rpeaks))/fs, np.diff(rpeaks)*1000/fs) if len(rpeaks) > 1 else None,
                            hr_mean=hr_mean,
                            anomaly_msg=msg,
                            ai_pred=st.session_state.pred,
                            ai_probas=st.session_state.probas,
                            class_mapping={
                                0: "Infarctus du myocarde",
                                1: "Sujet sain",
                                2: "Inconnu",
                                3: "Insuffisance cardiaque",
                                4: "Bloc de branche",
                                5: "Dysrythmie",
                                6: "Autres anomalies (Myocardite / Valvulopathie / Hypertrophie / Divers)"
                            }
                        )
                        st.success("✅ Rapport PDF généré")
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇️ Télécharger le rapport PDF", f, file_name="rapport_ecg.pdf")

            # --- Onglet Spectrogramme ---
            with tab_spectro:
                if st.button("Afficher le spectrogramme"):
                    fig = Spectrogram.plot_spectrogram(signal, fs)
                    st.pyplot(fig)
                    st.markdown("### ℹ️ Interprétation du spectrogramme")
                    st.info("""
                            Le **spectrogramme ECG** représente l’évolution du contenu fréquentiel du signal au cours du temps :

                            - Axe horizontal (x) : le temps (secondes).  
                            - Axe vertical (y) : les fréquences présentes dans le signal (Hertz).  
                            - Couleurs : l’intensité (énergie) en décibels (dB).  

                            ### Zones d’intérêt clinique :
                            - **< 0,5 Hz** : composantes liées aux tendances lentes (variations globales).  
                            - **0,2–0,4 Hz** (bande verte) : zone typique de la respiration, reflétant l’activité parasympathique.  
                            - **0,5–40 Hz** : bande ECG utile, où se trouvent les ondes P, QRS et T.  
                            - **> 20 Hz** (bande violette) : souvent du bruit musculaire ou artefacts.  

                            ⚠️ Le spectrogramme est un **outil complémentaire** : il permet de visualiser la répartition énergétique du signal mais ne remplace pas l’analyse des intervalles RR ou des métriques HRV.
                            """)

            # --- Onglet Infos Patient ---
            with tab_info:
                meta = preprocessing.extract_metadata_from_hea(record_path + ".hea")
                mapping_fr = {
                    "age": "Âge",
                    "sex": "Sexe",
                    "Diagnose": "Diagnostic",
                    "Reason for admission": "Raison d’admission"
                }

                for key, label in mapping_fr.items():
                    if key in meta and meta[key] not in [None, "", "NA"]:  # seulement si valeur non vide
                        st.write(f"**{label}** : {meta[key]}")

            # --- Onglet Infos IA ---
            with tab_ai:
                st.markdown("## 🔮 Diagnostic basé sur le modèle IA")
                st.warning("ℹ️ Toutes les prédictions se basent donc sur **Lead II**")

                 # Initialiser session_state si vide
                if "pred" not in st.session_state:
                    st.session_state.pred = None
                if "probas" not in st.session_state:
                    st.session_state.probas = None

                # --- Boutons côte à côte ---
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("⚡ Prédiction IA (fichier sélectionné)"):
                        # Forcer Lead II
                        default_idx = 0
                        for i, name in enumerate(lead_names):
                            if name.strip().upper() in ["II", "LEAD II", "2"]:
                                default_idx = i
                                break

                        sig = all_signals[:, default_idx]
                        sig = sig[:10000] if len(sig) > 10000 else np.pad(sig, (0, 10000 - len(sig)))
                        x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                        with torch.no_grad():
                            out = ai_model(x)
                            probas = F.softmax(out, dim=1).cpu().numpy()[0]

                        pred = int(np.argmax(probas))
                        st.session_state.probas = probas
                        st.session_state.pred = pred

                with col2:
                    if len(hea_files) > 1:  # seulement si plusieurs ECG
                        if st.button("⚡ Diagnostic global (tous les ECG)"):
                            all_probas = []
                            for hea_file in hea_files:
                                record_path2 = os.path.join(patient_path, hea_file.replace(".hea", ""))
                                signals2, fs2, lead_names2 = preprocessing.load_all_leads(record_path2)

                                # Forcer Lead II
                                default_idx2 = 0
                                for i, name in enumerate(lead_names2):
                                    if name.strip().upper() in ["II", "LEAD II", "2"]:
                                        default_idx2 = i
                                        break

                                signal2 = signals2[:, default_idx2]
                                sig2 = signal2[:10000] if len(signal2) > 10000 else np.pad(signal2, (0, 10000 - len(signal2)))
                                x2 = torch.tensor(sig2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                                with torch.no_grad():
                                    out2 = ai_model(x2)
                                    probas2 = F.softmax(out2, dim=1).cpu().numpy()[0]
                                    all_probas.append(probas2)

                            # Moyenne des probabilités
                            avg_probas = np.mean(all_probas, axis=0)
                            pred_final = int(np.argmax(avg_probas))

                            st.session_state.probas = avg_probas
                            st.session_state.pred = pred_final

                # --- Affichage résultats ---
                if st.session_state.probas is not None:
                    probas = st.session_state.probas
                    pred = int(np.argmax(probas))

                    class_mapping_fr = {
                        0: "Infarctus du myocarde",
                        1: "Sujet sain",
                        2: "Inconnu",
                        3: "Insuffisance cardiaque",
                        4: "Bloc de branche",
                        5: "Dysrythmie",
                        6: "Autres anomalies (Myocardite / Valvulopathie / Hypertrophie / Divers)"
                    }

                    st.success(f"✅ Diagnostic prédit : **{class_mapping_fr[pred]}**")

                    interpretations = {
                        "Infarctus du myocarde": "⚠️ Anomalies compatibles avec un infarctus du myocarde.",
                        "Sujet sain": "✅ Signal normal, aucun signe pathologique détecté.",
                        "Insuffisance cardiaque": "⚠️ Anomalies compatibles avec une insuffisance cardiaque.",
                        "Bloc de branche": "⚠️ Bloc de branche détecté, conduction électrique anormale.",
                        "Dysrythmie": "⚠️ Anomalies du rythme cardiaque.",
                        "Autres anomalies (Myocardite / Valvulopathie / Hypertrophie / Divers)": "ℹ️ Anomalie détectée, nécessite analyse approfondie."
                    }
                    st.info(interpretations.get(class_mapping_fr[pred], "Résultat à interpréter dans un contexte clinique."))


                    # Bouton pour voir la distribution
                    if st.button("📊 Voir la distribution des probabilités"):
                        df_proba = pd.DataFrame({
                            "Classe": [class_mapping_fr[i] for i in range(len(class_mapping_fr))],
                            "Probabilité (%)": [round(p*100, 2) for p in probas]
                        })

                        fig = px.bar(
                            df_proba,
                            x="Probabilité (%)",
                            y="Classe",
                            orientation="h",
                            color="Classe",
                            text="Probabilité (%)",
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        fig.update_traces(textposition="outside")
                        fig.update_layout(
                            title="Distribution des probabilités par classe",
                            xaxis=dict(title="Probabilité (%)", range=[0, 100]),
                            yaxis=dict(title="Classe"),
                            showlegend=False,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")
