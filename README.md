# Analyseur intelligent de signaux ECG

Ce projet propose un système d’analyse automatisée de signaux électrocardiographiques (ECG) combinant traitement du signal, extraction d’indicateurs physiologiques et intelligence artificielle.

L’application permet de visualiser des enregistrements ECG, d’analyser la variabilité de la fréquence cardiaque (HRV), d’explorer le contenu fréquentiel du signal et d’obtenir une prédiction diagnostique basée sur un modèle d’apprentissage profond.

---

# Fonctionnalités principales

## Visualisation des signaux ECG

- Visualisation interactive de tous les leads ECG
- Sélection d’un lead spécifique pour l’analyse
- Affichage inspiré des tracés ECG cliniques

L’affichage interactif est réalisé avec Plotly afin de permettre une exploration intuitive des signaux.

---

## Détection des battements cardiaques (R-peaks)

L’application détecte automatiquement les pics R du signal ECG afin de calculer la fréquence cardiaque.

Les fonctionnalités incluent :

- détection automatique des R-peaks
- calcul de la fréquence cardiaque moyenne
- identification simple d’anomalies de rythme telles que :
  - bradycardie
  - tachycardie

La détection repose sur les méthodes proposées par la bibliothèque NeuroKit2. :contentReference[oaicite:0]{index=0}

---

## Analyse de la variabilité de la fréquence cardiaque (HRV)

Le système calcule différents indicateurs de HRV dans plusieurs domaines :

### Domaine temporel

- SDNN  
- RMSSD  
- pNN50  
- TINN  

### Domaine fréquentiel

- ULF  
- VLF  
- LF  
- HF  
- ratio LF/HF  

### Domaine non-linéaire

- SD1  
- SD2  
- entropie approximative  
- entropie d’échantillon  

Ces métriques sont calculées et structurées automatiquement afin de faciliter leur interprétation. :contentReference[oaicite:1]{index=1}

---

## Analyse spectrale du signal

Le projet permet également de visualiser un **spectrogramme ECG**, représentant l’évolution du contenu fréquentiel du signal dans le temps.

Cette représentation permet notamment d’identifier :

- les tendances lentes du signal  
- les composantes respiratoires  
- le bruit haute fréquence  

Le spectrogramme est obtenu à partir d’une transformation de Fourier à court terme. :contentReference[oaicite:2]{index=2}

---

## Classification automatique par intelligence artificielle

Le système intègre un modèle de deep learning destiné à la classification de signaux ECG.

L’architecture du modèle combine :

- des couches convolutionnelles (CNN) pour l’extraction de caractéristiques
- une couche **BiLSTM** pour capturer la dynamique temporelle du signal
- des couches fully connected pour la classification finale

Le modèle est défini dans le fichier `model.py`. :contentReference[oaicite:3]{index=3}

Le système peut prédire plusieurs catégories de diagnostics cardiaques, notamment :

- infarctus du myocarde  
- insuffisance cardiaque  
- bloc de branche  
- dysrythmie  
- sujet sain  
- autres anomalies cardiaques  

---

## Interface interactive

L’application est développée avec **Streamlit**, ce qui permet de fournir une interface web interactive pour l’analyse ECG.

Les fonctionnalités de l’interface incluent :

- visualisation des signaux  
- analyse HRV  
- génération de spectrogrammes  
- prédiction par IA  
- consultation des informations patient  

La logique principale de l’application est implémentée dans `app.py`. :contentReference[oaicite:4]{index=4}

---

## Génération de rapports

L’application permet d’exporter automatiquement les résultats de l’analyse sous différents formats :

- fichier **Excel** contenant les métriques HRV  
- **rapport PDF** comprenant :
  - informations patient  
  - aperçu du signal ECG  
  - indicateurs HRV  
  - résultats de la prédiction IA  

La génération de rapports est réalisée avec la bibliothèque ReportLab. :contentReference[oaicite:5]{index=5}

---

## Structure du projet

```
ECG-AI-Analyzer
│
├── Analysis
│   ├── preprocessing.py
│   ├── detection.py
│   ├── hrv.py
│   └── Spectrogram.py
│
├── utils
│   ├── plot.py
│   └── export.py
│
├── ML
│   ├── exploration_fixed.ipynb
│   └── best_model_targeted.pth
│
├── Styles
│   └── style.css
│
├── app.py
├── model.py
├── requirements.txt
└── resultats_hrv.xlsx
```

---

## Jeu de données

## Description des données

Le projet utilise la **PTB Diagnostic ECG Database**, une base de données publique disponible sur PhysioNet et largement utilisée pour la recherche en analyse de signaux ECG et en apprentissage automatique.

Cette base contient des enregistrements électrocardiographiques provenant à la fois de sujets sains et de patients atteints de différentes pathologies cardiaques.

### Caractéristiques principales

- **549 enregistrements ECG**
- **290 sujets** (âge : 17 à 87 ans)
- Chaque sujet possède **1 à 5 enregistrements**
- **15 signaux enregistrés simultanément**
  - 12 dérivations ECG standard  
  - 3 dérivations Frank (VX, VY, VZ)
- **Fréquence d’échantillonnage : 1000 Hz**
- **Résolution : 16 bits**

Les dérivations ECG standard incluent :

- I, II, III  
- aVR, aVL, aVF  
- V1, V2, V3, V4, V5, V6  

### Classes diagnostiques principales

La base contient plusieurs catégories de diagnostics, notamment :

- infarctus du myocarde  
- insuffisance cardiaque / cardiomyopathie  
- bloc de branche  
- dysrythmie  
- hypertrophie myocardique  
- maladies valvulaires  
- myocardite  
- sujets sains  

### Métadonnées cliniques

La plupart des enregistrements incluent des informations cliniques détaillées dans les fichiers `.hea`, telles que :

- âge
- sexe
- diagnostic
- historique médical
- traitements ou interventions

Ces informations peuvent être utilisées pour enrichir l’analyse ou pour développer des modèles d’intelligence artificielle.

### Source

PhysioNet – PTB Diagnostic ECG Database  
https://physionet.org/content/ptbdb/
---

## Installation

Cloner le projet :

```bash
git clone https://github.com/username/ecg-ai-analyzer
cd ecg-ai-analyzer
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```
Les bibliothèques nécessaires sont listées dans le fichier requirements.txt. 


Lancer l’application
```bash
streamlit run app.py
```
