Introduction
============

ECG-AI-Analyzer est un système d’analyse automatisée de signaux électrocardiographiques (ECG),
combinant traitement du signal, analyse physiologique et intelligence artificielle.

Objectif du projet
------------------

L’objectif est de proposer une plateforme complète permettant :

- l’exploration interactive de signaux ECG
- l’extraction d’indicateurs physiologiques
- l’analyse de la variabilité cardiaque
- l’interprétation automatique via un modèle d’apprentissage profond

Approche générale
-----------------

Le système repose sur un pipeline structuré :

1. Chargement des données ECG
2. Sélection du signal à analyser
3. Détection des battements cardiaques
4. Analyse HRV
5. Analyse spectrale
6. Prédiction par intelligence artificielle
7. Génération de rapports

Technologies utilisées
----------------------

- Python
- Streamlit (interface utilisateur)
- PyTorch (modèle IA)
- Plotly (visualisation)
- NeuroKit2 (traitement ECG)
- ReportLab (génération PDF)