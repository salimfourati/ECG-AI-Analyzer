Pipeline de traitement
======================

Le fonctionnement général du projet suit les étapes suivantes :

1. Chargement des données ECG depuis les fichiers patients.
2. Sélection du lead à analyser.
3. Détection des pics R.
4. Calcul des métriques HRV.
5. Visualisation du spectrogramme.
6. Prédiction IA à partir du signal ECG.
7. Export des résultats en Excel ou PDF.

Modules principaux impliqués
----------------------------

- ``Analysis.preprocessing`` : chargement et préparation des signaux
- ``Analysis.detection`` : détection des pics R et fréquence cardiaque
- ``Analysis.hrv`` : calcul des métriques HRV
- ``Analysis.Spectrogram`` : analyse fréquentielle
- ``model`` : architecture du modèle IA
- ``utils.export`` : génération des exports