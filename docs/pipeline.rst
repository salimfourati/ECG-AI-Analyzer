Pipeline de traitement ECG
==========================

Le système ECG-AI-Analyzer suit un pipeline structuré permettant de transformer
un signal ECG brut en une analyse interprétable et exploitable.

Vue globale
-----------

Le pipeline est composé des étapes suivantes :

1. Chargement des données ECG
2. Sélection du signal à analyser
3. Détection des R-peaks
4. Calcul des métriques HRV
5. Analyse spectrale
6. Prédiction IA
7. Export des résultats

Description des étapes
----------------------

Chargement des données
^^^^^^^^^^^^^^^^^^^^^^

Les signaux ECG sont chargés à partir de fichiers issus de la base PhysioNet.
Chaque enregistrement contient plusieurs leads.

Sélection du lead
^^^^^^^^^^^^^^^^^

L’utilisateur peut choisir un lead spécifique (souvent Lead II)
pour effectuer une analyse plus ciblée.

Détection des R-peaks
^^^^^^^^^^^^^^^^^^^^^

Les pics R sont détectés automatiquement afin d’identifier les battements cardiaques
et calculer les intervalles RR.

Analyse HRV
^^^^^^^^^^^

Les métriques HRV sont calculées dans différents domaines :

- temporel
- fréquentiel
- non-linéaire

Analyse spectrale
^^^^^^^^^^^^^^^^^

Un spectrogramme est généré afin d’étudier la distribution fréquentielle du signal.

Prédiction par intelligence artificielle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Le signal ECG est utilisé comme entrée d’un modèle CNN + BiLSTM
afin de prédire une classe diagnostique.

Export des résultats
^^^^^^^^^^^^^^^^^^^^

Les résultats peuvent être exportés sous forme :

- de fichier Excel
- de rapport PDF complet