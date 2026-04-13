Jeu de données
==============

Le projet repose sur la PTB Diagnostic ECG Database, une base de données publique
disponible sur PhysioNet et largement utilisée en recherche.

Cette base permet de travailler sur des signaux ECG réels,
incluant à la fois des sujets sains et des patients atteints de pathologies cardiaques.

Description générale
--------------------

Cette base contient des enregistrements ECG provenant :

- de sujets sains
- de patients atteints de pathologies cardiaques

Caractéristiques principales
----------------------------

- 549 enregistrements ECG
- 290 patients
- 1 à 5 enregistrements par patient
- 15 signaux simultanés
- fréquence : 1000 Hz
- résolution : 16 bits

Structure des signaux
---------------------

Chaque enregistrement inclut :

- 12 dérivations ECG standard :
  I, II, III, aVR, aVL, aVF, V1 à V6
- 3 dérivations Frank :
  VX, VY, VZ

Classes diagnostiques
---------------------

Les principales catégories incluent :

- infarctus du myocarde
- insuffisance cardiaque
- dysrythmie
- bloc de branche
- hypertrophie myocardique
- maladies valvulaires
- myocardite
- sujets sains

Métadonnées cliniques
---------------------

Les fichiers `.hea` contiennent :

- âge
- sexe
- diagnostic
- historique médical

Source
------

PhysioNet – PTB Diagnostic ECG Database  
https://physionet.org/content/ptbdb/