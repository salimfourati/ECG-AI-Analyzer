Modèle de classification
=================================

Ce module définit l’architecture de deep learning utilisée
pour la classification automatique des signaux ECG.

Architecture
------------

Le modèle combine :

- des couches CNN pour l’extraction de caractéristiques
- une couche BiLSTM pour la modélisation temporelle
- des couches fully connected pour la classification

Objectif
--------

Le modèle permet de prédire la classe diagnostique d’un signal ECG.

.. automodule:: model
   :members:
   :undoc-members:
   :show-inheritance: