# Prédiction des Prix de Maisons

Application de prédiction des prix de maisons utilisant le dataset Kaggle "House Prices" avec une interface Streamlit.

## Description

Cette application permet de :
- Explorer les données du dataset House Prices
- Créer des features dérivées
- Entraîner et évaluer différents modèles de machine learning
- Faire des prédictions sur de nouvelles maisons

## Installation

1. Cloner le repository
2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```
3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Lancer l'application :
```bash
streamlit run app.py
```

## Structure du Projet

- `app.py` : Application principale Streamlit
- `requirements.txt` : Dépendances du projet
- `README.md` : Documentation

## Fonctionnalités

### 1. Exploration des Données
- Visualisation des données brutes
- Statistiques descriptives
- Distribution des prix
- Analyse des corrélations

### 2. Feature Engineering
- Création de features dérivées
- Gestion des valeurs manquantes
- Standardisation des données

### 3. Modèles de Prédiction
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 4. Évaluation
- RMSE
- R² Score
- Importance des features
- Validation croisée

## Performance

Le modèle actuel atteint un score R² de 0.8-0.9 sur l'ensemble de test.

## Auteur

[Votre nom]

## Licence

MIT 