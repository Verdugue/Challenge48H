# Documentation Technique - Prédiction des Prix de Maisons

## 1. Vue d'ensemble du Projet

### 1.1 Objectif
Le projet vise à prédire les prix des maisons en utilisant des techniques de machine learning. Il s'agit d'un challenge Kaggle de 48h visant à mettre en pratique les compétences en data science.

### 1.2 Structure du Projet
```
.
├── app.py                 # Application Streamlit principale
├── house_price_prediction.py  # Script de prédiction
├── requirements.txt       # Dépendances du projet
├── README.md             # Documentation générale
├── TECHNICAL_DOC.md      # Documentation technique
└── .gitignore           # Configuration Git
```

## 2. Chaîne de Traitement

### 2.1 Chargement et Exploration des Données
- Utilisation de pandas pour le chargement des données
- Analyse exploratoire des données (EDA)
- Visualisation des distributions et corrélations

### 2.2 Prétraitement des Données
- Gestion des valeurs manquantes avec SimpleImputer
- Standardisation des features numériques
- Feature engineering :
  - TotalSF = GrLivArea + TotalBsmtSF
  - TotalBathrooms = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
  - Age = YrSold - YearBuilt
  - SFPerRoom = TotalSF / TotRmsAbvGrd
  - OverallQualSF = OverallQual * TotalSF
  - TotalRooms = TotRmsAbvGrd + BsmtFinSF1/100
  - GarageSFPerCar = GarageArea / GarageCars
  - YearsSinceRemodel = YrSold - YearRemodAdd

### 2.3 Modèles Implémentés
1. **Random Forest**
   - Paramètres ajustables : n_estimators, max_depth
   - Avantages : Gestion des relations non linéaires, robuste aux outliers

2. **Gradient Boosting**
   - Paramètres ajustables : n_estimators, learning_rate
   - Avantages : Performance généralement supérieure au Random Forest

3. **XGBoost**
   - Paramètres ajustables : n_estimators, learning_rate, max_depth
   - Avantages : Performance optimisée, gestion efficace de la mémoire

4. **LightGBM**
   - Paramètres ajustables : n_estimators, learning_rate, max_depth
   - Avantages : Entraînement rapide, bonnes performances

### 2.4 Évaluation des Modèles
- Métriques utilisées :
  - RMSE (Root Mean Square Error)
  - R² Score
  - Validation croisée (5 folds)
- Visualisation de l'importance des features
- Intervalles de confiance pour les prédictions

## 3. Interface Utilisateur

### 3.1 Application Streamlit
- Navigation par sections :
  1. Exploration des Données
  2. Feature Engineering
  3. Prédiction
- Visualisations interactives
- Paramètres ajustables en temps réel

### 3.2 Fonctionnalités
- Exploration des données brutes
- Statistiques descriptives
- Visualisation des distributions
- Analyse des corrélations
- Création de features dérivées
- Entraînement de modèles
- Prédictions en temps réel

## 4. Performance et Résultats

### 4.1 Métriques de Performance
- Score R² : 0.8-0.9
- RMSE : Variable selon les paramètres
- Validation croisée : Scores stables

### 4.2 Features les Plus Importantes
1. OverallQual
2. GrLivArea
3. TotalBsmtSF
4. GarageArea
5. 1stFlrSF

## 5. Améliorations Futures

### 5.1 Optimisations Possibles
- Ajout de features catégorielles
- Optimisation des hyperparamètres
- Ensemble de modèles
- Réduction de dimensionnalité

### 5.2 Extensions
- Interface de comparaison de modèles
- Export des résultats
- API REST
- Dashboard de monitoring

## 6. Dépendances

### 6.1 Bibliothèques Principales
- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.1
- xgboost==2.0.3
- lightgbm==4.3.0

### 6.2 Visualisation
- matplotlib==3.8.3
- seaborn==0.13.2

## 7. Guide d'Installation et d'Utilisation

### 7.1 Installation
```bash
# Création de l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installation des dépendances
pip install -r requirements.txt
```

### 7.2 Lancement
```bash
streamlit run app.py
```

## 8. Contribution

### 8.1 Workflow Git
1. Fork du repository
2. Création d'une branche feature
3. Commit des changements
4. Push vers la branche
5. Création d'une Pull Request

### 8.2 Standards de Code
- PEP 8
- Documentation des fonctions
- Tests unitaires
- Gestion des erreurs 