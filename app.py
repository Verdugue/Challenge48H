"""
Application Streamlit pour la prédiction des prix de maisons.
Ce module implémente une interface utilisateur interactive permettant d'explorer les données,
de créer des features dérivées et de faire des prédictions de prix de maisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction des Prix de Maisons",
    page_icon="🏠",
    layout="wide"
)

# Titre et description de l'application
st.title("🏠 Prédiction des Prix de Maisons")
st.markdown("""
Cette application permet de prédire les prix des maisons en utilisant différentes caractéristiques.
Utilisez le menu de navigation à gauche pour explorer les différentes sections.
""")

# Fonction de chargement des données avec mise en cache
@st.cache_data
def load_data():
    """
    Charge le dataset des maisons depuis le fichier CSV.
    Utilise le décorateur @st.cache_data pour mettre en cache les données.
    
    Returns:
        pd.DataFrame: Dataset des maisons
    """
    df = pd.read_csv('train.csv')
    return df

# Chargement des données
df = load_data()

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une section:", ["Exploration des Données", "Feature Engineering", "Prédiction"])

# Section Exploration des Données
if page == "Exploration des Données":
    st.header("Exploration des Données")
    
    # Affichage des données brutes
    st.subheader("Données Brutes")
    st.dataframe(df.head())
    
    # Statistiques descriptives
    st.subheader("Statistiques Descriptives")
    st.dataframe(df.describe())
    
    # Distribution des prix
    st.subheader("Distribution des Prix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='SalePrice', bins=50, ax=ax)
    plt.title('Distribution des Prix de Vente')
    st.pyplot(fig)
    
    # Matrice de corrélation
    st.subheader("Corrélations avec le Prix")
    
    # Sélection des caractéristiques les plus importantes
    important_features = [
        'SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 
        'TotalBsmtSF', 'YearBuilt', '1stFlrSF', 'FullBath',
        'TotRmsAbvGrd', 'GarageCars'
    ]
    
    # Dictionnaire de traduction
    feature_names_fr = {
        'SalePrice': 'Prix de Vente',
        'OverallQual': 'Qualité Globale',
        'GrLivArea': 'Surface Habitable',
        'GarageArea': 'Surface Garage',
        'TotalBsmtSF': 'Surface Sous-sol',
        'YearBuilt': 'Année Construction',
        '1stFlrSF': 'Surface RDC',
        'FullBath': 'Salles de Bain',
        'TotRmsAbvGrd': 'Nombre de Pièces',
        'GarageCars': 'Places de Garage'
    }
    
    # Création de la matrice de corrélation
    corr_matrix = df[important_features].corr()
    
    # Renommage des index et colonnes
    corr_matrix.index = [feature_names_fr[col] for col in corr_matrix.index]
    corr_matrix.columns = [feature_names_fr[col] for col in corr_matrix.columns]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True,  # Afficher les valeurs
                cmap='coolwarm',  # Palette de couleurs
                center=0,  # Centre de la palette
                fmt='.2f',  # Format des nombres
                square=True,  # Cellules carrées
                ax=ax)
    plt.title('Matrice de Corrélation des Caractéristiques Principales')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    # Explication des corrélations
    st.write("""
    **Interprétation des corrélations:**
    - Une valeur proche de 1 (rouge) indique une forte corrélation positive
    - Une valeur proche de -1 (bleu) indique une forte corrélation négative
    - Une valeur proche de 0 (blanc) indique une faible corrélation
    
    **Observations principales:**
    - La Qualité Globale et la Surface Habitable sont les caractéristiques les plus corrélées avec le prix
    - La Surface du Garage et le nombre de Places de Garage sont fortement corrélés entre eux
    - L'Année de Construction a une influence positive sur le prix
    """)

# Section Feature Engineering
elif page == "Feature Engineering":
    st.header("Feature Engineering")
    
    # Création des features dérivées
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']  # Surface totale
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']  # Nombre total de salles de bain
    df['Age'] = df['YrSold'] - df['YearBuilt']  # Âge de la maison
    df['SFPerRoom'] = df['TotalSF'] / df['TotRmsAbvGrd']  # Surface par pièce
    df['OverallQualSF'] = df['OverallQual'] * df['TotalSF']  # Qualité globale pondérée
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['BsmtFinSF1'] / 100  # Nombre total de pièces
    df['GarageSFPerCar'] = df['GarageArea'] / df['GarageCars'].replace(0, 1)  # Surface du garage par voiture
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']  # Années depuis la dernière rénovation
    
    # Affichage des nouvelles features
    st.subheader("Nouvelles Features Créées")
    new_features = ['TotalSF', 'TotalBathrooms', 'Age', 'SFPerRoom', 
                    'OverallQualSF', 'TotalRooms', 'GarageSFPerCar', 'YearsSinceRemodel']
    st.dataframe(df[new_features].head())
    
    # Visualisation des relations avec le prix
    st.subheader("Relations avec le Prix")
    feature_to_plot = st.selectbox("Sélectionnez une feature:", new_features)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature_to_plot, y='SalePrice', ax=ax)
    plt.title(f'Relation entre {feature_to_plot} et Prix')
    st.pyplot(fig)

# Section Prédiction
else:
    st.header("Prédiction des Prix")
    st.write("""
    Cette section vous permet de prédire le prix d'une maison en trois étapes simples :
    1. Choisir le modèle de prédiction
    2. Configurer les paramètres principaux
    3. Entrer les caractéristiques de la maison
    """)
    
    # Préparation des données
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    # Liste des caractéristiques les plus importantes (prédéfinies)
    selected_features = [
        'OverallQual',    # Qualité globale
        'GrLivArea',      # Surface habitable
        'GarageArea',     # Surface du garage
        'TotalBsmtSF',    # Surface du sous-sol
        'YearBuilt',      # Année de construction
        '1stFlrSF',       # Surface du rez-de-chaussée
        'FullBath',       # Nombre de salles de bain
        'TotRmsAbvGrd'    # Nombre total de pièces
    ]
    
    # 1. Sélection du modèle (simplifié)
    st.subheader("1. Choix du Modèle")
    model_type = st.radio(
        "Choisissez votre modèle de prédiction:",
        ["Random Forest", "Gradient Boosting"],
        help="Random Forest est plus stable, Gradient Boosting peut être plus précis mais nécessite plus de réglages"
    )
    
    # 2. Configuration simple du modèle
    st.subheader("2. Configuration du Modèle")
    if model_type == "Random Forest":
        n_estimators = st.slider(
            "Nombre d'arbres (plus il y en a, plus le modèle est précis mais lent):",
            50, 300, 100
        )
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    else:
        learning_rate = st.slider(
            "Vitesse d'apprentissage (plus elle est basse, plus le modèle est stable):",
            0.01, 0.2, 0.1
        )
        model = GradientBoostingRegressor(learning_rate=learning_rate, random_state=42)
    
    # Préparation des données
    X = X[selected_features]
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Interface de prédiction
    st.subheader("3. Caractéristiques de la Maison")
    st.write("Entrez les informations de la maison dont vous souhaitez prédire le prix:")
    
    # Création des champs de saisie avec des descriptions claires
    input_data = {}
    feature_descriptions = {
        'OverallQual': 'Qualité globale de la maison (1-10)',
        'GrLivArea': 'Surface habitable (en pieds carrés)',
        'GarageArea': 'Surface du garage (en pieds carrés)',
        'TotalBsmtSF': 'Surface du sous-sol (en pieds carrés)',
        'YearBuilt': 'Année de construction',
        '1stFlrSF': 'Surface du rez-de-chaussée (en pieds carrés)',
        'FullBath': 'Nombre de salles de bain complètes',
        'TotRmsAbvGrd': 'Nombre total de pièces (hors sous-sol)'
    }
    
    # Création de deux colonnes pour les inputs
    col1, col2 = st.columns(2)
    for i, (feature, description) in enumerate(feature_descriptions.items()):
        with col1 if i < 4 else col2:
            input_data[feature] = st.number_input(
                f"{description}",
                value=float(df[feature].mean()),
                help=f"Moyenne: {df[feature].mean():.1f}, Min: {df[feature].min():.1f}, Max: {df[feature].max():.1f}"
            )
    
    # Bouton de prédiction
    if st.button("📊 Calculer le Prix"):
        with st.spinner("Calcul en cours..."):
            # Entraînement du modèle
            model.fit(X_train, y_train)
            
            # Préparation des données d'entrée
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Prédiction
            prediction = model.predict(input_scaled)[0]
            
            # Affichage du résultat
            st.success(f"💰 Prix estimé: ${prediction:,.2f}")
            
            # Score du modèle
            r2 = r2_score(y_test, model.predict(X_test))
            st.info(f"Fiabilité du modèle (R²): {r2:.2%}") 