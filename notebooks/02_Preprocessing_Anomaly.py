"""
Preparation des donnees pour la detection d'anomalies
Garder pour l'entrainement uniquement les donnees normales ou le churn=0 
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, sauvegarder_donnees, sauvegarder_modele

CHEMIN_INTERIM = '../data/interim/donnees_explorees.csv'
CHEMIN_PROCESSED = '../data/processed/anomalie/'
CHEMIN_MODELES = '../models/'

os.makedirs(CHEMIN_PROCESSED, exist_ok=True)
os.makedirs(CHEMIN_MODELES, exist_ok=True)

def charger_et_nettoyer_donnees():
    """Charge et nettoie les donnees"""
    df = charger_donnees(CHEMIN_INTERIM)
    
    colonnes_a_supprimer = ['customer_id']
    colonnes_existantes = [col for col in colonnes_a_supprimer if col in df.columns]
    
    if colonnes_existantes:
        df = df.drop(columns=colonnes_existantes)
    
    return df


def encoder_variables_categorielles(df):
    """Encode les variables categorielles"""
    df_encode = df.copy()
    
    if 'gender' in df_encode.columns:
        encodeur_genre = LabelEncoder()
        df_encode['gender'] = encodeur_genre.fit_transform(df_encode['gender'])
        print("Encodage gender :", dict(zip(encodeur_genre.classes_,
                                            encodeur_genre.transform(encodeur_genre.classes_))))
    
    if 'country' in df_encode.columns:
        df_encode = pd.get_dummies(df_encode, columns=['country'], prefix='pays', drop_first=True)
        print("Encodage One-Hot : country")
    
    return df_encode


def separer_features_cible(df):
    """Separe les features de la cible"""
    y = df['churn']
    X = df.drop('churn', axis=1)
    return X, y


def diviser_train_test(X, y, taille_test=0.2, graine_aleatoire=42):
    """Divise les donnees en train et test"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=taille_test, random_state=graine_aleatoire, stratify=y
    )
    print(f"Division train/test: {len(X_train)}/{len(X_test)}")
    return X_train, X_test, y_train, y_test


def filtrer_donnees_normales(X_train, y_train):
    """Filtre uniquement les donnees normales (churn=0) pour l'entrainement"""
    masque_normal = y_train == 0
    X_train_normal = X_train[masque_normal]
    
    print(f"Donnees normales filtrees: {len(X_train_normal)}/{len(X_train)} ({len(X_train_normal)/len(X_train)*100:.1f}%)")
    
    return X_train_normal


def normaliser_features(X_train_normal, X_test):
    """Normalise les features avec StandardScaler"""
    normaliseur = StandardScaler()
    
   
    X_train_normalise = normaliseur.fit_transform(X_train_normal)
    X_test_normalise = normaliseur.transform(X_test)
    
    # Sauvegarde du scaler
    sauvegarder_modele(normaliseur, CHEMIN_MODELES + 'normaliseur_anomalie.pkl')
    print(f"Normaliseur sauvegarde : {CHEMIN_MODELES}normaliseur_anomalie.pkl")
    
    # Conversion en DataFrame pour garder les noms de colonnes
    X_train_normalise = pd.DataFrame(X_train_normalise, columns=X_train_normal.columns)
    X_test_normalise = pd.DataFrame(X_test_normalise, columns=X_test.columns)
    
    return X_train_normalise, X_test_normalise


def sauvegarder_donnees_preparees(X_train_normal, X_test, y_test):
    """Sauvegarde les donnees preparees"""
    
    sauvegarder_donnees(X_train_normal, CHEMIN_PROCESSED + 'X_train_normal.csv')
    sauvegarder_donnees(X_test, CHEMIN_PROCESSED + 'X_test.csv')
    
    # On sauvegarde uniquement y_test pour l'evaluation
    y_test_df = pd.DataFrame(y_test, columns=['churn'])
    sauvegarder_donnees(y_test_df, CHEMIN_PROCESSED + 'y_test.csv')
    
    print(f"Donnees sauvegardees dans : {CHEMIN_PROCESSED}")


def main():
    """Fonction principale"""
   
    print("PREPROCESSING POUR DETECTION D'ANOMALIES")
    
    # Chargement et nettoyage des donnees
    df = charger_et_nettoyer_donnees()
    print(f"Donnees chargees : {df.shape}")
    
    # Encodage des variables categorielles
    df_encode = encoder_variables_categorielles(df)
    
    # Separation des features et de la cible
    X, y = separer_features_cible(df_encode)
    
    # Division des donnees en set d'entrainement et set de test
    X_train, X_test, y_train, y_test = diviser_train_test(X, y)
    
    # Filtrage des clients normaux uniquement pour l'entrainement
    X_train_normal = filtrer_donnees_normales(X_train, y_train)
    
    # Normalisation des features
    X_train_norm, X_test_norm = normaliser_features(X_train_normal, X_test)
    
    # Sauvegarde des donnees preparees
    sauvegarder_donnees_preparees(X_train_norm, X_test_norm, y_test)
    
    print(f"X_train   : {X_train_norm.shape}")
    print(f"X_test    : {X_test_norm.shape}")
    print(f"y_test    : {len(y_test)}")

if __name__ == "__main__":
    main()