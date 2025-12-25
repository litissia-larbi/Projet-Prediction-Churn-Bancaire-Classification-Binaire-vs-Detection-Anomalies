"""
Preparation pour la classification supervisee
Utilise SMOTE pour equilibrer les classes pour l'approche de classification binaire
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, sauvegarder_donnees, sauvegarder_modele

CHEMIN_INTERIM = '../data/interim/donnees_explorees.csv'
CHEMIN_PROCESSED = '../data/processed/classification/'
CHEMIN_MODELES = '../models/'

os.makedirs(CHEMIN_PROCESSED, exist_ok=True)
os.makedirs(CHEMIN_MODELES, exist_ok=True)

#Nettoyage des donnees
def charger_et_nettoyer_donnees():
    df = charger_donnees(CHEMIN_INTERIM)

    colonnes_a_supprimer = ['customer_id']
    colonnes_existantes = [col for col in colonnes_a_supprimer if col in df.columns]

    if colonnes_existantes:
        df = df.drop(columns=colonnes_existantes)

    return df

#Encodage des variables categoriellesS
def encoder_variables_categorielles(df):
    df_encode = df.copy()

    # Encodage du genre
    if 'gender' in df_encode.columns:
        encodeur_genre = LabelEncoder()
        df_encode['gender'] = encodeur_genre.fit_transform(df_encode['gender'])
        print("Encodage gender :", dict(zip(encodeur_genre.classes_,
                                            encodeur_genre.transform(encodeur_genre.classes_))))
        sauvegarder_modele(encodeur_genre, CHEMIN_MODELES + 'encodeur_genre.pkl')

    # Encodage du pays
    if 'country' in df_encode.columns:
        df_encode = pd.get_dummies(df_encode, columns=['country'], prefix='pays', drop_first=True)
        print("Encodage OneHot : country ", [col for col in df_encode.columns if 'pays' in col])

    return df_encode

#Separation des features et de la cible
def separer_features_cible(df):
    y = df['churn']
    X = df.drop('churn', axis=1)
    return X, y

# Division des donnees en set d'entrainement et set de test
def diviser_train_test(X, y, taille_test=0.2, graine_aleatoire=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=taille_test, random_state=graine_aleatoire, stratify=y
    )
    print("Repartition train/test :", len(X_train), "/", len(X_test))
    return X_train, X_test, y_train, y_test

# Application de SMOTE pour equilibrer les classes
def appliquer_smote(X_train, y_train, graine_aleatoire=42):
    print("Application de SMOTE")

    print("Avant SMOTE :", dict(y_train.value_counts()))
    smote = SMOTE(random_state=graine_aleatoire)
    X_train_equilibre, y_train_equilibre = smote.fit_resample(X_train, y_train)
    print("Apres SMOTE :", dict(y_train_equilibre.value_counts()))

    return X_train_equilibre, y_train_equilibre

# Normalisation des features
def normaliser_features(X_train, X_test):
    normaliseur = StandardScaler()

    X_train_normalise = normaliseur.fit_transform(X_train)
    X_test_normalise = normaliseur.transform(X_test)
    
    # Sauvegarde du scaler
    sauvegarder_modele(normaliseur, CHEMIN_MODELES + 'normaliseur_classification.pkl')
    print("Normaliseur sauvegarde :", CHEMIN_MODELES + 'normaliseur_classification.pkl')

    X_train_normalise = pd.DataFrame(X_train_normalise, columns=X_train.columns)
    X_test_normalise = pd.DataFrame(X_test_normalise, columns=X_test.columns)

    return X_train_normalise, X_test_normalise

# Sauvegarde des donnees preparees
def sauvegarder_donnees_preparees(X_train, X_test, y_train, y_test):
    sauvegarder_donnees(X_train, CHEMIN_PROCESSED + 'X_train.csv')
    sauvegarder_donnees(X_test, CHEMIN_PROCESSED + 'X_test.csv')

    sauvegarder_donnees(pd.DataFrame(y_train, columns=['churn']),
                        CHEMIN_PROCESSED + 'y_train.csv')
    sauvegarder_donnees(pd.DataFrame(y_test, columns=['churn']),
                        CHEMIN_PROCESSED + 'y_test.csv')

    print("Donnees sauvegardees dans :", CHEMIN_PROCESSED)


def main():
    # Charger et nettoyer les donnees
    df = charger_et_nettoyer_donnees()

    # Encodage des variables categorielles
    df_encode = encoder_variables_categorielles(df)

    #Separation des features  et de la cible
    X, y = separer_features_cible(df_encode)

    # Division des donnees en set d'entrainement et set de test
    X_train, X_test, y_train, y_test = diviser_train_test(X, y)
    
    # Application de SMOTE sur le set d'entrainment pour equilibrer les classes
    X_train_equilibre, y_train_equilibre = appliquer_smote(X_train, y_train)

    # Normalisation des features APRES SMOTE
    X_train_norm, X_test_norm = normaliser_features(X_train_equilibre, X_test)

    # Sauvegarde des donnees preparees
    sauvegarder_donnees_preparees(X_train_norm, X_test_norm, y_train_equilibre, y_test)

    print("\nPreprocessing termine")
    print("  X_train :", X_train_norm.shape)
    print("  X_test  :", X_test_norm.shape)
    print("  y_train :", len(y_train_equilibre))
    print("  y_test  :", len(y_test))


if __name__ == "__main__":
    main()