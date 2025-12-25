"""
Analyse exploratoire des donnees bancaires pour la prediction du churn ( Exploration Data Analysis - EDA )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, sauvegarder_donnees

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

CHEMIN_DONNEES = '../data/raw/Bank Customer Churn Prediction.csv'
CHEMIN_INTERIM = '../data/interim/donnees_explorees.csv'
CHEMIN_FIGURES = '../reports/figures/'

os.makedirs(CHEMIN_FIGURES, exist_ok=True)

def charger_et_inspecter_donnees():
    """Charge et inspecte les donnees"""
    df = charger_donnees(CHEMIN_DONNEES, sep=',')
    
    print(f"\n{'='*60}")
    print(f"INFORMATIONS GENERALES SUR LE DATASET")
    print(f"\nDimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"\nColonnes disponibles:")
    print(df.columns.tolist())
    print(f"\nTypes de donnees:")
    print(df.dtypes)
    
    return df

def verifier_qualite_donnees(df):
    """Verifie la qualite des donnees (manquants, doublons, etc)"""
    # Valeurs manquantes
    missing = df.isnull().sum()
    pct_missing = (missing / len(df)) * 100
    
    if missing.sum() > 0:
        print(f"\n Valeurs manquantes detectees:")
        for col in missing[missing > 0].index:
            print(f"  - {col}: {missing[col]} ({pct_missing[col]:.2f}%)")
    else:
        print(f"\n Aucune valeur manquante detectee")
    
    # Doublons
    nb_doublons = df.duplicated().sum()
    if nb_doublons > 0:
        print(f"\n {nb_doublons} ligne(s) en doublon detectee(s)")
    else:
        print(f"\n Aucun doublon detecte")
    
    # Statistiques descriptives
    print(f"\n{'='*60}")
    print(f"STATISTIQUES DESCRIPTIVES")
    print(f"{'='*60}")
    print(df.describe())
    
    return df

def analyser_variable_cible(df):
    """Analyse la variable cible (churn)"""
    comptes_churn = df['churn'].value_counts()
    
    print(f"ANALYSE DE LA VARIABLE CIBLE (CHURN)")
    print(f"\nRepartition:")
    print(f"  Clients restes (0): {comptes_churn[0]} ({comptes_churn[0]/len(df)*100:.1f}%)")
    print(f"  Clients partis (1): {comptes_churn[1]} ({comptes_churn[1]/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Graphique en barres
    comptes_churn.plot(kind='bar', ax=axes[0], color=['green', 'red'])
    axes[0].set_title('Distribution du Churn')
    axes[0].set_xlabel('churn (0=Reste, 1=Part)')
    axes[0].set_ylabel('Nombre de clients')
    axes[0].set_xticklabels(['Reste', 'Part'], rotation=0)

    # Graphique en camembert
    axes[1].pie(
        comptes_churn,
        labels=['Reste', 'Part'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        startangle=90
    )
    axes[1].set_title('Proportion du Churn')

    plt.tight_layout()
    plt.savefig(CHEMIN_FIGURES + '01_distribution_cible.png', dpi=300, bbox_inches='tight')
    print(f"\ngraphique sauvegarde : {CHEMIN_FIGURES}01_distribution_cible.png")
    plt.close()

def analyser_variables_numeriques(df):
    """Analyse les variables numeriques"""
    colonnes_numeriques = [
        'credit_score', 'age', 'tenure', 'balance',
        'products_number', 'estimated_salary'
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, col in enumerate(colonnes_numeriques):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution de {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequence')

    plt.tight_layout()
    plt.savefig(CHEMIN_FIGURES + '02_distributions_numeriques.png', dpi=300, bbox_inches='tight')
    print(f"\n graphique sauvegarde : {CHEMIN_FIGURES}02_distributions_numeriques.png")
    plt.close()

def analyser_variables_categorielles(df):
    """Analyse les variables categorielles"""
    colonnes_categorielles = ['country', 'gender', 'credit_card', 'active_member']
    
    print(f"ANALYSE DES VARIABLES CATEGORIELLES")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, col in enumerate(colonnes_categorielles):
        taux_churn = df.groupby(col)['churn'].mean() * 100
        taux_churn.plot(kind='bar', ax=axes[idx], color='coral')

        axes[idx].set_title(f'Taux de Churn par {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Taux de Churn (%)')
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)

        for container in axes[idx].containers:
            axes[idx].bar_label(container, fmt='%.1f%%')

    plt.tight_layout()
    plt.savefig(CHEMIN_FIGURES + '03_taux_churn_categorielles.png', dpi=300, bbox_inches='tight')
    print(f"\n graphique sauvegarde : {CHEMIN_FIGURES}03_taux_churn_categorielles.png")
    plt.close()


def main():
   
    print("DEMARRAGE DE L'ANALYSE EXPLORATOIRE (EDA)")
    df = charger_et_inspecter_donnees()
    verifier_qualite_donnees(df)
    analyser_variable_cible(df)
    analyser_variables_numeriques(df)
    analyser_variables_categorielles(df)

    sauvegarder_donnees(df, CHEMIN_INTERIM)
    print(f"ANALYSE TERMINEE")
    print(f" Donnees explorees sauvegardees : {CHEMIN_INTERIM}")
    print(f" Graphiques generes dans : {CHEMIN_FIGURES}")


if __name__ == "__main__":
    main()