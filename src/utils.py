"""
Fonctions utilitaires pour le projet de prediction du churn bancaire
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def charger_donnees(chemin_fichier, sep=';'):
    """Charge les donnees depuis un fichier CSV"""
    return pd.read_csv(chemin_fichier, sep=sep, decimal=',')



def sauvegarder_donnees(donnees, chemin_fichier):
    """Sauvegarde les donnees dans un fichier CSV"""
    os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
    donnees.to_csv(chemin_fichier, index=False, sep=';', decimal=',')
    print(f"Donnees sauvegardees : {chemin_fichier}")


def sauvegarder_modele(modele, chemin_fichier):
    """Sauvegarde un modele entraine au format .pkl"""
    os.makedirs(os.path.dirname(chemin_fichier), exist_ok=True)
    joblib.dump(modele, chemin_fichier)
    print(f"Modele sauvegarde : {chemin_fichier}")


def charger_modele(chemin_fichier):
    """Charge un modele entraine depuis un fichier .pkl"""
    return joblib.load(chemin_fichier)


def calculer_metriques(y_vrai, y_pred, y_pred_proba=None):
    """
    Calcule les metriques de performance d'un modele
    
    Parametres:
        y_vrai: Vraies valeurs
        y_pred: Predictions
        y_pred_proba: Probabilites predites (optionnel pour AUC-ROC)
    
    Retour:
        Dictionnaire contenant toutes les metriques
    """
    metriques = {
        'accuracy': accuracy_score(y_vrai, y_pred),
        'precision': precision_score(y_vrai, y_pred, zero_division=0),
        'recall': recall_score(y_vrai, y_pred, zero_division=0),
        'f1_score': f1_score(y_vrai, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metriques['auc_roc'] = roc_auc_score(y_vrai, y_pred_proba)
        except:
            metriques['auc_roc'] = None
    
    return metriques


def afficher_metriques(nom_modele, metriques):
    """Affiche les metriques de maniere formatee"""
    print(f"\n{nom_modele}:")
    print(f"  Exactitude : {metriques['accuracy']:.4f}")
    print(f"  Precision :  {metriques['precision']:.4f}")
    print(f"  Rappel :     {metriques['recall']:.4f}")
    print(f"  F1-Score :   {metriques['f1_score']:.4f}")
    if 'auc_roc' in metriques and metriques['auc_roc'] is not None:
        print(f"  AUC-ROC :    {metriques['auc_roc']:.4f}")