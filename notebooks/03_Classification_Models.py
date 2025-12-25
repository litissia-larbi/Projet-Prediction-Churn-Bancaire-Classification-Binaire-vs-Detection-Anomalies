"""
Classification avec optimisation automatique des hyperparametres
Entrainement de KNN, Arbre de Decision et Foret Aleatoire
"""

import pandas as pd
import sys
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, sauvegarder_modele

CHEMIN_PROCESSED = '../data/processed/classification/'
CHEMIN_MODELES = '../models/'

os.makedirs(CHEMIN_MODELES, exist_ok=True)

def charger_donnees_preparees():
    """Charge les donnees preparees pour la classification"""

    X_train = charger_donnees(CHEMIN_PROCESSED + 'X_train.csv')
    y_train = charger_donnees(CHEMIN_PROCESSED + 'y_train.csv')['churn']

    
    return X_train, y_train

def optimiser_et_entrainer_knn(X_train, y_train):
    """Optimise et entraine KNN avec GridSearchCV"""
    print("MODELE 1/3 : K-NEAREST NEIGHBORS (KNN)")
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    print("Recherche des meilleurs hyperparametres")
    debut = time.time()
    grid_search.fit(X_train, y_train)
    duree = time.time() - debut
    
    print(f"Temps d'optimisation : {duree:.1f}s")
    print(f"Score F1: {grid_search.best_score_:.4f}")
    print("Meilleurs parametres :", grid_search.best_params_)
    
    modele_final = grid_search.best_estimator_
    sauvegarder_modele(modele_final, CHEMIN_MODELES + 'knn_classification.pkl')
    
    return grid_search.best_params_

def optimiser_et_entrainer_arbre_decision(X_train, y_train):
    """Optimise et entraine Decision Tree avec GridSearchCV"""
    print("MODELE 2/3 : ARBRE DE DECISION")
    
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [10, 20, 50], #minimum d'echantillons pour diviser un noeud
        'min_samples_leaf': [5, 10, 20], #minimum d'echantillons dans une feuille
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    print("Recherche des meilleurs hyperparametres...")
    debut = time.time()
    grid_search.fit(X_train, y_train)
    duree = time.time() - debut
    
    print(f"Temps d'optimisation : {duree:.1f}s")
    print(f"Score F1 (validation croisee) : {grid_search.best_score_:.4f}")
    print("Meilleurs parametres :", grid_search.best_params_)
    
    modele_final = grid_search.best_estimator_
    sauvegarder_modele(modele_final, CHEMIN_MODELES + 'arbre_decision_classification.pkl')
    
    return grid_search.best_params_

def optimiser_et_entrainer_foret_aleatoire(X_train, y_train):
    """Optimise et entraine Random Forest avec GridSearchCV"""
    print("MODELE 3/3 : FORET ALEATOIRE")
 
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [5, 10, 20], #minimum d'echantillons pour diviser un noeud
        'min_samples_leaf': [2, 5, 10]    #minimum d'echantillons dans une feuille
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    print("Recherche des meilleurs hyperparametres")
    debut = time.time()
    grid_search.fit(X_train, y_train)
    duree = time.time() - debut
    
    print(f"Temps d'optimisation : {duree:.1f}s ({duree/60:.1f} min)")
    print(f"Score F1 (validation croisee) : {grid_search.best_score_:.4f}")
    print("Meilleurs parametres :", grid_search.best_params_)
    
    modele_final = grid_search.best_estimator_
    sauvegarder_modele(modele_final, CHEMIN_MODELES + 'foret_aleatoire_classification.pkl')
    
    return grid_search.best_params_


def main():
    """Fonction principale d'execution"""

    X_train, y_train = charger_donnees_preparees()
    print(f"X_train : {X_train.shape}")
    
    meilleurs_params = {}
    
    # Entrainement des 3 modeles avec optimisation
    meilleurs_params['KNN'] = optimiser_et_entrainer_knn(X_train, y_train)
    meilleurs_params['Arbre de Decision'] = optimiser_et_entrainer_arbre_decision(X_train, y_train)
    meilleurs_params['Foret Aleatoire'] = optimiser_et_entrainer_foret_aleatoire(X_train, y_train)
    
    # Sauvegarde des meilleurs parametres
  
    params_df = pd.DataFrame(meilleurs_params).T
    params_df.to_csv(CHEMIN_MODELES + 'meilleurs_parametres.csv', sep=';', decimal=',', index=True)

    print("\nMEILLEURS HYPERPARAMETRES TROUVES :")
    for modele, params in meilleurs_params.items():
        print(f"\n{modele} :")
        for param, valeur in params.items():
            print(f"  {param:25s} : {valeur}")
    
    print("3 modeles sauvegardes dans :", CHEMIN_MODELES)

if __name__ == "__main__":
    main()