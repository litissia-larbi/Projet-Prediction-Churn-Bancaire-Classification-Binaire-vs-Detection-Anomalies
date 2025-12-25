"""
Modeles de detection d'anomalies
Entrainement de  Isolation Forest et One-Class SVM

"""

import pandas as pd
import sys
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, sauvegarder_modele

CHEMIN_PROCESSED = '../data/processed/anomalie/'
CHEMIN_MODELES = '../models/'

os.makedirs(CHEMIN_MODELES, exist_ok=True)

def charger_donnees_preparees():
    """Charge les donnees preparees pour la detection d'anomalies"""
    X_train = charger_donnees(CHEMIN_PROCESSED + 'X_train_normal.csv')
    return X_train

def entrainer_isolation_forest(X_train, contamination=0.2):
    """Entraine le modele Isolation Forest"""
    print("MODELE 1/2 : ISOLATION FOREST")
    
    modele = IsolationForest(
        n_estimators=100, # nombre d'arbres dans la foret par defaut
        contamination=contamination, # 20% d'anomalies attendues (20%churn attendus)
        max_samples='auto', # Taille d'échantillon automatique
        random_state=42
     
    )
    
    print("Entrainement en cours")
    modele.fit(X_train)
    
    print(f"Modele entraine sur {X_train.shape[0]} echantillons normaux")
    print(f"Contamination : {contamination}")
    
    sauvegarder_modele(modele, CHEMIN_MODELES + 'isolation_forest_anomalie.pkl')
    
    return modele

def entrainer_one_class_svm(X_train, nu=0.2):
    """Entraine le modele One-Class SVM"""
    print("MODELE 2/2 : ONE-CLASS SVM")
    
    #nu = 0.2 qui est le taux d'anomalies attendu 20% de churn 
    #gamma automatique 'scale' pour qu'il s'ajuste a la variance des donnees
    #kernet='rbf' pour capturer les relations non lineaires
    modele = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    
    print("Entrainement en cours")
    modele.fit(X_train)
    
    print(f"Modele entraine sur {X_train.shape[0]} echantillons normaux")
    print(f"Nu : {nu}")
    
    sauvegarder_modele(modele, CHEMIN_MODELES + 'one_class_svm_anomalie.pkl')
    
    return modele

def main():
    """Fonction principale d'execution"""

    X_train = charger_donnees_preparees()
    print(f"X_train : {X_train.shape}")
    
    # Entrainement des 2 modeles
    entrainer_isolation_forest(X_train, contamination=0.2)
    entrainer_one_class_svm(X_train, nu=0.2)
    
    print("ENTRAINEMENT TERMINE")
    print("\n2 modeles sauvegardes dans :", CHEMIN_MODELES)
  
if __name__ == "__main__":
    main()