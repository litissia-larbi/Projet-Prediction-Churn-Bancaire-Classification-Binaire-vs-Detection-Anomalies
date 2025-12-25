# Projet Academique realise dans le cadre du module Apprentissage Artificiel
# Prediction du Churn Bancaire - Comparaison de Deux Approches ( Classification binaire VS Detection d'anomalie )
# Auteur : LARBI Litissia 

## Vue d'ensemble du projet
Ce projet compare deux approches differentes de machine learning pour predire le churn (depart) des clients bancaires :

1. **Classification Binaire Supervisee** (avec equilibrage de classe SMOTE)
2. **Detection d'Anomalies Non Supervisee** (entrainement uniquement sur cas normaux ( churn = 0))

## Pourquoi deux approches ?

### Approche par Classification Supervisee
- Traite le churn comme un probleme classique de classification binaire
- Utilise les donnees etiquetees (0 = reste, 1 = part)
- Applique SMOTE pour equilibrer les classes
- Modeles utilises : KNN, Arbre de Decision, Foret Aleatoire

### Approche par Detection d'Anomalies
- Traite les clients qui partent comme des "anomalies"
- Entraine UNIQUEMENT sur les clients normaux (churn = 0)
- PAS d'equilibrage des classes (pas de SMOTE ou autre)
- Modeles utilises : Isolation Forest, One-Class SVM

## Structure du projet

```
bank-churn-prediction/
├── data/
│   ├── raw/
│   │   └── Bank Customer Churn Prediction.csv
│   ├── interim/
│   │   └── donnees_explorees.csv
│   └── processed/
│       ├── classification/
│       │   ├── X_train.csv
│       │   ├── X_test.csv
│       │   ├── y_train.csv
│       │   └── y_test.csv
│       └── anomalie/
│           ├── X_train_normal.csv
│           ├── X_test.csv
│           ├── y_train_normal.csv
│           └── y_test.csv
├── models/
│   ├── knn_classification.pkl
│   ├── arbre_decision_classification.pkl
│   ├── foret_aleatoire_classification.pkl
│   ├── isolation_forest_anomalie.pkl
│   ├── one_class_svm_anomalie.pkl
│   ├── normaliseur_classification.pkl
│   └── normaliseur_anomalie.pkl
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   └── utils.py
└── notebooks/
    ├── 01_EDA.py
    ├── 02_Preprocessing_Classification.py
    ├── 02_Preprocessing_Anomaly.py
    ├── 03_Classification_Models.py
    ├── 03_Anomaly_Detection_Models.py
    └── 04_Comparaison_Finale.py
```

## Explication des fichiers

### utils.py
**Role :** Fonctions utilitaires partagees

**Fonctions principales :**
- `charger_donnees()` : Charge un fichier CSV
- `sauvegarder_donnees()` : Sauvegarde un fichier CSV
- `sauvegarder_modele()` : Sauvegarde un modele entraine (.pkl)
- `charger_modele()` : Charge un modele entraine
- `calculer_metriques()` : Calcule accuracy, precision, recall, F1-Score, AUC-ROC
- `afficher_metriques()` : Affiche les metriques de facon formatee

### 01_EDA.py
**Role :** Analyse Exploratoire des Donnees

**Actions :**
- Charge les donnees brutes du CSV
- Affiche les statistiques de base (dimensions, types, valeurs manquantes)
- Analyse la variable cible (churn)
- Visualise les distributions des variables numeriques
- Analyse les taux de churn par variables categorielles
- Sauvegarde les donnees explorees

### 02_Preprocessing_Classification.py
**Role :** Preparation des donnees pour la classification supervisee

**Actions :**
1. Nettoie les donnees (supprime customer_id)
2. Encode les variables categorielles (gender, country)
3. Separe features (X) et cible (y)
4. Divise en set de train/test (80/20)
5. **APPLIQUE SMOTE** pour equilibrer les classes 
6. Normalise les features avec StandardScaler
7. Sauvegarde les donnees preparees

**Pourquoi SMOTE ici :**
Les modeles de classification apprennent mieux avec des classes equilibrees
SMOTE cree des exemples synthetiques de la classe minoritaire (churn = 1)

### 02_Preprocessing_Anomaly.py
**Role :** Preparation des donnees pour la detection d'anomalies

**Actions :**
1. Nettoie les donnees (supprime customer_id)
2. Encode les variables categorielles (gender, country)
3. Separe features (x) et cible (y)
4. Divise en set de train/test (80/20)
5. **FILTRE pour ne garder QUE les cas normaux** (churn=0) 
6. Normalise les features avec StandardScaler
7. Sauvegarde les donnees preparees

**On  a pas utiliser une methode d'equilibre de classes comme SMOTE car :**
Les modeles d'anomalie doivent apprendre ce qui est "normal"
Ils detectent ensuite automatiquement ce qui s'en ecarte (les anomalies)

### 03_Classification_Models.py
**Role :** Entrainement des modeles de classification supervisee

**Modeles :**
1. **KNN** : Classe selon les k plus proches voisins
2. **Arbre de Decision** : Cree des regles de type if-then-else
3. **Foret Aleatoire** : Ensemble d'arbres qui votent

**Actions :**
- Charge les donnees equilibrees (avec SMOTE)
- Entraine chaque modele
- Sauvegarde les modeles entraines

### 03_Anomaly_Detection_Models.py
**Role :** Entrainement des modeles de detection d'anomalies

**Modeles :**
1. **Isolation Forest** : Isole les anomalies dans des arbres
2. **One-Class SVM** : Apprend la frontiere des donnees normales

**Actions :**
- Charge les donnees normales uniquement
- Entraine les modeles sans labels (non supervise)
- Sauvegarde les modeles entraines

### 04_Comparaison_Finale.py
**Role :** Comparaison complete des deux approches

**Actions :**
- Charge tous les 5 modeles entrainees (3 modeles de classification + 2 modleles de detectiion d'anomalie)
- Evalue chaque modele sur les donnees de test
- Cree des visualisations :
  - Comparaison des metriques par approche
  - Matrices de confusion pour tous les modeles
  - Courbes ROC pour les modeles de classification
- Genere un tableau recapitulatif
- Identifie les meilleurs modeles 

## Ordre d'execution

```bash
# Etape 1 : Analyse exploratoire
python notebooks/01_EDA.py

# Etape 2a : Preprocessing pour classification
python notebooks/02_Preprocessing_Classification.py

# Etape 2b : Preprocessing pour anomalie
python notebooks/02_Preprocessing_Anomaly.py

# Etape 3a : Entrainement classification
python notebooks/03_Classification_Models.py

# Etape 3b : Entrainement anomalie
python notebooks/03_Anomaly_Detection_Models.py

# Etape 4 : Comparaison finale
python notebooks/04_Comparaison_Finale.py
```

## Differences cles entre les approches

| Aspect                 | Classification                      | Detection Anomalie                           |
|------------------------|-------------------------------------|----------------------------------------------|
| Donnees entrainement   |Tous echantillons (equilibres SMOTE) | Uniquement cas normaux avec chirn =0         |
| Labels utilises        |Oui (supervise)                      | Non (non supervise)                          |
| Equilibrage classes    | SMOTE applique                      | Non applicable                               |
| Objectif apprentissage | Distinguer churn vs non-churn       | Apprendre le comportement normal d'un client |
| Prediction             | Probabilite de churn                | Score d'anomalie                             |

## Interpretation des metriques

- **Accuracy (Exactitude)** : Pourcentage de predictions correctes
- **Precision** : Des clients predits churn, combien le sont vraiment
- **Recall (Rappel)** : Des clients qui partent, combien on detecte
- **F1-Score** : Moyenne harmonique precision/recall (metrique equilibree)
- **AUC-ROC** : Capacite globale a discriminer (classification uniquement)

## Dependances

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
```

Installation :
```bash
pip install -r requirements.txt
```

## Dataset

Utilise "Bank Customer Churn Prediction.csv" de Kaggle
Placer dans : `data/raw/Bank Customer Churn Prediction.csv`

## Optimisations du code

1. **Fonctions reutilisables** : Toutes les operations communes dans utils.py
2. **Commentaires pertinents** : Explication breve de ce qui ete fait 
3. **Structure modulaire** : Chaque fichier a une responsabilite claire
4. **Pas de code duplique** : Maximise la reutilisation
5. **Gestion des erreurs** : Protection contre les valeurs manquantes
6. **Chemins relatifs** : Facilite le deploiement

## Notes importantes

Ce projet demontre que la prediction du churn peut etre abordee de deux manieres :
1. Classification supervisee (meilleure pour la plupart des cas)
2. Detection d'anomalies 

La comparaison aide a comprendre quelle approche fonctionne le mieux pour un cas specifique comme le churn bancaire 

**Point critique** : SMOTE est utilise UNIQUEMENT pour la classification,mais JAMAIS pour la detection d'anomalies                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
**Technologies Utilisées**                                                                                                                                                                                                                                                                                                                                                                                                            
Python 3.x                                                                                                                                                                                                            
Scikit-learn (modèles ML, métriques, preprocessing)                                                                                                                                                                                
Pandas & NumPy (manipulation de données)                                                                                                                                                                                        
Matplotlib & Seaborn (visualisation)                                                                                                                                                                                    
Imbalanced-learn (SMOTE)
