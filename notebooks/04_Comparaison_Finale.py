"""
Comparaison entre classification et detection d'anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import charger_donnees, charger_modele, calculer_metriques, afficher_metriques

# Chemins
CHEMIN_CLASS = '../data/processed/classification/'
CHEMIN_ANOM = '../data/processed/anomalie/'
CHEMIN_MODELES = '../models/'
CHEMIN_FIGURES = '../reports/figures/'

os.makedirs(CHEMIN_FIGURES, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

#charger les donnees de test pour les deux approches
def charger_donnees_test():
    X_test_class = charger_donnees(CHEMIN_CLASS + 'X_test.csv')
    y_test_class = charger_donnees(CHEMIN_CLASS + 'y_test.csv')['churn']

    X_test_anom = charger_donnees(CHEMIN_ANOM + 'X_test.csv')
    y_test_anom = charger_donnees(CHEMIN_ANOM + 'y_test.csv')['churn']
    return X_test_class, y_test_class, X_test_anom, y_test_anom

#chargement tous les modeles entraines
def charger_tous_modeles():
    modeles_classification = {
        'KNN': charger_modele(CHEMIN_MODELES + 'knn_classification.pkl'),
        'Arbre Decision': charger_modele(CHEMIN_MODELES + 'arbre_decision_classification.pkl'),
        'Foret Aleatoire': charger_modele(CHEMIN_MODELES + 'foret_aleatoire_classification.pkl')
    }

    modeles_anomalie = {
        'Isolation Forest': charger_modele(CHEMIN_MODELES + 'isolation_forest_anomalie.pkl'),
        'One-Class SVM': charger_modele(CHEMIN_MODELES + 'one_class_svm_anomalie.pkl')
    }

    return modeles_classification, modeles_anomalie

#evaluation des modeles pour les deux approches
def evaluer_modeles_classification(modeles, X_test, y_test):
    resultats = {}
    predictions = {}

    for nom, modele in modeles.items():
        y_pred = modele.predict(X_test)
        y_pred_proba = modele.predict_proba(X_test)[:, 1]
        metriques = calculer_metriques(y_test, y_pred, y_pred_proba)

        afficher_metriques(nom, metriques)
        resultats[nom] = metriques
        predictions[nom] = {'y_pred': y_pred, 'y_pred_proba': y_pred_proba}

    return resultats, predictions

#evaluation des modeles de detection d'anomalies
def evaluer_modeles_anomalie(modeles, X_test, y_test):
    resultats = {}
    predictions = {}

    for nom, modele in modeles.items():
        #conversion des predictions d'anomalies en labels binaires de churn (1 pour churn ey 0 pour non-churn)
        y_pred = np.where(modele.predict(X_test) == -1, 1, 0)
        metriques = calculer_metriques(y_test, y_pred)

        afficher_metriques(nom, metriques)
        resultats[nom] = metriques
        predictions[nom] = {'y_pred': y_pred, 'y_pred_proba': None}

    return resultats, predictions

# tracage des resultats
def tracer_comparaison_approches(resultats_class, resultats_anom):
    df = pd.DataFrame({**resultats_class, **resultats_anom}).T
    df['Approche'] = ['Classification'] * len(resultats_class) + ['Anomalie'] * len(resultats_anom)

    metriques = ['accuracy', 'precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes = axes.ravel()
    for i, m in enumerate(metriques):
        plot_df = df[[m, 'Approche']].reset_index()
        colors = ['skyblue' if a == 'Classification' else 'salmon' for a in plot_df['Approche']]

        axes[i].bar(plot_df.index, plot_df[m], color=colors, edgecolor='black')
        axes[i].set_title(m.upper())
        axes[i].set_xticks(plot_df.index)
        axes[i].set_xticklabels(plot_df['index'], rotation=45)

    plt.tight_layout()
    plt.savefig(CHEMIN_FIGURES + '10_comparaison_approches.png')
    plt.close()

# tracage des matrices de confusion
def tracer_matrices_confusion(pred_class, pred_anom, y_test_class, y_test_anom):
    toutes_preds = {**pred_class, **pred_anom}

    n = len(toutes_preds)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel()

    for i, (nom, preds) in enumerate(toutes_preds.items()):
        y_true = y_test_class if nom in pred_class else y_test_anom
        cm = confusion_matrix(y_true, preds['y_pred'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(nom)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(CHEMIN_FIGURES + '11_matrices_confusion_tous.png')
    plt.close()

# tracage des courbes ROC pour les modeles de classification
def tracer_courbes_roc(pred_class, y_test):
    plt.figure(figsize=(10, 8))

    for nom, preds in pred_class.items():
        if preds['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{nom} (AUC={auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig(CHEMIN_FIGURES + '12_courbes_roc_classification.png')
    plt.close()

# creation du tableau recapitulatif
def creer_tableau_recapitulatif(resultats_class, resultats_anom):
    df = pd.DataFrame({**resultats_class, **resultats_anom}).T.round(4)
    df['Approche'] = ['Classification'] * len(resultats_class) + ['Anomalie'] * len(resultats_anom)
    df = df.sort_values('f1_score', ascending=False)

    df.to_csv(CHEMIN_FIGURES + 'recapitulatif_tous_modeles.csv', sep=';', decimal=',', index=True)
    df[df['Approche'] == 'Classification'].drop('Approche', axis=1).to_csv(
        CHEMIN_MODELES + 'metriques_classification.csv', sep=';', decimal=',', index=False
    )
    df[df['Approche'] == 'Anomalie'].drop('Approche', axis=1).to_csv(
        CHEMIN_MODELES + 'metriques_anomalie.csv', sep=';', decimal=',', index=False
    )

    return df

#identification des meilleurs modeles
def identifier_meilleurs_modeles(df):
    meilleur_class = df[df['Approche'] == 'Classification']['f1_score'].idxmax()
    meilleur_anom = df[df['Approche'] == 'Anomalie']['f1_score'].idxmax()
    meilleur_global = df['f1_score'].idxmax()
    return meilleur_class, meilleur_anom, meilleur_global


def main():
    Xc, yc, Xa, ya = charger_donnees_test()
    modeles_class, modeles_anom = charger_tous_modeles()

    res_class, pred_class = evaluer_modeles_classification(modeles_class, Xc, yc)
    res_anom, pred_anom = evaluer_modeles_anomalie(modeles_anom, Xa, ya)

    tracer_comparaison_approches(res_class, res_anom)
    tracer_matrices_confusion(pred_class, pred_anom, yc, ya)
    tracer_courbes_roc(pred_class, yc)

    df = creer_tableau_recapitulatif(res_class, res_anom)
    identifier_meilleurs_modeles(df)


if __name__ == "__main__":
    main()