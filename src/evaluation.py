#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'Évaluation - Calcul des Métriques de Performance

Ce module gère le calcul des métriques de performance (Accuracy, Precision, Recall, F1-Score),
la génération de matrices de confusion et la création de rapports comparatifs pour les modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         model_name: str) -> Dict[str, float]:
    """
    Calcule les 5 métriques requises pour un modèle

    LES 5 MÉTRIQUES OBLIGATOIRES:

    1. ACCURACY: Score global
       - Proportion de prédictions correctes
       - Formule: (TP + TN) / Total
       - Attention: peut être trompeuse si déséquilibre de classes

    2. PRECISION (weighted): Exactitude des prédictions positives
       - Sur les films prédits comme "Drama", combien sont vraiment Drama?
       - Formule: TP / (TP + FP)
       - weighted: moyenne pondérée par le support de chaque classe

    3. RECALL (weighted): Capacité à détecter les positifs
       - Sur tous les vrais "Drama", combien a-t-on détecté?
       - Formule: TP / (TP + FN)
       - weighted: moyenne pondérée par le support de chaque classe

    4. F1-SCORE (weighted): Équilibre precision/recall
       - Moyenne harmonique de precision et recall
       - Formule: 2 * (precision * recall) / (precision + recall)
       - weighted: moyenne pondérée par le support de chaque classe

    5. MATRICE DE CONFUSION: Détail par classe
       - Visualisation complète des erreurs de classification
       - Lignes = vraies classes, Colonnes = prédictions

    Args:
        y_true: Labels réels (test set)
        y_pred: Prédictions du modèle
        model_name: Nom du modèle (pour affichage)

    Returns:
        Dict avec les 4 métriques numériques (confusion_matrix à part)

    Note: average='weighted' gère le déséquilibre des classes en
          pondérant chaque classe par son nombre d'échantillons

    Reference: Métriques d'évaluation (cours ML)
    """
    print(f"\n   Calcul des métriques pour {model_name}...")

    # 1. ACCURACY - Score global
    accuracy = accuracy_score(y_true, y_pred)

    # 2. PRECISION - Weighted average pour gérer déséquilibre
    # Sur les prédictions "positives", combien sont vraiment positives?
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    # 3. RECALL - Weighted average pour gérer déséquilibre
    # Sur tous les vrais positifs, combien a-t-on détecté?
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # 4. F1-SCORE - Weighted average (équilibre precision/recall)
    # Moyenne harmonique de precision et recall
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 5. MATRICE DE CONFUSION - Calculée mais retournée séparément
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    # Affichage des résultats
    print(f"   ✓ Accuracy:  {accuracy*100:.2f}%")
    print(f"   ✓ Precision: {precision*100:.2f}% (weighted)")
    print(f"   ✓ Recall:    {recall*100:.2f}% (weighted)")
    print(f"   ✓ F1-Score:  {f1*100:.2f}% (weighted)")

    return metrics


def evaluate_all_models(models: Dict[str, Any],
                       X_test: np.ndarray,
                       y_test: np.ndarray) -> pd.DataFrame:
    """
    Évalue tous les modèles et compile les résultats

    Génère un tableau comparatif avec les 4 métriques pour chaque modèle.
    La matrice de confusion sera générée séparément pour le meilleur modèle.

    Args:
        models: Dictionnaire {nom_modèle: modèle_entraîné}
        X_test: Features de test (JAMAIS vues pendant l'entraînement)
        y_test: Labels de test

    Returns:
        pd.DataFrame: Tableau comparatif avec colonnes:
            - Model: Nom du modèle
            - Accuracy: Score global
            - Precision: Weighted average
            - Recall: Weighted average
            - F1-Score: Weighted average
    """
    print("\n" + "="*60)
    print("ÉVALUATION DE TOUS LES MODÈLES")
    print("="*60)

    print(f"\n   Évaluation sur {len(y_test):,} échantillons de test")
    print(f"   Dimension features: {X_test.shape[1]:,}")

    results = []

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        # Prédictions sur le test set
        y_pred = model.predict(X_test)

        # Calcul des métriques
        metrics = calculate_all_metrics(y_test, y_pred, model_name)

        # Ajouter le nom du modèle
        metrics['Model'] = model_name
        results.append(metrics)

    # Créer le DataFrame
    df_results = pd.DataFrame(results)

    # Réorganiser les colonnes
    df_results = df_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

    # Trier par F1-Score (meilleur critère pour déséquilibre)
    df_results = df_results.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print("\n" + "="*60)
    print("TABLEAU COMPARATIF DES PERFORMANCES")
    print("="*60)
    print(df_results.to_string(index=False))

    return df_results


def plot_metrics_comparison(results_df: pd.DataFrame,
                           output_path: str) -> None:
    """
    Crée un bar chart comparant les 4 métriques par modèle

    Visualisation avec:
        - 4 groupes (Accuracy, Precision, Recall, F1-Score)
        - 3 barres par groupe (Naive Bayes, SVM, Random Forest)
        - Couleurs distinctes par modèle

    Args:
        results_df: DataFrame avec résultats de evaluate_all_models()
        output_path: Chemin de sauvegarde (outputs/metrics_comparison.png)
    """
    print("\n" + "="*60)
    print("CRÉATION DU GRAPHIQUE COMPARATIF")
    print("="*60)

    # Préparer les données
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = results_df['Model'].tolist()

    # Configuration de la figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Positions des barres
    x = np.arange(len(metrics))
    width = 0.25  # Largeur de chaque barre

    # Couleurs distinctes pour chaque modèle
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Bleu, Rouge, Vert

    # Créer les barres groupées
    for i, (model, color) in enumerate(zip(models, colors)):
        values = results_df[results_df['Model'] == model][metrics].values[0] * 100
        offset = width * (i - 1)  # Centrer les barres
        bars = ax.bar(x + offset, values, width, label=model,
                     color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Ajouter les valeurs sur les barres
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Configuration des axes
    ax.set_xlabel('Métriques de Performance', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comparaison des Performances - 3 Modèles × 4 Métriques',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Sauvegarder
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n   ✓ Graphique comparatif sauvegardé: {output_file}")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         classes: List[str],
                         model_name: str,
                         output_path: str) -> None:
    """
    Génère une heatmap de la matrice de confusion

    La matrice de confusion montre:
        - Lignes: Vraies classes (genres réels)
        - Colonnes: Classes prédites
        - Diagonale: Prédictions correctes
        - Hors diagonale: Erreurs de classification

    Args:
        y_true: Labels réels
        y_pred: Prédictions
        classes: Noms des genres (pour axes)
        model_name: Nom du modèle (pour titre)
        output_path: Chemin de sauvegarde (outputs/confusion_matrix_*.png)

    Reference: Matrice de confusion (cours évaluation)
    """
    print(f"\n   Génération de la matrice de confusion pour {model_name}...")

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Créer la figure
    plt.figure(figsize=(10, 8))

    # Créer la heatmap avec seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes,
               cbar_kws={'label': 'Nombre de Prédictions'},
               linewidths=0.5, linecolor='gray')

    plt.xlabel('Classe Prédite', fontsize=13, fontweight='bold')
    plt.ylabel('Classe Réelle', fontsize=13, fontweight='bold')
    plt.title(f'Matrice de Confusion - {model_name}',
             fontsize=15, fontweight='bold', pad=20)

    # Ajuster les labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Sauvegarder
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Matrice de confusion sauvegardée: {output_file}")

    # Analyse de la matrice
    total = cm.sum()
    correct = np.trace(cm)  # Somme de la diagonale
    accuracy_from_cm = correct / total

    print(f"\n   ANALYSE DE LA MATRICE:")
    print(f"   - Prédictions correctes (diagonale): {correct}/{total} ({accuracy_from_cm*100:.2f}%)")
    print(f"   - Erreurs de classification: {total - correct}")


def save_classification_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              classes: List[str],
                              model_name: str,
                              output_path: str) -> None:
    """
    Sauvegarde le rapport de classification détaillé

    Le rapport contient pour CHAQUE classe:
        - Precision: Exactitude des prédictions pour cette classe
        - Recall: Capacité à détecter cette classe
        - F1-Score: Équilibre precision/recall
        - Support: Nombre d'échantillons de cette classe dans le test

    Args:
        y_true: Labels réels
        y_pred: Prédictions
        classes: Noms des genres
        model_name: Nom du modèle
        output_path: Chemin de sauvegarde (outputs/classification_report_*.txt)
    """
    print(f"\n   Génération du rapport de classification pour {model_name}...")

    # Générer le rapport
    report = classification_report(y_true, y_pred,
                                  target_names=classes,
                                  digits=4,
                                  zero_division=0)

    # Sauvegarder
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"RAPPORT DE CLASSIFICATION - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write("Pour chaque classe:\n")
        f.write("  - Precision: Sur les prédictions de cette classe, combien sont correctes?\n")
        f.write("  - Recall: Sur tous les vrais échantillons de cette classe, combien détectés?\n")
        f.write("  - F1-Score: Moyenne harmonique de precision et recall\n")
        f.write("  - Support: Nombre d'échantillons réels de cette classe\n\n")
        f.write("-"*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
        f.write("Note: weighted avg = moyenne pondérée par le support de chaque classe\n")
        f.write("="*60 + "\n")

    print(f"   ✓ Rapport sauvegardé: {output_file}")


def identify_best_model(results_df: pd.DataFrame) -> str:
    """
    Identifie le meilleur modèle basé sur le F1-Score

    Le F1-Score est le meilleur critère car:
        - Équilibre precision et recall
        - Plus robuste que accuracy en cas de déséquilibre
        - Moyenne harmonique (pénalise les déséquilibres P/R)

    Args:
        results_df: DataFrame avec résultats

    Returns:
        str: Nom du meilleur modèle
    """
    # Le DataFrame est déjà trié par F1-Score décroissant
    best_model = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['F1-Score']

    print("\n" + "="*60)
    print("IDENTIFICATION DU MEILLEUR MODÈLE")
    print("="*60)
    print(f"\n   Critère de sélection: F1-Score (weighted)")
    print(f"   Meilleur modèle: {best_model}")
    print(f"   F1-Score: {best_f1*100:.2f}%")

    return best_model


def evaluate_and_save_all(models: Dict[str, Any],
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         class_names: List[str]) -> pd.DataFrame:
    """
    Pipeline complet d'évaluation

    Génère:
        1. Tableau comparatif des 4 métriques
        2. Graphique de comparaison
        3. Matrice de confusion du meilleur modèle
        4. Rapport détaillé du meilleur modèle

    Args:
        models: Dictionnaire {nom_modèle: modèle_entraîné}
        X_test: Features de test
        y_test: Labels de test
        class_names: Noms des genres (5 classes)

    Returns:
        pd.DataFrame: Tableau comparatif des résultats
    """
    print("\n" + "="*70)
    print("ÉVALUATION COMPLÈTE - LES 5 MÉTRIQUES DE PERFORMANCE")
    print("="*70)

    # 1. Évaluer tous les modèles (Accuracy, Precision, Recall, F1)
    results_df = evaluate_all_models(models, X_test, y_test)

    # 2. Sauvegarder le tableau comparatif
    results_df.to_csv('outputs/metrics_results.csv', index=False)
    print(f"\n   ✓ Résultats sauvegardés: outputs/metrics_results.csv")

    # 3. Graphique de comparaison
    plot_metrics_comparison(results_df, 'outputs/metrics_comparison.png')

    # 4. Identifier le meilleur modèle
    best_model_name = identify_best_model(results_df)
    best_model = models[best_model_name]

    # 5. Matrice de confusion pour le meilleur modèle
    y_pred_best = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_best, class_names,
                         best_model_name,
                         'outputs/confusion_matrix_best_model.png')

    # 6. Rapport détaillé pour le meilleur modèle
    save_classification_report(y_test, y_pred_best, class_names,
                              best_model_name,
                              'outputs/classification_report_best_model.txt')

    print("\n" + "="*70)
    print("✓ ÉVALUATION COMPLÈTE TERMINÉE")
    print("="*70)
    print("\nFichiers générés:")
    print("  1. outputs/metrics_results.csv - Tableau comparatif")
    print("  2. outputs/metrics_comparison.png - Graphique comparatif")
    print("  3. outputs/confusion_matrix_best_model.png - Matrice de confusion")
    print("  4. outputs/classification_report_best_model.txt - Rapport détaillé")

    return results_df


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE EVALUATION")
    print("="*60)

    # Simuler des données pour test
    print("\n⚠ Ce module nécessite des modèles entraînés du module models.py")
    print("Pour un test complet, exécutez main.py")

    # Exemple avec prédictions simulées
    print("\n--- Test avec données simulées ---")
    np.random.seed(42)

    # Créer des vrais labels et prédictions
    classes = ['Drama', 'Comedy', 'Action', 'Horror', 'Thriller']
    n_samples = 125

    y_true = np.random.choice(classes, size=n_samples)

    # Simuler des prédictions avec différentes accuracies
    predictions = {
        'Naive Bayes': np.random.choice(classes, size=n_samples),
        'SVM': np.random.choice(classes, size=n_samples),
        'Random Forest': np.random.choice(classes, size=n_samples)
    }

    # Calculer les métriques pour chaque modèle
    results = []
    for model_name, y_pred in predictions.items():
        metrics = calculate_all_metrics(y_true, y_pred, model_name)
        metrics['Model'] = model_name
        results.append(metrics)

    # Créer le DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    df_results = df_results.sort_values('F1-Score', ascending=False).reset_index(drop=True)

    print("\n" + "="*60)
    print("RÉSULTATS DE TEST")
    print("="*60)
    print(df_results.to_string(index=False))

    # Générer les visualisations de test
    plot_metrics_comparison(df_results, 'outputs/metrics_comparison_test.png')
    plot_confusion_matrix(y_true, predictions['Random Forest'], classes,
                         'Random Forest (Test)',
                         'outputs/confusion_matrix_test.png')

    print("\n" + "="*60)
    print("✓ MODULE EVALUATION TESTÉ AVEC SUCCÈS")
    print("="*60)
