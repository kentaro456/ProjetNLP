#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Détection d'Anomalies - Isolation Forest

Ce module gère la détection et la suppression d'outliers dans le jeu d'entraînement
en utilisant l'algorithme Isolation Forest, afin d'améliorer la robustesse des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


def detect_outliers_isolation_forest(X_train: np.ndarray,
                                    contamination: float = 0.1,
                                    random_state: int = 42) -> np.ndarray:
    """
    Détecte les outliers avec Isolation Forest (Slide 155)

    Principe de l'Isolation Forest:
        - Construit des arbres de décision aléatoires
        - Les anomalies sont isolées plus rapidement (moins de splits)
        - contamination = proportion attendue d'outliers dans les données

    IMPORTANT - Application Train Only:
        Cette fonction doit être appelée UNIQUEMENT sur le train set.
        Le test set ne doit jamais être nettoyé d'outliers car il
        représente des données futures non vues en production.

    Args:
        X_train: Features d'entraînement (matrice TF-IDF sparse ou dense)
        contamination: Proportion d'outliers attendue (default=0.1 = 10%)
        random_state: Seed pour reproductibilité (default=42)

    Returns:
        np.ndarray: Mask booléen (True = inlier à garder, False = outlier à supprimer)

    Reference: Slide 155 - Isolation Forest pour détection d'anomalies

    Paramètres Isolation Forest:
        - contamination=0.1: On s'attend à ~10% d'outliers
        - n_estimators=100: Nombre d'arbres dans l'ensemble
        - max_samples='auto': Taille des échantillons pour chaque arbre
        - random_state=42: Reproductibilité
    """
    print("\n" + "="*60)
    print("DÉTECTION D'ANOMALIES - ISOLATION FOREST (Slide 155)")
    print("="*60)

    print(f"\n1. CONFIGURATION:")
    print(f"   - contamination: {contamination} ({contamination*100:.0f}% outliers attendus)")
    print(f"   - n_estimators: 100 arbres")
    print(f"   - random_state: {random_state}")

    # Initialiser l'Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,  # Proportion d'outliers attendue
        n_estimators=100,  # Nombre d'arbres
        max_samples='auto',  # Taille échantillon par arbre
        random_state=random_state,
        n_jobs=-1,  # Utilise tous les CPUs disponibles
        verbose=0
    )

    # Fit et prédiction
    print(f"\n2. FITTING ISOLATION FOREST:")
    print(f"   Analyse de {X_train.shape[0]:,} échantillons...")
    print(f"   Dimension des features: {X_train.shape[1]:,}")

    # Prédiction: 1 = inlier, -1 = outlier
    predictions = iso_forest.fit_predict(X_train)

    # Convertir en mask booléen (True = keep, False = remove)
    inlier_mask = predictions == 1

    # Statistiques
    n_outliers = np.sum(~inlier_mask)
    n_inliers = np.sum(inlier_mask)
    outlier_percentage = (n_outliers / len(predictions)) * 100

    print(f"\n3. RÉSULTATS DE LA DÉTECTION:")
    print(f"   Inliers (à garder): {n_inliers:,} ({(n_inliers/len(predictions))*100:.2f}%)")
    print(f"   Outliers (à supprimer): {n_outliers:,} ({outlier_percentage:.2f}%)")

    # Scores d'anomalie (plus négatif = plus anormal)
    anomaly_scores = iso_forest.score_samples(X_train)
    print(f"\n4. SCORES D'ANOMALIE:")
    print(f"   Score minimum (plus anormal): {anomaly_scores.min():.4f}")
    print(f"   Score maximum (plus normal): {anomaly_scores.max():.4f}")
    print(f"   Score moyen: {anomaly_scores.mean():.4f}")

    return inlier_mask


def remove_outliers(X_train: np.ndarray,
                   y_train: pd.Series,
                   outlier_mask: np.ndarray) -> Tuple:
    """
    Supprime les outliers détectés du train set

    Args:
        X_train: Features d'entraînement (TF-IDF matrix)
        y_train: Labels d'entraînement (genres)
        outlier_mask: Mask booléen (True = garder, False = supprimer)

    Returns:
        Tuple: (X_train_clean, y_train_clean)
        - X_train_clean: Features sans outliers
        - y_train_clean: Labels sans outliers (avec index reset)
    """
    print("\n" + "="*60)
    print("SUPPRESSION DES OUTLIERS")
    print("="*60)

    # Dimensions avant nettoyage
    n_before = X_train.shape[0]

    # Filtrer les données
    X_train_clean = X_train[outlier_mask]
    y_train_clean = y_train[outlier_mask].reset_index(drop=True)

    # Dimensions après nettoyage
    n_after = X_train_clean.shape[0]
    n_removed = n_before - n_after

    print(f"\n1. STATISTIQUES:")
    print(f"   Avant nettoyage: {n_before:,} échantillons")
    print(f"   Après nettoyage: {n_after:,} échantillons")
    print(f"   Supprimés: {n_removed:,} ({(n_removed/n_before)*100:.2f}%)")

    # Vérifier les dimensions
    print(f"\n2. DIMENSIONS:")
    print(f"   X_train_clean: {X_train_clean.shape}")
    print(f"   y_train_clean: {len(y_train_clean)}")

    return X_train_clean, y_train_clean


def visualize_outliers(X_train: np.ndarray,
                      outlier_mask: np.ndarray,
                      output_path: str) -> None:
    """
    Visualise la distribution des outliers avec PCA 2D

    Crée deux visualisations:
        1. Bar chart: Inliers vs Outliers
        2. PCA 2D scatter: Points colorés par statut (inlier/outlier)

    Args:
        X_train: Features d'entraînement (TF-IDF matrix)
        outlier_mask: Mask booléen (True = inlier, False = outlier)
        output_path: Chemin de sauvegarde (outputs/outlier_detection.png)
    """
    print("\n" + "="*60)
    print("VISUALISATION DES OUTLIERS")
    print("="*60)

    # Créer une figure avec 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot 1: Bar Chart Inliers vs Outliers ---
    n_inliers = np.sum(outlier_mask)
    n_outliers = np.sum(~outlier_mask)

    categories = ['Inliers\n(à garder)', 'Outliers\n(à supprimer)']
    counts = [n_inliers, n_outliers]
    colors = ['#2ecc71', '#e74c3c']  # Vert et Rouge

    axes[0].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Nombre d\'échantillons', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution Inliers vs Outliers', fontsize=14, fontweight='bold')

    # Ajouter les valeurs sur les barres
    for i, (count, color) in enumerate(zip(counts, colors)):
        percentage = (count / (n_inliers + n_outliers)) * 100
        axes[0].text(i, count + 50, f'{count:,}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[0].grid(axis='y', alpha=0.3)

    # --- Subplot 2: PCA 2D Scatter ---
    print("\n   Application PCA pour visualisation 2D...")

    # Réduire à 2 dimensions avec PCA
    pca = PCA(n_components=2, random_state=42)

    # Convertir en dense si sparse
    if hasattr(X_train, 'toarray'):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train

    X_pca = pca.fit_transform(X_dense)

    # Créer labels pour coloration
    labels = np.where(outlier_mask, 'Inlier', 'Outlier')

    # Scatter plot
    for label, color, marker in [('Inlier', '#2ecc71', 'o'),
                                   ('Outlier', '#e74c3c', 'X')]:
        mask = labels == label
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=color, label=label, alpha=0.6,
                       s=30, marker=marker, edgecolors='black', linewidths=0.5)

    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                      fontsize=12, fontweight='bold')
    axes[1].set_title('Visualisation PCA 2D des Outliers', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(alpha=0.3)

    print(f"   Variance expliquée PC1+PC2: {sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Sauvegarder
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n   ✓ Visualisation sauvegardée: {output_file}")


def save_outlier_report(outlier_mask: np.ndarray,
                       y_train: pd.Series,
                       output_path: str) -> None:
    """
    Génère un rapport détaillé sur les outliers détectés

    Analyse:
        - Total outliers détectés
        - Distribution des outliers par genre
        - Pourcentage supprimé par genre

    Args:
        outlier_mask: Mask booléen (True = inlier, False = outlier)
        y_train: Labels d'entraînement (genres)
        output_path: Chemin de sauvegarde (outputs/outlier_report.txt)
    """
    print("\n" + "="*60)
    print("GÉNÉRATION DU RAPPORT D'OUTLIERS")
    print("="*60)

    # Statistiques globales
    n_total = len(outlier_mask)
    n_inliers = np.sum(outlier_mask)
    n_outliers = np.sum(~outlier_mask)

    # Analyse par genre
    genres = y_train.unique()
    genre_stats = []

    print("\n   ANALYSE PAR GENRE:")
    for genre in sorted(genres):
        genre_mask = y_train == genre
        n_genre_total = np.sum(genre_mask)
        n_genre_outliers = np.sum(~outlier_mask & genre_mask)
        n_genre_inliers = np.sum(outlier_mask & genre_mask)
        outlier_percentage = (n_genre_outliers / n_genre_total) * 100

        genre_stats.append({
            'Genre': genre,
            'Total': n_genre_total,
            'Inliers': n_genre_inliers,
            'Outliers': n_genre_outliers,
            'Outlier_Percentage': outlier_percentage
        })

        print(f"   - {genre}:")
        print(f"     Total: {n_genre_total:,} | Outliers: {n_genre_outliers:,} ({outlier_percentage:.2f}%)")

    # Créer DataFrame pour le rapport
    df_stats = pd.DataFrame(genre_stats)

    # Sauvegarder le rapport
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RAPPORT DE DÉTECTION D'OUTLIERS - ISOLATION FOREST\n")
        f.write("="*60 + "\n\n")

        f.write("1. STATISTIQUES GLOBALES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total échantillons: {n_total:,}\n")
        f.write(f"Inliers (gardés): {n_inliers:,} ({(n_inliers/n_total)*100:.2f}%)\n")
        f.write(f"Outliers (supprimés): {n_outliers:,} ({(n_outliers/n_total)*100:.2f}%)\n\n")

        f.write("2. STATISTIQUES PAR GENRE\n")
        f.write("-" * 60 + "\n")
        for _, row in df_stats.iterrows():
            f.write(f"\n{row['Genre']}:\n")
            f.write(f"  Total: {row['Total']:,}\n")
            f.write(f"  Inliers: {row['Inliers']:,}\n")
            f.write(f"  Outliers: {row['Outliers']:,} ({row['Outlier_Percentage']:.2f}%)\n")

        f.write("\n" + "="*60 + "\n")
        f.write("Note: Les outliers ont été détectés sur le TRAIN SET uniquement.\n")
        f.write("Le TEST SET reste intact pour une évaluation non biaisée.\n")
        f.write("="*60 + "\n")

    print(f"\n   ✓ Rapport sauvegardé: {output_file}")

    # Vérifier si la détection est équilibrée
    max_outlier_pct = df_stats['Outlier_Percentage'].max()
    min_outlier_pct = df_stats['Outlier_Percentage'].min()
    pct_diff = max_outlier_pct - min_outlier_pct

    print(f"\n   ÉQUILIBRE DE LA DÉTECTION:")
    print(f"   - Genre avec le plus d'outliers: {max_outlier_pct:.2f}%")
    print(f"   - Genre avec le moins d'outliers: {min_outlier_pct:.2f}%")
    print(f"   - Différence: {pct_diff:.2f}%")

    if pct_diff > 5:
        print(f"   ⚠ Attention: Détection déséquilibrée entre genres (>{pct_diff:.1f}% diff)")
    else:
        print(f"   ✓ Détection équilibrée entre genres (<{pct_diff:.1f}% diff)")


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE OUTLIER DETECTION")
    print("="*60)

    # Simuler des données pour test
    print("\n⚠ Ce module nécessite des features TF-IDF du module features.py")
    print("Pour un test complet, exécutez main.py")

    # Exemple avec données aléatoires
    print("\n--- Test avec données simulées ---")
    np.random.seed(42)

    # Créer des données synthétiques (500 échantillons, 100 features)
    X_train_synthetic = np.random.randn(500, 100)
    y_train_synthetic = pd.Series(['Drama'] * 150 + ['Comedy'] * 150 +
                                  ['Action'] * 100 + ['Horror'] * 50 + ['Thriller'] * 50)

    # Ajouter quelques outliers évidents
    X_train_synthetic[:20] = np.random.randn(20, 100) * 5  # Outliers

    # Test de détection
    outlier_mask = detect_outliers_isolation_forest(X_train_synthetic, contamination=0.1)
    X_clean, y_clean = remove_outliers(X_train_synthetic, y_train_synthetic, outlier_mask)
    visualize_outliers(X_train_synthetic, outlier_mask, 'outputs/outlier_detection_test.png')
    save_outlier_report(outlier_mask, y_train_synthetic, 'outputs/outlier_report_test.txt')

    print("\n" + "="*60)
    print("✓ MODULE OUTLIER DETECTION TESTÉ AVEC SUCCÈS")
    print("="*60)
