#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Visualisation - PCA (Analyse en Composantes Principales)

Ce module gère la réduction de dimensionnalité avec PCA et la création de visualisations 2D
des données TF-IDF colorées par genre pour analyser la séparabilité des classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
from sklearn.decomposition import PCA


def apply_pca(X_train: np.ndarray,
             X_test: np.ndarray,
             n_components: int = 2,
             random_state: int = 42) -> Tuple:
    """
    Réduit la dimensionnalité avec PCA (Analyse en Composantes Principales)

    Principe de la PCA (Slide 129):
        - Transformation linéaire vers axes de variance maximale
        - Permet la visualisation 2D de données 5000-dimensionnelles
        - Préserve au maximum la variance des données originales
        - Les composantes principales sont orthogonales

    IMPORTANT - Prévention du Data Leakage:
        1. Fit sur X_train UNIQUEMENT (après removal outliers)
        2. Transform X_train et X_test avec le PCA fitté sur train

    Args:
        X_train: Features d'entraînement (après removal outliers)
        X_test: Features de test
        n_components: Nombre de composantes (2 pour visualisation)
        random_state: Seed pour reproductibilité (default=42)

    Returns:
        Tuple: (X_train_pca, X_test_pca, pca_model, explained_variance_ratio)
        - X_train_pca: Données train transformées (n_train × 2)
        - X_test_pca: Données test transformées (n_test × 2)
        - pca_model: Objet PCA fitté
        - explained_variance_ratio: % variance expliquée par composante

    Reference: Slide 129 - ACP/PCA pour réduction dimensionnelle
    """
    print("\n" + "="*60)
    print("RÉDUCTION DE DIMENSIONNALITÉ - PCA (Slide 129)")
    print("="*60)

    print(f"\n1. CONFIGURATION:")
    print(f"   - n_components: {n_components} (pour visualisation 2D)")
    print(f"   - random_state: {random_state}")
    print(f"\n   Dimensions originales:")
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")

    # Initialiser la PCA
    pca = PCA(n_components=n_components, random_state=random_state)

    # 2. Fit sur TRAIN uniquement (CRITIQUE pour éviter data leakage)
    print(f"\n2. FITTING SUR LE TRAIN SET:")
    print(f"   Fitting PCA sur {X_train.shape[0]:,} échantillons...")
    X_train_pca = pca.fit_transform(X_train)

    # 3. Transform du test set avec le PCA fitté
    print(f"\n3. TRANSFORMATION DU TEST SET:")
    print(f"   Transformation de {X_test.shape[0]:,} échantillons...")
    X_test_pca = pca.transform(X_test)

    # Variance expliquée
    explained_variance_ratio = pca.explained_variance_ratio_

    print(f"\n4. VARIANCE EXPLIQUÉE:")
    print(f"   PC1 (Composante 1): {explained_variance_ratio[0]*100:.2f}%")
    print(f"   PC2 (Composante 2): {explained_variance_ratio[1]*100:.2f}%")
    print(f"   Total (PC1+PC2): {sum(explained_variance_ratio)*100:.2f}%")

    # Dimensions après transformation
    print(f"\n5. DIMENSIONS APRÈS PCA:")
    print(f"   X_train_pca: {X_train_pca.shape}")
    print(f"   X_test_pca: {X_test_pca.shape}")
    print(f"\n   Réduction: {X_train.shape[1]} → {X_train_pca.shape[1]} features")

    # Analyse des composantes principales
    print(f"\n6. ANALYSE DES COMPOSANTES:")
    print(f"   Les {n_components} premières composantes capturent")
    print(f"   {sum(explained_variance_ratio)*100:.2f}% de la variance totale")

    if sum(explained_variance_ratio) < 0.2:
        print(f"\n   ⚠ Note: Variance expliquée faible (<20%)")
        print(f"      C'est normal pour des données textuelles TF-IDF")
        print(f"      qui sont très haute dimension et sparse.")

    return X_train_pca, X_test_pca, pca, explained_variance_ratio


def plot_pca_scatter(X_pca: np.ndarray,
                    y: pd.Series,
                    title: str,
                    output_path: str,
                    explained_variance: np.ndarray = None) -> None:
    """
    Crée un scatter plot 2D coloré par genre

    Visualisation professionnelle avec:
        - 5 couleurs distinctes pour les 5 genres
        - Légende claire
        - Axes annotés avec variance expliquée
        - Haute résolution (300 DPI)

    Args:
        X_pca: Données transformées PCA (2 composantes)
        y: Labels (genres)
        title: Titre du plot
        output_path: Chemin de sauvegarde
        explained_variance: Variance expliquée par chaque composante
    """
    print(f"\n   Création du scatter plot: {title}...")

    # Configuration du style
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Palette de couleurs distinctes pour 5 genres
    unique_genres = sorted(y.unique())
    n_genres = len(unique_genres)
    colors = sns.color_palette("Set2", n_genres)

    # Créer le scatter plot
    for i, genre in enumerate(unique_genres):
        mask = y == genre
        plt.scatter(X_pca[mask, 0],
                   X_pca[mask, 1],
                   c=[colors[i]],
                   label=genre,
                   alpha=0.6,
                   s=40,
                   edgecolors='black',
                   linewidths=0.5)

    # Labels des axes avec variance expliquée
    if explained_variance is not None:
        xlabel = f'PC1 ({explained_variance[0]*100:.2f}% variance)'
        ylabel = f'PC2 ({explained_variance[1]*100:.2f}% variance)'
    else:
        xlabel = 'Composante Principale 1'
        ylabel = 'Composante Principale 2'

    plt.xlabel(xlabel, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)

    # Légende
    plt.legend(title='Genres', loc='best', fontsize=11,
              title_fontsize=12, framealpha=0.9)

    # Grille
    plt.grid(alpha=0.3, linestyle='--')

    # Sauvegarder
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Scatter plot sauvegardé: {output_file}")


def plot_variance_explained(pca_model: PCA,
                           output_path: str,
                           n_components_show: int = 10) -> None:
    """
    Visualise la variance expliquée par chaque composante

    Crée deux visualisations:
        1. Bar chart: Variance par composante
        2. Ligne: Variance cumulée

    Args:
        pca_model: Objet PCA fitté
        output_path: Chemin de sauvegarde (outputs/pca_variance_explained.png)
        n_components_show: Nombre de composantes à afficher (default=10)
    """
    print(f"\n   Création du graphique de variance expliquée...")

    # Récupérer la variance expliquée
    explained_variance = pca_model.explained_variance_ratio_

    # Limiter au nombre de composantes à afficher
    n_show = min(n_components_show, len(explained_variance))
    variance_to_plot = explained_variance[:n_show]
    cumulative_variance = np.cumsum(variance_to_plot)

    # Créer la figure avec 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot 1: Variance par composante ---
    components = [f'PC{i+1}' for i in range(n_show)]
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))

    axes[0].bar(components, variance_to_plot * 100, color=colors,
               alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Composante Principale', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Variance Expliquée (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Variance Expliquée par Composante',
                     fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Ajouter les valeurs sur les barres
    for i, (comp, var) in enumerate(zip(components, variance_to_plot)):
        axes[0].text(i, var * 100 + 0.2, f'{var*100:.2f}%',
                    ha='center', va='bottom', fontsize=9)

    # --- Subplot 2: Variance cumulée ---
    axes[1].plot(range(1, n_show + 1), cumulative_variance * 100,
                marker='o', color='steelblue', linewidth=2.5,
                markersize=8, markerfacecolor='orange',
                markeredgecolor='black', markeredgewidth=1.5)
    axes[1].fill_between(range(1, n_show + 1), cumulative_variance * 100,
                         alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Nombre de Composantes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Variance Cumulée (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Variance Cumulée Expliquée',
                     fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Ajouter une ligne horizontale à 80%
    axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2,
                   label='80% variance', alpha=0.7)
    axes[1].legend(loc='best', fontsize=11)

    # Ajouter les valeurs sur les points clés
    for i in [0, n_show-1]:  # Premier et dernier point
        axes[1].text(i+1, cumulative_variance[i] * 100 + 2,
                    f'{cumulative_variance[i]*100:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Sauvegarder
    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Graphique variance sauvegardé: {output_file}")


def create_all_visualizations(X_train_pca: np.ndarray,
                             X_test_pca: np.ndarray,
                             y_train: pd.Series,
                             y_test: pd.Series,
                             pca_model: PCA,
                             explained_variance: np.ndarray) -> None:
    """
    Crée toutes les visualisations PCA

    Génère:
        1. Scatter plot train set
        2. Scatter plot test set
        3. Graphique variance expliquée

    Args:
        X_train_pca: Train set transformé PCA
        X_test_pca: Test set transformé PCA
        y_train: Labels train
        y_test: Labels test
        pca_model: Objet PCA fitté
        explained_variance: Variance expliquée
    """
    print("\n" + "="*60)
    print("CRÉATION DES VISUALISATIONS PCA")
    print("="*60)

    # 1. Scatter plot train
    plot_pca_scatter(
        X_train_pca,
        y_train,
        'Visualisation PCA 2D - Train Set (après removal outliers)',
        'outputs/pca_scatter_train.png',
        explained_variance
    )

    # 2. Scatter plot test
    plot_pca_scatter(
        X_test_pca,
        y_test,
        'Visualisation PCA 2D - Test Set (données non vues)',
        'outputs/pca_scatter_test.png',
        explained_variance
    )

    # 3. Variance expliquée
    plot_variance_explained(
        pca_model,
        'outputs/pca_variance_explained.png',
        n_components_show=10
    )

    print("\n   ✓ Toutes les visualisations PCA créées avec succès")


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE VISUALIZATION")
    print("="*60)

    # Simuler des données pour test
    print("\n⚠ Ce module nécessite des features TF-IDF du module features.py")
    print("Pour un test complet, exécutez main.py")

    # Exemple avec données aléatoires
    print("\n--- Test avec données simulées ---")
    np.random.seed(42)

    # Créer des données synthétiques (500 échantillons train, 125 test, 100 features)
    X_train_synthetic = np.random.randn(500, 100)
    X_test_synthetic = np.random.randn(125, 100)

    y_train_synthetic = pd.Series(['Drama'] * 150 + ['Comedy'] * 150 +
                                  ['Action'] * 100 + ['Horror'] * 50 + ['Thriller'] * 50)
    y_test_synthetic = pd.Series(['Drama'] * 40 + ['Comedy'] * 40 +
                                 ['Action'] * 25 + ['Horror'] * 10 + ['Thriller'] * 10)

    # Test de PCA
    X_train_pca, X_test_pca, pca_model, variance_ratio = apply_pca(
        X_train_synthetic,
        X_test_synthetic,
        n_components=2
    )

    # Créer les visualisations
    create_all_visualizations(
        X_train_pca,
        X_test_pca,
        y_train_synthetic,
        y_test_synthetic,
        pca_model,
        variance_ratio
    )

    print("\n" + "="*60)
    print("✓ MODULE VISUALIZATION TESTÉ AVEC SUCCÈS")
    print("="*60)
