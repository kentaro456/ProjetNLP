#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Modélisation - Entraînement des Classifiers

Ce module gère l'entraînement, la sauvegarde et l'utilisation de 3 classificateurs :
Naive Bayes, SVM Linéaire et Random Forest.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def train_naive_bayes(X_train: np.ndarray,
                     y_train: pd.Series,
                     alpha: float = 1.0) -> MultinomialNB:
    """
    Entraîne un classificateur Naive Bayes

    Principe Naive Bayes:
        - Modèle probabiliste basé sur le théorème de Bayes
        - Assume l'indépendance des features (d'où "naive")
        - Très rapide et efficace pour les données textuelles
        - MultinomialNB adapté pour les comptages (TF-IDF)

    Args:
        X_train: Features d'entraînement (TF-IDF, après outlier removal)
        y_train: Labels d'entraînement (genres)
        alpha: Paramètre de lissage Laplace (default=1.0)

    Returns:
        MultinomialNB: Modèle entraîné

    Reference: Algorithmes de classification probabiliste

    Hyperparamètres:
        - alpha=1.0: Lissage de Laplace (gère les mots non vus)
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT: NAIVE BAYES (Baseline)")
    print("="*60)

    print(f"\n1. CONFIGURATION:")
    print(f"   - Modèle: MultinomialNB")
    print(f"   - alpha (Laplace smoothing): {alpha}")
    print(f"   - Données: {X_train.shape[0]:,} échantillons × {X_train.shape[1]:,} features")

    # Initialiser et entraîner
    print(f"\n2. ENTRAÎNEMENT EN COURS...")
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    # Précision sur le train (pour détecter overfitting)
    train_score = model.score(X_train, y_train)
    print(f"\n3. PERFORMANCE TRAIN:")
    print(f"   Accuracy train: {train_score*100:.2f}%")

    # Statistiques du modèle
    n_classes = len(model.classes_)
    print(f"\n4. STATISTIQUES:")
    print(f"   Nombre de classes: {n_classes}")
    print(f"   Classes: {list(model.classes_)}")

    print(f"\n   ✓ Naive Bayes entraîné avec succès")

    return model


def train_svm(X_train: np.ndarray,
             y_train: pd.Series,
             C: float = 1.0,
             max_iter: int = 1000,
             random_state: int = 42) -> LinearSVC:
    """
    Entraîne un classificateur SVM (Support Vector Machine)

    Principe SVM:
        - Trouve l'hyperplan optimal séparant les classes
        - Maximise la marge entre les classes
        - Kernel linéaire optimal pour données textuelles haute dimension
        - class_weight='balanced' gère le déséquilibre des classes

    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        C: Paramètre de régularisation (default=1.0)
        max_iter: Nombre max d'itérations (default=1000)
        random_state: Seed pour reproductibilité

    Returns:
        LinearSVC: Modèle entraîné

    Hyperparamètres:
        - C=1.0: Régularisation standard (balance marge/erreurs)
        - class_weight='balanced': Gère le déséquilibre des classes
        - max_iter=1000: Limite pour convergence
        - dual=False: Plus efficace pour n_samples > n_features
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT: SVM LINÉAIRE (Performance)")
    print("="*60)

    print(f"\n1. CONFIGURATION:")
    print(f"   - Modèle: LinearSVC (kernel linéaire)")
    print(f"   - C (régularisation): {C}")
    print(f"   - class_weight: 'balanced' (gère déséquilibre)")
    print(f"   - max_iter: {max_iter}")
    print(f"   - Données: {X_train.shape[0]:,} échantillons × {X_train.shape[1]:,} features")

    # Initialiser et entraîner
    print(f"\n2. ENTRAÎNEMENT EN COURS...")
    print(f"   (Cela peut prendre quelques minutes...)")

    model = LinearSVC(
        C=C,
        class_weight='balanced',  # Gère le déséquilibre des classes
        max_iter=max_iter,
        random_state=random_state,
        dual=False,  # Plus efficace pour notre cas
        verbose=0
    )

    model.fit(X_train, y_train)

    # Précision sur le train
    train_score = model.score(X_train, y_train)
    print(f"\n3. PERFORMANCE TRAIN:")
    print(f"   Accuracy train: {train_score*100:.2f}%")

    # Statistiques du modèle
    n_classes = len(model.classes_)
    print(f"\n4. STATISTIQUES:")
    print(f"   Nombre de classes: {n_classes}")
    print(f"   Classes: {list(model.classes_)}")
    print(f"   Itérations: {model.n_iter_}")

    print(f"\n   ✓ SVM entraîné avec succès")

    return model


def train_random_forest(X_train: np.ndarray,
                       y_train: pd.Series,
                       n_estimators: int = 100,
                       max_depth: int = None,
                       min_samples_split: int = 5,
                       min_samples_leaf: int = 2,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Entraîne un Random Forest (Méthode Ensemble - cours)

    Principe Random Forest (Méthode Ensemble):
        - Ensemble de multiples arbres de décision
        - Chaque arbre entraîné sur un sous-ensemble aléatoire
        - Prédiction finale = vote majoritaire
        - Réduit l'overfitting vs un seul arbre
        - Robuste au bruit et aux outliers

    Args:
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        n_estimators: Nombre d'arbres dans la forêt (default=100)
        max_depth: Profondeur max des arbres (default=None = illimitée)
        min_samples_split: Min échantillons pour split (default=5)
        min_samples_leaf: Min échantillons par feuille (default=2)
        random_state: Seed pour reproductibilité

    Returns:
        RandomForestClassifier: Modèle entraîné

    Reference: Méthodes Ensemble (cours Big Data)

    Hyperparamètres:
        - n_estimators=100: Nombre d'arbres (balance accuracy/temps)
        - max_depth=None: Arbres complètement développés
        - class_weight='balanced': Gère le déséquilibre
        - min_samples_split=5: Évite l'overfitting
        - min_samples_leaf=2: Feuilles avec au moins 2 échantillons
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT: RANDOM FOREST (Méthode Ensemble)")
    print("="*60)

    print(f"\n1. CONFIGURATION:")
    print(f"   - Modèle: RandomForestClassifier (Ensemble)")
    print(f"   - n_estimators (arbres): {n_estimators}")
    print(f"   - max_depth: {max_depth if max_depth else 'None (illimitée)'}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    print(f"   - class_weight: 'balanced' (gère déséquilibre)")
    print(f"   - Données: {X_train.shape[0]:,} échantillons × {X_train.shape[1]:,} features")

    # Initialiser et entraîner
    print(f"\n2. ENTRAÎNEMENT EN COURS:")
    print(f"   Construction de {n_estimators} arbres...")
    print(f"   (Cela peut prendre quelques minutes...)")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',  # Gère le déséquilibre
        random_state=random_state,
        n_jobs=-1,  # Utilise tous les CPUs
        verbose=0
    )

    model.fit(X_train, y_train)

    # Précision sur le train
    train_score = model.score(X_train, y_train)
    print(f"\n3. PERFORMANCE TRAIN:")
    print(f"   Accuracy train: {train_score*100:.2f}%")

    # Statistiques du modèle
    n_classes = len(model.classes_)
    print(f"\n4. STATISTIQUES:")
    print(f"   Nombre de classes: {n_classes}")
    print(f"   Classes: {list(model.classes_)}")
    print(f"   Nombre d'arbres: {len(model.estimators_)}")

    # Feature importance (top 10)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        print(f"\n5. TOP 10 FEATURES LES PLUS IMPORTANTES:")
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i:2d}. Feature {idx}: {importances[idx]:.6f}")

    print(f"\n   ✓ Random Forest entraîné avec succès")

    return model


def save_model(model: Any,
              model_name: str,
              output_dir: str = "models") -> None:
    """
    Sauvegarde un modèle entraîné avec joblib

    Args:
        model: Modèle scikit-learn entraîné
        model_name: Nom du fichier (ex: 'naive_bayes.pkl')
        output_dir: Dossier de sauvegarde (default='models')
    """
    # Créer le dossier si nécessaire
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Chemin complet
    file_path = output_path / model_name

    # Sauvegarder avec joblib (plus efficace que pickle)
    joblib.dump(model, file_path)

    # Taille du fichier
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB

    print(f"   ✓ Modèle sauvegardé: {file_path}")
    print(f"     Taille: {file_size:.2f} MB")


def load_model(model_path: str) -> Any:
    """
    Charge un modèle sauvegardé

    Args:
        model_path: Chemin vers le fichier .pkl

    Returns:
        Modèle chargé
    """
    model = joblib.load(model_path)
    print(f"   ✓ Modèle chargé: {model_path}")
    return model


def train_all_models(X_train: np.ndarray,
                    y_train: pd.Series) -> Dict[str, Any]:
    """
    Entraîne les 3 modèles et les sauvegarde

    Args:
        X_train: Features d'entraînement (après outlier removal)
        y_train: Labels d'entraînement

    Returns:
        Dict[str, Any]: Dictionnaire {nom_modèle: modèle_entraîné}
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DE TOUS LES MODÈLES")
    print("="*60)

    models = {}

    # 1. Naive Bayes
    print("\n[1/3] NAIVE BAYES")
    nb_model = train_naive_bayes(X_train, y_train)
    save_model(nb_model, 'naive_bayes.pkl')
    models['Naive Bayes'] = nb_model

    # 2. SVM
    print("\n[2/3] SVM LINÉAIRE")
    svm_model = train_svm(X_train, y_train)
    save_model(svm_model, 'svm.pkl')
    models['SVM'] = svm_model

    # 3. Random Forest
    print("\n[3/3] RANDOM FOREST")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'random_forest.pkl')
    models['Random Forest'] = rf_model

    print("\n" + "="*60)
    print("✓ TOUS LES MODÈLES ENTRAÎNÉS ET SAUVEGARDÉS")
    print("="*60)

    return models


def predict_all_models(models: Dict[str, Any],
                      X_test: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Génère les prédictions pour tous les modèles sur le test set

    Args:
        models: Dictionnaire {nom_modèle: modèle_entraîné}
        X_test: Features de test

    Returns:
        Dict[str, np.ndarray]: Dictionnaire {nom_modèle: prédictions}
    """
    print("\n" + "="*60)
    print("PRÉDICTIONS SUR LE TEST SET")
    print("="*60)

    predictions = {}

    for name, model in models.items():
        print(f"\n   Prédiction avec {name}...")
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        print(f"   ✓ {len(y_pred):,} prédictions générées")

    print("\n   ✓ Toutes les prédictions générées")

    return predictions


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE MODELS")
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

    X_test_synthetic = np.random.randn(125, 100)
    y_test_synthetic = pd.Series(['Drama'] * 40 + ['Comedy'] * 40 +
                                 ['Action'] * 25 + ['Horror'] * 10 + ['Thriller'] * 10)

    # S'assurer que les valeurs sont positives pour MultinomialNB
    X_train_synthetic = np.abs(X_train_synthetic)
    X_test_synthetic = np.abs(X_test_synthetic)

    # Entraîner tous les modèles
    models = train_all_models(X_train_synthetic, y_train_synthetic)

    # Générer les prédictions
    predictions = predict_all_models(models, X_test_synthetic)

    print("\n" + "="*60)
    print("✓ MODULE MODELS TESTÉ AVEC SUCCÈS")
    print("="*60)
    print(f"\nModèles entraînés: {list(models.keys())}")
    print(f"Prédictions générées pour {len(predictions)} modèles")
