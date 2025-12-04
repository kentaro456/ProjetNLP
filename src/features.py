#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'Extraction de Features - TF-IDF et Train/Test Split

Ce module gère la division stratifiée des données (Train/Test Split) et la vectorisation TF-IDF
(Term Frequency - Inverse Document Frequency) en évitant le data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def split_data(X: pd.Series,
               y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple:
    """
    Divise les données en ensembles d'entraînement et de test avec stratification

    La stratification garantit que la distribution des genres est préservée
    dans les ensembles train et test, ce qui est essentiel pour éviter
    le biais d'échantillonnage.

    Args:
        X: Textes (colonne 'Plot')
        y: Labels (colonne 'Genre')
        test_size: Proportion du test set (default=0.2 pour 80/20 split)
        random_state: Seed pour reproductibilité (default=42)

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)

    Reference: Validation croisée (Slide cours ML)

    Note: stratify=y garantit la même distribution de genres dans train/test
    """
    print("\n" + "="*60)
    print("DIVISION TRAIN/TEST STRATIFIÉE")
    print("="*60)

    # Division stratifiée
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratification pour préserver la distribution des genres
    )

    print(f"\n1. DIMENSIONS DES ENSEMBLES:")
    print(f"   Train set: {len(X_train):,} échantillons ({(1-test_size)*100:.0f}%)")
    print(f"   Test set:  {len(X_test):,} échantillons ({test_size*100:.0f}%)")
    print(f"   Total:     {len(X):,} échantillons")

    # Vérification de la stratification
    print(f"\n2. VÉRIFICATION DE LA STRATIFICATION:")
    print(f"\n   Distribution TRAIN:")
    train_dist = y_train.value_counts(normalize=True).sort_index()
    for genre, prop in train_dist.items():
        count = (y_train == genre).sum()
        print(f"   - {genre}: {count:,} ({prop*100:.2f}%)")

    print(f"\n   Distribution TEST:")
    test_dist = y_test.value_counts(normalize=True).sort_index()
    for genre, prop in test_dist.items():
        count = (y_test == genre).sum()
        print(f"   - {genre}: {count:,} ({prop*100:.2f}%)")

    # Sauvegarder les informations sur le split
    split_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_ratio': test_size,
        'random_state': random_state,
        'train_distribution': train_dist.to_dict(),
        'test_distribution': test_dist.to_dict()
    }

    output_path = Path("data/processed/train_test_split_info.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("INFORMATIONS SUR LE TRAIN/TEST SPLIT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train size: {split_info['train_size']:,}\n")
        f.write(f"Test size: {split_info['test_size']:,}\n")
        f.write(f"Test ratio: {split_info['test_ratio']}\n")
        f.write(f"Random state: {split_info['random_state']}\n\n")
        f.write("Distribution Train:\n")
        for genre, prop in split_info['train_distribution'].items():
            f.write(f"  {genre}: {prop*100:.2f}%\n")
        f.write("\nDistribution Test:\n")
        for genre, prop in split_info['test_distribution'].items():
            f.write(f"  {genre}: {prop*100:.2f}%\n")

    print(f"\n   ✓ Informations sauvegardées: {output_path}")

    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train: pd.Series,
                         X_test: pd.Series,
                         max_features: int = 5000,
                         ngram_range: Tuple = (1, 2)) -> Tuple:
    """
    Transforme les textes en features TF-IDF (Term Frequency - Inverse Document Frequency)

    TF-IDF mesure l'importance d'un mot dans un document par rapport à
    l'ensemble du corpus. Les mots fréquents partout (ex: 'the') ont un
    score faible, tandis que les mots discriminants ont un score élevé.

    IMPORTANT - Prévention du Data Leakage:
        1. Fit sur X_train UNIQUEMENT
        2. Transform X_train et X_test avec le vectorizer fitté sur train
        3. Le test set ne doit JAMAIS influencer le vocabulaire ou les IDF scores

    Args:
        X_train: Textes d'entraînement
        X_test: Textes de test
        max_features: Nombre maximum de features (default=5000)
        ngram_range: Plage de n-grams (default=(1,2) = unigrams + bigrams)

    Returns:
        Tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
        - X_train_tfidf: scipy sparse matrix (shape: n_train × 5000)
        - X_test_tfidf: scipy sparse matrix (shape: n_test × 5000)
        - vectorizer: Objet TfidfVectorizer fitté

    Reference: Feature extraction textuelle (NLP)

    Paramètres TF-IDF:
        - max_features=5000: Garde les 5000 mots les plus importants
        - ngram_range=(1,2): Capture unigrams ("action") et bigrams ("science fiction")
        - min_df=2: Ignore les termes apparaissant dans moins de 2 documents
        - max_df=0.8: Ignore les termes apparaissant dans plus de 80% des documents
        - sublinear_tf=True: Applique log scaling au TF (améliore performance)
    """
    print("\n" + "="*60)
    print("VECTORISATION TF-IDF")
    print("="*60)

    # Initialisation du vectorizer
    print(f"\n1. CONFIGURATION DU VECTORIZER:")
    print(f"   - max_features: {max_features}")
    print(f"   - ngram_range: {ngram_range} (unigrams + bigrams)")
    print(f"   - min_df: 2 (ignore termes très rares)")
    print(f"   - max_df: 0.8 (ignore termes très fréquents)")
    print(f"   - sublinear_tf: True (log scaling)")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,  # Ignore les termes apparaissant dans moins de 2 documents
        max_df=0.8,  # Ignore les termes apparaissant dans plus de 80% des documents
        sublinear_tf=True,  # Applique log scaling (1 + log(tf))
        strip_accents='unicode',
        lowercase=True,  # Déjà fait dans preprocessing, mais par sécurité
        stop_words=None  # Déjà supprimés dans preprocessing
    )

    # 2. Fit sur TRAIN uniquement (CRITIQUE pour éviter data leakage)
    print(f"\n2. FITTING SUR LE TRAIN SET:")
    print(f"   Fitting du vectorizer sur {len(X_train):,} documents...")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 3. Transform du test set avec le vectorizer fitté
    print(f"\n3. TRANSFORMATION DU TEST SET:")
    print(f"   Transformation de {len(X_test):,} documents...")
    X_test_tfidf = vectorizer.transform(X_test)

    # Statistiques sur les features
    print(f"\n4. STATISTIQUES DES FEATURES:")
    print(f"   Vocabulaire size: {len(vectorizer.vocabulary_):,} termes uniques")
    print(f"   X_train_tfidf shape: {X_train_tfidf.shape}")
    print(f"   X_test_tfidf shape: {X_test_tfidf.shape}")
    print(f"   Sparsité train: {(1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape))*100:.2f}%")
    print(f"   Sparsité test: {(1 - X_test_tfidf.nnz / np.prod(X_test_tfidf.shape))*100:.2f}%")

    # Analyse du vocabulaire
    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    # Top features par score IDF (mots les plus discriminants)
    top_n = 50
    top_indices = np.argsort(idf_scores)[-top_n:][::-1]
    top_features = [(feature_names[i], idf_scores[i]) for i in top_indices]

    print(f"\n5. TOP {top_n} FEATURES PAR SCORE IDF (mots les plus discriminants):")
    for i, (feature, idf) in enumerate(top_features[:10], 1):
        print(f"   {i:2d}. {feature}: {idf:.4f}")

    # Sauvegarder les top features dans un CSV
    save_feature_info(vectorizer, top_features, "outputs/tfidf_features.csv")

    print(f"\n   Note: Les scores IDF élevés = termes discriminants (rares)")
    print(f"         Les scores IDF faibles = termes communs (fréquents)")

    return X_train_tfidf, X_test_tfidf, vectorizer


def save_feature_info(vectorizer: TfidfVectorizer,
                     top_features: list,
                     output_path: str) -> None:
    """
    Sauvegarde les informations sur les features TF-IDF

    Saves:
        - Top features par score IDF
        - Vocabulaire size
        - Paramètres du vectorizer

    Args:
        vectorizer: Objet TfidfVectorizer fitté
        top_features: Liste de tuples (feature, idf_score)
        output_path: Chemin de sauvegarde (outputs/tfidf_features.csv)
    """
    # Créer un DataFrame avec les top features
    df_features = pd.DataFrame(top_features, columns=['Feature', 'IDF_Score'])
    df_features['Rank'] = range(1, len(df_features) + 1)
    df_features = df_features[['Rank', 'Feature', 'IDF_Score']]

    # Sauvegarder en CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n   ✓ Top features sauvegardées: {output_file}")

    # Sauvegarder les paramètres du vectorizer
    params_path = Path("outputs/tfidf_vectorizer_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("PARAMÈTRES DU TFIDF VECTORIZER\n")
        f.write("="*60 + "\n\n")
        f.write(f"Vocabulary size: {len(vectorizer.vocabulary_):,}\n")
        f.write(f"max_features: {vectorizer.max_features}\n")
        f.write(f"ngram_range: {vectorizer.ngram_range}\n")
        f.write(f"min_df: {vectorizer.min_df}\n")
        f.write(f"max_df: {vectorizer.max_df}\n")
        f.write(f"sublinear_tf: {vectorizer.sublinear_tf}\n")

    print(f"   ✓ Paramètres sauvegardés: {params_path}")


def get_feature_names(vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Récupère les noms des features du vectorizer

    Args:
        vectorizer: Objet TfidfVectorizer fitté

    Returns:
        np.ndarray: Tableau des noms de features
    """
    return vectorizer.get_feature_names_out()


def transform_new_text(vectorizer: TfidfVectorizer, text: str) -> sp.csr_matrix:
    """
    Transforme un nouveau texte en vecteur TF-IDF

    Utile pour faire des prédictions sur de nouvelles données

    Args:
        vectorizer: Objet TfidfVectorizer fitté
        text: Nouveau texte à transformer

    Returns:
        sp.csr_matrix: Vecteur TF-IDF sparse
    """
    return vectorizer.transform([text])


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE FEATURES")
    print("="*60)

    # Charger les données nettoyées
    data_path = Path("data/processed/cleaned_data.csv")

    if not data_path.exists():
        print("\n✗ ERREUR: Fichier cleaned_data.csv introuvable")
        print("  Veuillez d'abord exécuter preprocessing.py")
    else:
        df = pd.read_csv(data_path)
        print(f"\nDonnées chargées: {len(df):,} échantillons")

        # Test du pipeline
        X = df['Plot']
        y = df['Genre']

        # Train/Test Split
        X_train, X_test, y_train, y_test = split_data(X, y)

        # TF-IDF
        X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(
            X_train, X_test, max_features=5000
        )

        print("\n" + "="*60)
        print("✓ MODULE FEATURES TESTÉ AVEC SUCCÈS")
        print("="*60)
        print(f"\nMatrices TF-IDF créées:")
        print(f"  Train: {X_train_tfidf.shape}")
        print(f"  Test: {X_test_tfidf.shape}")
