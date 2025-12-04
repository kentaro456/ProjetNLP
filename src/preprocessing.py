#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Preprocessing - Acquisition et Nettoyage des Données

Ce module gère le téléchargement du dataset, l'identification des genres les plus fréquents,
le nettoyage textuel (lowercase, suppression caractères spéciaux, stopwords) et la sauvegarde des données.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from pathlib import Path

# Configuration NLTK pour les stopwords
import nltk
try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords


def download_dataset() -> str:
    """
    Charge le dataset depuis le fichier local

    Returns:
        str: Chemin absolu vers le fichier dataset.csv

    Reference: Acquisition des données Big Data
    """
    print("\n" + "="*60)
    print("CHARGEMENT DU DATASET")
    print("="*60)

    try:
        # Chemin vers le fichier CSV existant
        source_path = Path("C:/Users/rluka/OneDrive/Bureau/technique/wiki_movie_plots_deduped.csv")

        print(f"Chargement du dataset depuis: {source_path}")

        # Charger le CSV avec l'encodage correct
        df = pd.read_csv(source_path, encoding='latin-1')

        # Sauvegarder le dataset dans data/raw/
        output_path = Path("data/raw/dataset.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Dataset sauvegardé: {output_path.resolve()}")
        print(f"  Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")

        return str(output_path.resolve())

    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        raise


def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset et affiche les statistiques descriptives

    Args:
        filepath: Chemin vers dataset.csv

    Returns:
        pd.DataFrame: Dataset brut

    Affiche:
        - Nombre total d'échantillons
        - Distribution des genres
        - Exemples de résumés (plots)
    """
    print("\n" + "="*60)
    print("EXPLORATION DES DONNÉES")
    print("="*60)

    # Charger le dataset
    df = pd.read_csv(filepath)

    print(f"\n1. DIMENSIONS DU DATASET:")
    print(f"   Nombre total d'échantillons: {len(df):,}")
    print(f"   Nombre de colonnes: {df.shape[1]}")
    print(f"\n   Colonnes disponibles: {list(df.columns)}")

    # Afficher les premières lignes
    print(f"\n2. APERÇU DES DONNÉES:")
    print(df.head(3))

    # Distribution des genres
    if 'Genre' in df.columns:
        print(f"\n3. DISTRIBUTION DES GENRES:")
        genre_counts = df['Genre'].value_counts()
        print(f"   Nombre total de genres: {len(genre_counts)}")
        print(f"\n   Top 10 genres:")
        for genre, count in genre_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"   - {genre}: {count:,} ({percentage:.2f}%)")

        # Visualisation de la distribution des genres (tous genres)
        plt.figure(figsize=(12, 6))
        genre_counts.head(20).plot(kind='bar', color='steelblue')
        plt.title('Distribution des 20 Genres les Plus Fréquents (Dataset Original)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Nombre de Films', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_path = Path("outputs/genre_distribution_original.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n   ✓ Graphique sauvegardé: {output_path}")

    # Vérifier la présence de valeurs manquantes
    print(f"\n4. VALEURS MANQUANTES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   Aucune valeur manquante détectée")

    # Exemples de résumés (Plot)
    if 'Plot' in df.columns:
        print(f"\n5. EXEMPLE DE RÉSUMÉ (PLOT):")
        sample_plot = df['Plot'].iloc[0]
        print(f"   Genre: {df['Genre'].iloc[0]}")
        print(f"   Plot: {sample_plot[:200]}...")

    return df


def get_top_n_genres(df: pd.DataFrame, n: int = 5) -> List[str]:
    """
    Identifie les N genres les plus fréquents

    Args:
        df: DataFrame avec colonne 'Genre'
        n: Nombre de genres à retenir (default=5)

    Returns:
        List[str]: Liste des N genres les plus fréquents

    Reference: Préparation des données (cours Big Data)
    """
    print(f"\n" + "="*60)
    print(f"IDENTIFICATION DES TOP {n} GENRES")
    print("="*60)

    # Compter les occurrences de chaque genre
    genre_counts = df['Genre'].value_counts()

    # Sélectionner les top N
    top_genres = genre_counts.head(n).index.tolist()

    print(f"\nLes {n} genres sélectionnés:")
    for i, genre in enumerate(top_genres, 1):
        count = genre_counts[genre]
        percentage = (count / len(df)) * 100
        print(f"   {i}. {genre}: {count:,} films ({percentage:.2f}%)")

    # Statistiques sur la sélection
    total_selected = genre_counts.head(n).sum()
    percentage_coverage = (total_selected / len(df)) * 100
    print(f"\n   Total films conservés: {total_selected:,} / {len(df):,} ({percentage_coverage:.2f}%)")

    return top_genres


def clean_text(text: str) -> str:
    """
    Nettoie un texte individuel

    Steps:
        1. Conversion en minuscules (lowercase)
        2. Suppression des caractères spéciaux (regex)
        3. Suppression des espaces multiples
        4. Suppression des stopwords anglais (NLTK)

    Args:
        text: Texte brut

    Returns:
        str: Texte nettoyé

    Reference: Preprocessing textuel (NLP)
    """
    if not isinstance(text, str):
        return ""

    # 1. Conversion en minuscules
    text = text.lower()

    # 2. Suppression des caractères spéciaux (garde uniquement lettres, chiffres et espaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # 4. Suppression des stopwords anglais
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]

    text = ' '.join(words)

    return text.strip()


def preprocess_dataset(df: pd.DataFrame, top_genres: List[str]) -> pd.DataFrame:
    """
    Applique le preprocessing complet

    Steps:
        1. Filtrage: garde uniquement les top_genres
        2. Nettoyage de tous les résumés (Plot)
        3. Suppression des résumés vides après nettoyage
        4. Reset de l'index

    Args:
        df: DataFrame brut
        top_genres: Liste des genres à garder

    Returns:
        pd.DataFrame: Dataset nettoyé avec colonnes ['Plot', 'Genre']
    """
    print("\n" + "="*60)
    print("PREPROCESSING DES DONNÉES")
    print("="*60)

    # 1. Filtrage des genres
    print(f"\n1. FILTRAGE DES GENRES")
    print(f"   Avant filtrage: {len(df):,} films")
    df_filtered = df[df['Genre'].isin(top_genres)].copy()
    print(f"   Après filtrage: {len(df_filtered):,} films")
    print(f"   Films supprimés: {len(df) - len(df_filtered):,}")

    # 2. Nettoyage des textes
    print(f"\n2. NETTOYAGE DES RÉSUMÉS (PLOT)")
    print(f"   Application du nettoyage textuel...")

    # Garder uniquement les colonnes nécessaires
    if 'Plot' in df_filtered.columns and 'Genre' in df_filtered.columns:
        df_clean = df_filtered[['Plot', 'Genre']].copy()
    else:
        raise ValueError("Colonnes 'Plot' et 'Genre' requises dans le dataset")

    # Appliquer le nettoyage
    df_clean['Plot'] = df_clean['Plot'].apply(clean_text)

    # 3. Suppression des résumés vides
    print(f"   Avant suppression plots vides: {len(df_clean):,}")
    df_clean = df_clean[df_clean['Plot'].str.len() > 0]
    print(f"   Après suppression plots vides: {len(df_clean):,}")

    # 4. Reset de l'index
    df_clean = df_clean.reset_index(drop=True)

    # Statistiques finales
    print(f"\n3. STATISTIQUES FINALES")
    print(f"   Dataset nettoyé: {len(df_clean):,} échantillons")
    print(f"   Distribution finale des genres:")
    for genre in top_genres:
        count = len(df_clean[df_clean['Genre'] == genre])
        percentage = (count / len(df_clean)) * 100
        print(f"   - {genre}: {count:,} ({percentage:.2f}%)")

    # Visualisation de la distribution filtrée
    plt.figure(figsize=(10, 6))
    genre_counts = df_clean['Genre'].value_counts()
    colors = sns.color_palette("Set2", len(genre_counts))
    genre_counts.plot(kind='bar', color=colors)
    plt.title('Distribution des 5 Genres Sélectionnés (Dataset Nettoyé)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Nombre de Films', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = Path("outputs/genre_distribution_filtered.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n   ✓ Graphique sauvegardé: {output_path}")

    # Exemple de nettoyage
    print(f"\n4. EXEMPLE DE NETTOYAGE:")
    if len(df_filtered) > 0:
        idx = 0
        original = df_filtered['Plot'].iloc[idx]
        cleaned = df_clean['Plot'].iloc[idx]
        print(f"   Original: {original[:150]}...")
        print(f"   Nettoyé:  {cleaned[:150]}...")

    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Sauvegarde le dataset nettoyé

    Args:
        df: DataFrame nettoyé
        output_path: Chemin de sauvegarde (data/processed/cleaned_data.csv)
    """
    print("\n" + "="*60)
    print("SAUVEGARDE DES DONNÉES NETTOYÉES")
    print("="*60)

    # Créer le dossier si nécessaire
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder
    df.to_csv(output_file, index=False)

    print(f"\n✓ Dataset nettoyé sauvegardé: {output_file.resolve()}")
    print(f"  Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  Colonnes: {list(df.columns)}")

    # Informations sur le fichier
    file_size = output_file.stat().st_size / (1024 * 1024)  # Taille en MB
    print(f"  Taille du fichier: {file_size:.2f} MB")


# Point d'entrée pour tester ce module indépendamment
if __name__ == "__main__":
    print("="*60)
    print("TEST DU MODULE PREPROCESSING")
    print("="*60)

    # Test du pipeline complet
    dataset_path = download_dataset()
    df = load_and_explore_data(dataset_path)
    top_genres = get_top_n_genres(df, n=5)
    df_clean = preprocess_dataset(df, top_genres)
    save_processed_data(df_clean, 'data/processed/cleaned_data.csv')

    print("\n" + "="*60)
    print("✓ MODULE PREPROCESSING TESTÉ AVEC SUCCÈS")
    print("="*60)
