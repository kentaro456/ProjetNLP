"""
Script de téléchargement du dataset Kaggle Wikipedia Movie Plots
Auteur: Étudiant M1 Big Data - Cours Mme Rakia JAZIRI
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import shutil
import os

def download_dataset():
    """
    Télécharge le dataset depuis Kaggle et le place dans data/raw/
    """
    print("📥 Téléchargement du dataset depuis Kaggle...")

    # Télécharger le dataset
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jrobischon/wikipedia-movie-plots",
        "",  # Fichier vide = charge tous les fichiers
    )

    print(f"✅ Dataset chargé avec succès!")
    print(f"Shape: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    print(f"\nPremières lignes:\n{df.head()}")

    # Sauvegarder dans data/raw/
    output_path = "data/raw/dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset sauvegardé dans: {output_path}")

    return df

if __name__ == "__main__":
    download_dataset()
