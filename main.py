#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJET NLP: Classification de Genres de Films

Pipeline principal du projet:
1. Téléchargement et preprocessing du dataset
2. Extraction de features (TF-IDF)
3. Détection d'outliers (Isolation Forest)
4. Réduction de dimension (PCA)
5. Entraînement de modèles (NB, SVM, RF)
6. Évaluation complète
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Ajouter le dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Imports des modules du projet
from src import preprocessing, features, outlier_detection, visualization, models, evaluation


def setup_logging():
    """Configure le système de logging pour le pipeline"""
    # Créer le dossier outputs si nécessaire
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # Configuration du logging
    log_filename = f"outputs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def create_directories():
    """Crée l'arborescence complète du projet"""
    dirs = [
        'data/raw',
        'data/processed',
        'outputs',
        'models',
        'src'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✓ Arborescence du projet créée")


def print_header(title: str, char: str = "="):
    """Affiche un en-tête formaté"""
    width = 70
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def main():
    """
    Pipeline principal du projet NLP

    Exécute séquentiellement toutes les étapes du projet
    avec gestion des erreurs et logging détaillé.
    """
    # Démarrer le chronomètre
    start_time = time.time()

    # Configuration du logging
    logger = setup_logging()

    # En-tête du projet
    print_header("PROJET NLP - CLASSIFICATION DE GENRES DE FILMS", "=")
    print("Auteur: LUKAU")
    print("Professeur: Rakia JAZIRI")
    print("Master 1 Big Data")
    print(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_header("", "=")

    try:
        # ================================================================
        # ÉTAPE 0: SETUP
        # ================================================================
        print_header("ÉTAPE 0/7: INITIALISATION", "-")
        create_directories()
        logger.info("Projet initialisé avec succès")

        # ================================================================
        # ÉTAPE 1: TÉLÉCHARGEMENT ET PREPROCESSING
        # ================================================================
        print_header("ÉTAPE 1/7: TÉLÉCHARGEMENT ET PREPROCESSING", "-")
        logger.info("Début du téléchargement du dataset...")

        # Télécharger le dataset
        dataset_path = preprocessing.download_dataset()
        logger.info(f"Dataset téléchargé: {dataset_path}")

        # Explorer le dataset
        df = preprocessing.load_and_explore_data(dataset_path)
        logger.info(f"Dataset chargé: {len(df):,} échantillons")

        # Identifier les top 5 genres
        top_genres = preprocessing.get_top_n_genres(df, n=5)
        logger.info(f"Top 5 genres sélectionnés: {top_genres}")

        # Nettoyer et filtrer
        df_clean = preprocessing.preprocess_dataset(df, top_genres)
        logger.info(f"Dataset nettoyé: {len(df_clean):,} échantillons")

        # Sauvegarder
        preprocessing.save_processed_data(df_clean, 'data/processed/cleaned_data.csv')
        logger.info("Dataset nettoyé sauvegardé")

        # ================================================================
        # ÉTAPE 2: EXTRACTION DE FEATURES
        # ================================================================
        print_header("ÉTAPE 2/7: EXTRACTION DE FEATURES (TF-IDF)", "-")
        logger.info("Début de l'extraction de features...")

        # Séparer X et y
        X = df_clean['Plot']
        y = df_clean['Genre']

        # Train/Test Split stratifié (80/20)
        X_train, X_test, y_train, y_test = features.split_data(X, y, test_size=0.2, random_state=42)
        logger.info(f"Split: {len(X_train):,} train / {len(X_test):,} test")

        # Vectorisation TF-IDF (fit sur train uniquement)
        X_train_tfidf, X_test_tfidf, vectorizer = features.create_tfidf_features(
            X_train, X_test, max_features=5000, ngram_range=(1, 2)
        )
        logger.info(f"TF-IDF: {X_train_tfidf.shape[1]:,} features")

        # ================================================================
        # ÉTAPE 3: DÉTECTION D'OUTLIERS (TRAIN ONLY)
        # ================================================================
        print_header("ÉTAPE 3/7: DÉTECTION D'OUTLIERS (Isolation Forest - Slide 155)", "-")
        logger.info("Début de la détection d'outliers (TRAIN SET uniquement)...")

        # Détecter les outliers avec Isolation Forest
        outlier_mask = outlier_detection.detect_outliers_isolation_forest(
            X_train_tfidf, contamination=0.1, random_state=42
        )

        # Supprimer les outliers du train set
        X_train_clean, y_train_clean = outlier_detection.remove_outliers(
            X_train_tfidf, y_train, outlier_mask
        )
        logger.info(f"Outliers supprimés: {len(y_train) - len(y_train_clean):,}")

        # Visualiser les outliers
        outlier_detection.visualize_outliers(
            X_train_tfidf, outlier_mask, 'outputs/outlier_detection.png'
        )

        # Rapport sur les outliers
        outlier_detection.save_outlier_report(
            outlier_mask, y_train, 'outputs/outlier_report.txt'
        )
        logger.info("Détection d'outliers terminée")

        # ================================================================
        # ÉTAPE 4: VISUALISATION (PCA)
        # ================================================================
        print_header("ÉTAPE 4/7: VISUALISATION (PCA 2D - Slide 129)", "-")
        logger.info("Début de la réduction dimensionnelle avec PCA...")

        # Convertir sparse matrices en dense pour PCA
        X_train_dense = X_train_clean.toarray()
        X_test_dense = X_test_tfidf.toarray()

        # Appliquer PCA (2 composantes)
        X_train_pca, X_test_pca, pca_model, var_ratio = visualization.apply_pca(
            X_train_dense, X_test_dense, n_components=2, random_state=42
        )
        logger.info(f"PCA: Variance expliquée = {sum(var_ratio)*100:.2f}%")

        # Créer toutes les visualisations
        visualization.create_all_visualizations(
            X_train_pca, X_test_pca, y_train_clean, y_test, pca_model, var_ratio
        )
        logger.info("Visualisations PCA générées")

        # ================================================================
        # ÉTAPE 5: ENTRAÎNEMENT DES MODÈLES
        # ================================================================
        print_header("ÉTAPE 5/7: ENTRAÎNEMENT DES MODÈLES (3 Classifiers)", "-")
        logger.info("Début de l'entraînement des modèles...")

        # Entraîner les 3 modèles
        all_models = models.train_all_models(X_train_clean, y_train_clean)
        logger.info(f"Modèles entraînés: {list(all_models.keys())}")

        # ================================================================
        # ÉTAPE 6: ÉVALUATION (LES 5 MÉTRIQUES)
        # ================================================================
        print_header("ÉTAPE 6/7: ÉVALUATION (LES 5 MÉTRIQUES)", "-")
        logger.info("Début de l'évaluation sur le test set...")

        # Évaluation complète avec les 5 métriques
        results_df = evaluation.evaluate_and_save_all(
            all_models, X_test_tfidf, y_test, top_genres
        )
        logger.info("Évaluation complète terminée")

        # ================================================================
        # ÉTAPE 7: RÉSUMÉ FINAL
        # ================================================================
        print_header("ÉTAPE 7/7: RÉSUMÉ FINAL", "-")

        # Temps d'exécution
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print(f"\n✓ PIPELINE TERMINÉ AVEC SUCCÈS")
        print(f"\nTemps d'exécution: {minutes} min {seconds} sec")

        print("\n" + "="*70)
        print("RÉSULTATS FINAUX")
        print("="*70)

        # Afficher le tableau des résultats
        print("\nPerformances des Modèles:")
        print(results_df.to_string(index=False))

        # Meilleur modèle
        best_model = results_df.iloc[0]['Model']
        best_f1 = results_df.iloc[0]['F1-Score']
        print(f"\n🏆 Meilleur modèle: {best_model} (F1-Score: {best_f1*100:.2f}%)")

        print("\n" + "="*70)
        print("FICHIERS GÉNÉRÉS")
        print("="*70)

        print("\n📊 DONNÉES:")
        print("  - data/raw/dataset.csv - Dataset original Kaggle")
        print("  - data/processed/cleaned_data.csv - Dataset nettoyé (5 genres)")

        print("\n📈 GRAPHIQUES:")
        print("  - outputs/genre_distribution_original.png - Distribution tous genres")
        print("  - outputs/genre_distribution_filtered.png - Distribution 5 genres")
        print("  - outputs/outlier_detection.png - Visualisation outliers")
        print("  - outputs/pca_scatter_train.png - PCA train set")
        print("  - outputs/pca_scatter_test.png - PCA test set")
        print("  - outputs/pca_variance_explained.png - Variance PCA")
        print("  - outputs/metrics_comparison.png - Comparaison modèles")
        print("  - outputs/confusion_matrix_best_model.png - Matrice confusion")

        print("\n📋 RAPPORTS:")
        print("  - outputs/metrics_results.csv - Tableau comparatif")
        print("  - outputs/tfidf_features.csv - Top features TF-IDF")
        print("  - outputs/outlier_report.txt - Rapport outliers")
        print("  - outputs/classification_report_best_model.txt - Rapport détaillé")
        print(f"  - {Path('outputs').glob('pipeline_*.log').__next__()} - Log pipeline")

        print("\n🤖 MODÈLES:")
        print("  - models/naive_bayes.pkl - Modèle Naive Bayes")
        print("  - models/svm.pkl - Modèle SVM")
        print("  - models/random_forest.pkl - Modèle Random Forest")

        print("\n" + "="*70)
        print("VALIDATION PÉDAGOGIQUE")
        print("="*70)

        print("\n✅ CONCEPTS DU COURS VALIDÉS:")
        print("  1. ✓ Détection d'Anomalies: Isolation Forest (Slide 155)")
        print("  2. ✓ Réduction de Dimension: PCA (Slide 129)")
        print("  3. ✓ Méthodes Ensemblistes: Random Forest")
        print("  4. ✓ Les 5 Métriques: Accuracy, Precision, Recall, F1, Confusion Matrix")

        print("\n✅ RIGUEUR ACADÉMIQUE:")
        print("  - Train/Test split stratifié (80/20)")
        print("  - TF-IDF fitté sur train uniquement")
        print("  - Outliers détectés sur train uniquement")
        print("  - PCA fitté sur train uniquement")
        print("  - Évaluation sur test set uniquement")
        print("  - Reproductibilité: random_state=42")

        print("\n" + "="*70)
        logger.info("Pipeline terminé avec succès")

    except Exception as e:
        logger.error(f"ERREUR CRITIQUE: {str(e)}", exc_info=True)
        print(f"\n❌ ERREUR: {str(e)}")
        print("Consultez le fichier log pour plus de détails")
        sys.exit(1)


if __name__ == "__main__":
    main()
