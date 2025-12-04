#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Test d'Installation - Projet NLP
Auteur: LUKAU
Professeur: Rakia JAZIRI

Ce script vérifie que toutes les dépendances sont correctement installées
et que l'environnement est prêt pour exécuter le projet.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path

def print_header(title):
    """Affiche un en-tête formaté"""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60)

def test_python_version():
    """Vérifie la version de Python"""
    print("\n1. VERSION DE PYTHON")
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("   ✓ Version Python compatible (≥3.8)")
        return True
    else:
        print("   ✗ Version Python incompatible (besoin ≥3.8)")
        return False

def test_imports():
    """Teste l'import de toutes les dépendances"""
    print("\n2. TEST DES DÉPENDANCES")

    packages = {
        'pandas': 'Manipulation de données',
        'numpy': 'Calculs numériques',
        'sklearn': 'Machine Learning (scikit-learn)',
        'nltk': 'Natural Language Processing',
        'matplotlib': 'Visualisations',
        'seaborn': 'Visualisations statistiques',
        'kagglehub': 'Dataset Kaggle',
        'joblib': 'Sauvegarde modèles',
        'scipy': 'Matrices sparse'
    }

    failed = []

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"   ✓ {package:15s} - {description}")
        except ImportError:
            print(f"   ✗ {package:15s} - MANQUANT")
            failed.append(package)

    return len(failed) == 0, failed

def test_nltk_data():
    """Vérifie les données NLTK (stopwords)"""
    print("\n3. TEST DONNÉES NLTK")

    try:
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        print(f"   ✓ Stopwords anglais disponibles ({len(stop_words)} mots)")
        return True
    except LookupError:
        print("   ✗ Stopwords non téléchargés")
        print("   → Exécutez: python -c \"import nltk; nltk.download('stopwords')\"")
        return False

def test_project_structure():
    """Vérifie la structure du projet"""
    print("\n4. STRUCTURE DU PROJET")

    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/preprocessing.py',
        'src/features.py',
        'src/outlier_detection.py',
        'src/visualization.py',
        'src/models.py',
        'src/evaluation.py'
    ]

    required_dirs = [
        'src',
        'data/raw',
        'data/processed',
        'outputs',
        'models'
    ]

    all_ok = True

    # Vérifier les fichiers
    for file in required_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - MANQUANT")
            all_ok = False

    # Vérifier les dossiers
    print("\n   Dossiers:")
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"   ✓ {directory}/")
        else:
            print(f"   ✗ {directory}/ - MANQUANT")
            all_ok = False

    return all_ok

def test_sklearn_components():
    """Teste les composants scikit-learn nécessaires"""
    print("\n5. COMPOSANTS SCIKIT-LEARN")

    components = {
        'TfidfVectorizer': 'sklearn.feature_extraction.text',
        'train_test_split': 'sklearn.model_selection',
        'MultinomialNB': 'sklearn.naive_bayes',
        'LinearSVC': 'sklearn.svm',
        'RandomForestClassifier': 'sklearn.ensemble',
        'IsolationForest': 'sklearn.ensemble',
        'PCA': 'sklearn.decomposition',
        'accuracy_score': 'sklearn.metrics',
        'precision_score': 'sklearn.metrics',
        'recall_score': 'sklearn.metrics',
        'f1_score': 'sklearn.metrics',
        'confusion_matrix': 'sklearn.metrics'
    }

    failed = []

    for component, module in components.items():
        try:
            mod = __import__(module, fromlist=[component])
            getattr(mod, component)
            print(f"   ✓ {component}")
        except (ImportError, AttributeError):
            print(f"   ✗ {component} - MANQUANT")
            failed.append(component)

    return len(failed) == 0, failed

def main():
    """Fonction principale de test"""
    print_header("TEST D'INSTALLATION - PROJET NLP")
    print("Auteur: LUKAU")
    print("Professeur: Rakia JAZIRI")
    print("Master 1 Big Data")

    all_tests_passed = True

    # Test 1: Version Python
    if not test_python_version():
        all_tests_passed = False

    # Test 2: Imports
    imports_ok, failed_imports = test_imports()
    if not imports_ok:
        all_tests_passed = False

    # Test 3: Données NLTK
    if not test_nltk_data():
        all_tests_passed = False

    # Test 4: Structure projet
    if not test_project_structure():
        all_tests_passed = False

    # Test 5: Composants scikit-learn
    sklearn_ok, failed_sklearn = test_sklearn_components()
    if not sklearn_ok:
        all_tests_passed = False

    # Résumé final
    print_header("RÉSUMÉ")

    if all_tests_passed:
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n   Votre environnement est prêt.")
        print("   Vous pouvez exécuter le projet avec:")
        print("\n   → python main.py")
        print("\n" + "="*60)
        return 0
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("\n   Actions recommandées:")

        if failed_imports:
            print(f"\n   1. Installer les packages manquants:")
            print(f"      pip install {' '.join(failed_imports)}")

        if not test_nltk_data():
            print(f"\n   2. Télécharger les données NLTK:")
            print(f"      python -c \"import nltk; nltk.download('stopwords')\"")

        print("\n   Puis relancez ce script:")
        print("   → python test_installation.py")
        print("\n" + "="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
