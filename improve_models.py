#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'amélioration des modèles

Ce script implémente une stratégie d'Undersampling (réduction des classes majoritaires)
pour gérer le déséquilibre du dataset et améliorer la détection des classes minoritaires (ex: Action).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.utils import resample
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("AMÉLIORATION DES MODÈLES - UNDERSAMPLING (MÉTHODE DU TP)")
print("="*80)
print("\n📚 Référence: TP 'Détection de Fraudes' - Mme Rakia JAZIRI")
print("   Solution enseignée: Sous-échantillonner la classe majoritaire")
print("="*80)

start_time = time.time()

# ============================================================================
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
print("="*80)

df = pd.read_csv('data/processed/cleaned_data.csv')
print(f"\nDataset chargé: {len(df):,} échantillons")

X = df['Plot']
y = df['Genre']

print(f"\n📊 Distribution ORIGINALE:")
dist_original = y.value_counts().sort_index()
for genre, count in dist_original.items():
    pct = (count / len(y)) * 100
    print(f"   {genre:8s}: {count:5,} ({pct:5.2f}%)")

print(f"\n⚠️  PROBLÈME IDENTIFIÉ:")
print(f"   - Classe minoritaire 'action': {(y == 'action').sum():,} (5.87%)")
print(f"   - Classe majoritaire 'unknown': {(y == 'unknown').sum():,} (32.55%)")
print(f"   - Ratio déséquilibre: 1 action pour 5.5 unknown")

# ============================================================================
# ÉTAPE 2: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 2: TRAIN/TEST SPLIT STRATIFIÉ")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train):,} échantillons")
print(f"Test set:  {len(X_test):,} échantillons")

# ============================================================================
# ÉTAPE 3: UNDERSAMPLING (MÉTHODE DU TP)
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 3: UNDERSAMPLING - MÉTHODE DU TP FRAUDES")
print("="*80)

print("\n📚 Citation du TP:")
print('   "Pour rééquilibrer les données, nous allons garder aléatoirement')
print('    5000 transactions de cartes normales (classe 0) et toutes les')
print('    transactions aberrantes (classe 1)."')

print("\n🎯 Application à notre projet:")
print("   - Garder TOUS les films 'action' (classe minoritaire)")
print("   - Réduire les autres classes à 1,500 échantillons chacune")

# Créer un DataFrame temporaire pour le train set
df_train = pd.DataFrame({'Plot': X_train, 'Genre': y_train})

# Séparer par classe
df_action = df_train[df_train['Genre'] == 'action']
df_comedy = df_train[df_train['Genre'] == 'comedy']
df_drama = df_train[df_train['Genre'] == 'drama']
df_horror = df_train[df_train['Genre'] == 'horror']
df_unknown = df_train[df_train['Genre'] == 'unknown']

print(f"\n📊 Distribution AVANT undersampling:")
print(f"   action:  {len(df_action):,}")
print(f"   comedy:  {len(df_comedy):,}")
print(f"   drama:   {len(df_drama):,}")
print(f"   horror:  {len(df_horror):,}")
print(f"   unknown: {len(df_unknown):,}")
print(f"   TOTAL:   {len(df_train):,}")

# Sous-échantillonner à 1,500 (sauf action qu'on garde tous)
TARGET_SIZE = 1500

df_comedy_sampled = resample(
    df_comedy,
    n_samples=TARGET_SIZE,
    random_state=42,
    replace=False
)
df_drama_sampled = resample(
    df_drama,
    n_samples=TARGET_SIZE,
    random_state=42,
    replace=False
)
df_horror_sampled = resample(
    df_horror,
    n_samples=min(TARGET_SIZE, len(df_horror)),  # Horror < 1500
    random_state=42,
    replace=(len(df_horror) < TARGET_SIZE)  # Oversampling si nécessaire
)
df_unknown_sampled = resample(
    df_unknown,
    n_samples=TARGET_SIZE,
    random_state=42,
    replace=False
)

# Recombiner
df_balanced = pd.concat([
    df_action,           # Tous gardés
    df_comedy_sampled,
    df_drama_sampled,
    df_horror_sampled,
    df_unknown_sampled
], axis=0)

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = df_balanced['Plot']
y_train_balanced = df_balanced['Genre']

print(f"\n📊 Distribution APRÈS undersampling:")
dist_balanced = y_train_balanced.value_counts().sort_index()
for genre, count in dist_balanced.items():
    pct = (count / len(y_train_balanced)) * 100
    print(f"   {genre:8s}: {count:5,} ({pct:5.2f}%)")
print(f"   TOTAL:   {len(df_balanced):,}")

print(f"\n✅ Dataset rééquilibré!")
print(f"   Réduction: {len(df_train):,} → {len(df_balanced):,} échantillons")
print(f"   Gain: Plus équilibré pour la classe 'action'")

# ============================================================================
# ÉTAPE 4: VECTORISATION TF-IDF
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 4: VECTORISATION TF-IDF")
print("="*80)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Bigrams
    min_df=2,
    max_df=0.8,
    sublinear_tf=True,
    strip_accents='unicode',
    lowercase=True
)

X_train_balanced_tfidf = vectorizer.fit_transform(X_train_balanced)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\n✓ Matrices TF-IDF créées:")
print(f"   Train (balanced): {X_train_balanced_tfidf.shape}")
print(f"   Test (inchangé):  {X_test_tfidf.shape}")

# Sauvegarder le vectorizer
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print(f"   ✓ Vectorizer sauvegardé: models/tfidf_vectorizer.pkl")

# ============================================================================
# ÉTAPE 5: ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 5: ENTRAÎNEMENT DES MODÈLES (avec dataset équilibré)")
print("="*80)

models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),
    'SVM': LinearSVC(
        C=1.0,
        class_weight='balanced',  # Principe du Boosting (Slide 84)
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,  # Éviter overfitting
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

results_improved = []

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"🚀 {model_name} (IMPROVED)")
    print('='*80)

    # Entraînement
    print(f"   Entraînement en cours...")
    model.fit(X_train_balanced_tfidf, y_train_balanced)

    # Prédiction sur le test set
    y_pred = model.predict(X_test_tfidf)

    # Métriques globales
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n   ✓ Accuracy: {accuracy*100:.2f}%")
    print(f"   ✓ F1-Score (weighted): {f1_weighted*100:.2f}%")

    # Métriques PAR CLASSE (Slide 154 - Détection Anomalies)
    print(f"\n   📊 Rapport PAR CLASSE (Slide 154 - Métriques adaptées):")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=sorted(y_test.unique()), zero_division=0
    )

    for i, genre in enumerate(sorted(y_test.unique())):
        emoji = {'action': '💥', 'comedy': '😂', 'drama': '🎭',
                 'horror': '👻', 'unknown': '❓'}.get(genre, '🎬')
        print(f"      {emoji} {genre:8s}: P={precision[i]*100:5.1f}% | "
              f"R={recall[i]*100:5.1f}% | F1={f1[i]*100:5.1f}% (n={support[i]})")

    # Focus sur ACTION
    action_idx = sorted(y_test.unique()).index('action')
    action_f1 = f1[action_idx]

    print(f"\n   ⭐ F1-Score ACTION: {action_f1*100:.2f}%")

    # Sauvegarder
    model_filename = f'models/{model_name.lower().replace(" ", "_")}_undersampled.pkl'
    joblib.dump(model, model_filename)
    print(f"   ✓ Modèle sauvegardé: {model_filename}")

    results_improved.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Weighted': f1_weighted,
        'F1-Action': action_f1
    })

# ============================================================================
# ÉTAPE 6: COMPARAISON AVANT/APRÈS
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 6: COMPARAISON AVANT (original) / APRÈS (undersampling)")
print("="*80)

# Résultats AVANT (modèles originaux)
results_before = pd.DataFrame([
    {'Model': 'Naive Bayes', 'Accuracy': 0.6357, 'F1-Weighted': 0.6229, 'F1-Action': 0.3862},
    {'Model': 'SVM', 'Accuracy': 0.6138, 'F1-Weighted': 0.6141, 'F1-Action': 0.35},
    {'Model': 'Random Forest', 'Accuracy': 0.5635, 'F1-Weighted': 0.5570, 'F1-Action': 0.32}
])

results_after = pd.DataFrame(results_improved)

print("\n📊 TABLEAU COMPARATIF:")
print("\n" + "-"*100)
print(f"{'Modèle':<18} {'Accuracy AVANT':<15} {'Accuracy APRÈS':<15} {'Δ Acc':<10} "
      f"{'F1-Action AVANT':<17} {'F1-Action APRÈS':<17} {'Δ F1-Action'}")
print("-"*100)

for i in range(len(results_before)):
    before = results_before.iloc[i]
    after = results_after.iloc[i]

    acc_before = before['Accuracy'] * 100
    acc_after = after['Accuracy'] * 100
    acc_delta = acc_after - acc_before

    f1_before = before['F1-Action'] * 100
    f1_after = after['F1-Action'] * 100
    f1_delta = f1_after - f1_before

    indicator = '✅' if f1_delta > 5 else ('⚠️' if f1_delta > 0 else '❌')

    print(f"{before['Model']:<18} {acc_before:6.2f}%          {acc_after:6.2f}%          "
          f"{acc_delta:+6.2f}%    {f1_before:6.2f}%            {f1_after:6.2f}%            "
          f"{f1_delta:+6.2f}%  {indicator}")

print("-"*100)

# Générer matrice de confusion pour le meilleur modèle
best_idx = results_after['F1-Action'].argmax()
best_model_name = results_after.iloc[best_idx]['Model']

print(f"\n🏆 Meilleur modèle pour ACTION: {best_model_name}")
print(f"   F1-Score ACTION: {results_after.iloc[best_idx]['F1-Action']*100:.2f}%")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("✅ AMÉLIORATION TERMINÉE AVEC SUCCÈS!")
print("="*80)

print(f"\n⏱️  Temps d'exécution: {elapsed_time:.1f} secondes")

print("\n📁 Fichiers générés:")
print("   ✓ models/naive_bayes_undersampled.pkl")
print("   ✓ models/svm_undersampled.pkl")
print("   ✓ models/random_forest_undersampled.pkl")

print("\n🎯 RÉSULTATS:")
avg_gain = results_after['F1-Action'].mean() - results_before['F1-Action'].mean()
print(f"   - Gain moyen F1-Score ACTION: {avg_gain*100:+.2f}%")
print(f"   - Méthode: Undersampling (TP Détection de Fraudes)")
print(f"   - Dataset réduit: {len(df_train):,} → {len(df_balanced):,} échantillons")

print("\n💡 JUSTIFICATION PÉDAGOGIQUE:")
print("   Cette amélioration se base sur:")
print("   1. TP 'Détection de Fraudes' (Undersampling de la classe majoritaire)")
print("   2. Slide 84 - Boosting (class_weight='balanced')")
print("   3. Slide 154 - Métriques adaptées (Precision/Recall par classe)")

print("\n📊 POUR ALLER PLUS LOIN:")
print("   - Tester SMOTE (oversampling) comme extension")
print("   - Optimiser les hyperparamètres (Grid Search)")
print("   - Analyser la courbe Precision-Recall pour Action")

print("\n" + "="*80)
