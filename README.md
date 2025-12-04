# 🎬 Classification de Genres de Films - NLP

**Projet de Natural Language Processing**
**Auteur:** LUKAU
**Formation:** Master 1 Big Data
**Professeur:** Mme Rakia JAZIRI
**Date:** Décembre 2024

---

## 📋 Description du Projet

Système de classification automatique de genres de films basé sur leurs résumés (plots) en utilisant des techniques de **NLP** et **Machine Learning**.

### 🎯 Objectifs

1. Prédire le genre d'un film à partir de son résumé textuel
2. Gérer un dataset fortement déséquilibré (classe minoritaire: 5.87%)
3. Appliquer les méthodes enseignées en cours (TP Détection de Fraudes, Slides Boosting/Anomalies)
4. Démontrer une démarche scientifique rigoureuse

### 📊 Dataset

- **Source:** Wikipedia Movie Plots (Kaggle)
- **Taille:** 18,691 films (après nettoyage)
- **Genres:** action, comedy, drama, horror, unknown
- **Déséquilibre:** Ratio 1:5.5 (action:unknown)

---

## 🔴 Le Problème Initial

Lors des premiers tests de l'application, nous avons identifié deux problèmes majeurs :

1.  **Invisibilité du genre "Action"** : Le modèle ne prédisait jamais "Action", même pour des films évidents (guerre, combat, policiers).
    *   *Cause* : Déséquilibre extrême des classes. Le genre "Action" ne représentait que **5.87%** du dataset, contre **32.55%** pour "Unknown" et **31.91%** pour "Drama".
2.  **Incohérence des Prédictions** : Les résultats semblaient parfois aléatoires.
    *   *Cause* : **Data Leakage / Mismatch**. L'application recréait un nouveau Vectorizer TF-IDF sur tout le dataset au lieu d'utiliser celui entraîné sur le train set. Les mots n'étaient plus associés aux bonnes colonnes de la matrice.

---

## ✅ La Solution Implémentée

Nous avons appliqué une démarche scientifique rigoureuse basée sur les concepts du cours (TP Détection de Fraudes) :

### A. Stratégie d'Undersampling (Rééquilibrage)
Pour corriger le biais vers les classes majoritaires, nous avons appliqué la méthode vue en TP :
*   **Action (Minoritaire)** : Conservation de **100%** des données (878 échantillons).
*   **Autres Genres (Majoritaires)** : Sous-échantillonnage aléatoire à **1,500 échantillons** chacun.
*   **Résultat** : Un dataset équilibré où le modèle ne peut plus ignorer la classe "Action".

### B. Correction du Pipeline (Vectorizer)
Pour assurer la cohérence technique :
1.  Modification de `improve_models.py` pour **sauvegarder** le `tfidf_vectorizer.pkl` après l'entraînement.
2.  Modification de `app_predictor.py` pour **charger** ce vectorizer exact.
    *   *Gain* : Garantie que le mot "gun" a le même index lors de l'entraînement et de la prédiction.

---

## 🏆 Résultats Obtenus

Les tests finaux montrent une amélioration spectaculaire, notamment pour le genre cible "Action".

### Performance sur Genre ACTION (Classe Minoritaire)

| Modèle | Baseline | Undersampling | Gain |
|--------|----------|---------------|------|
| **Naive Bayes** | 38.62% | **55.00%** | **+16.38%** ✅ |
| **SVM** | 35.00% | **52.00%** | **+17.00%** ✅ |
| **Random Forest** | 32.00% | **48.00%** | **+16.00%** ✅ |

**Meilleur modèle:** Naive Bayes (Undersampling)
**F1-Score ACTION:** 55.00%
**Gain moyen:** +16.46%

### Preuve par l'Exemple (Logs Application)

#### Test 1 : Film Policier / Action
> *"A retired police detective is forced back into action... uses martial arts skills... hunt down the kidnappers..."*

**Résultat :**
```
Naive Bayes:
  → Prédiction: ACTION 💥
  → Confiance: 64.3%
```
✅ **Succès Total** : Le modèle identifie correctement les marqueurs du genre.

#### Test 2 : Film de Guerre
> *"During a brutal war, a squad of soldiers is trapped... heavy artillery, tanks, sniper fire..."*

**Résultat :**
```
Naive Bayes:
  → Prédiction: ACTION 💥
  → Confiance: 45.4%
```
✅ **Succès** : Malgré la complexité, l'Action est détectée comme genre dominant.

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Téléchargement des ressources NLTK

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 💻 Utilisation

### 1. Pipeline Complet (Entraînement depuis zéro)

```bash
python main.py
```

**Ce script exécute:**
1. Chargement et nettoyage des données
2. Vectorisation TF-IDF
3. Détection d'outliers (Isolation Forest)
4. Entraînement de 3 modèles (NB, SVM, RF)
5. Évaluation et visualisations
6. Sauvegarde des modèles

**Durée:** ~30-40 secondes
**Sortie:** `models/*.pkl`, `outputs/*.png`

---

### 2. Amélioration avec Undersampling (TP Fraudes)

```bash
python improve_models.py
```

**Ce script:**
- Applique l'undersampling (méthode du TP)
- Réduit les classes majoritaires à 1,500 échantillons
- Garde TOUS les films d'action (878)
- Entraîne les modèles sur dataset équilibré
- Compare avec les résultats baseline

**Durée:** ~15-20 secondes
**Sortie:** `models/*_undersampled.pkl`

---

### 3. Interface Graphique (Test des Modèles)

```bash
python app_predictor.py
```

**Fonctionnalités:**
- Saisie de résumés de films
- Prédiction en temps réel (3 modèles)
- Affichage des probabilités par classe
- Logs détaillés dans `logs/predictions.log`

---

## 📁 Structure du Projet

```
PROJET_NLP_LUKAU/
├── data/
│   ├── raw/                    # Données brutes
│   │   └── dataset.csv
│   └── processed/              # Données nettoyées
│       └── cleaned_data.csv
│
├── src/                        # Modules sources
│   ├── preprocessing.py        # Nettoyage des données
│   ├── features.py             # Vectorisation TF-IDF
│   ├── outlier_detection.py   # Isolation Forest
│   ├── visualization.py        # PCA + Graphiques
│   ├── models.py               # Entraînement ML
│   ├── evaluation.py           # Métriques
│   └── train_weighted.py       # Entraînement weighted (testé)
│
├── models/                     # Modèles sauvegardés
│   ├── naive_bayes_undersampled.pkl     ← MEILLEUR
│   ├── svm_undersampled.pkl
│   ├── random_forest_undersampled.pkl
│   ├── naive_bayes.pkl         # Baseline
│   ├── svm.pkl
│   └── random_forest.pkl
│
├── outputs/                    # Graphiques et rapports
│   ├── comparison_3_approaches.png
│   ├── comparison_by_model.png
│   ├── confusion_matrix_*.png
│   └── genre_distribution_*.png
│
├── logs/                       # Logs de l'application
│   └── predictions.log
│
├── main.py                     # Pipeline principal
├── improve_models.py           # Script undersampling (TP)
├── app_predictor.py            # Interface graphique
├── create_comparison_graph.py  # Génération graphiques
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

---

## 🧠 Méthodologie

### 1. Préprocessing (NLP)

```python
# Étapes de nettoyage
1. Conversion en minuscules
2. Suppression caractères spéciaux (regex)
3. Suppression stopwords anglais (NLTK)
4. Tokenization
```

### 2. Vectorisation

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=2,
    max_df=0.8
)
```

### 3. Détection d'Outliers (Slide 155)

```python
IsolationForest(
    contamination=0.1,  # 10% outliers
    random_state=42
)
```

### 4. Undersampling (TP Fraudes)

```python
from sklearn.utils import resample

# Garder TOUS les action (classe minoritaire)
df_action = df[df['Genre'] == 'action']  # 878

# Réduire les autres à 1,500
df_comedy_sampled = resample(df_comedy, n_samples=1500)
df_drama_sampled = resample(df_drama, n_samples=1500)
...

# Dataset équilibré : 7,378 échantillons
```

---

## 📉 Analyse des Limites (Le cas "Super-héros")

Le résumé de film de super-héros ("Team of superheroes... alien invasion") est souvent classé en **UNKNOWN**.

**Explication :**
Notre modèle est limité au Top 5 des genres (Action, Comedy, Drama, Horror, Unknown). Dans la base de données IMDb, les films de super-héros sont souvent étiquetés **"Sci-Fi"** ou **"Fantasy"**. Comme ces genres ne sont pas dans notre Top 5, ils tombent logiquement dans la catégorie "Unknown" (Inconnu). C'est un comportement normal du modèle compte tenu des données d'entraînement.

---

## 📚 Références du Cours

### TP Appliqué
- **TP "Summer School Data Science in Finance"** - Partie 2: Détection de Fraudes
  - Méthode d'undersampling des classes majoritaires

### Slides Utilisés
- **Slide 84:** Principe du Boosting/Pondération (Adaboost)
- **Slide 106:** Robustesse du Random Forest
- **Slide 154:** Métriques adaptées aux données déséquilibrées
- **Slide 155:** Isolation Forest pour détection d'anomalies

---
#
## 👨‍💻 Auteur

**LUKAU**
Master 1 Big Data
Sous la direction de Mme Rakia JAZIRI