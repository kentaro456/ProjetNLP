# 🚀 Guide de Démarrage Rapide

**Projet**: Classification de Genres de Films (NLP)
**Auteur**: LUKAU
**Professeur**: Rakia JAZIRI

---

## ⚡ Démarrage en 5 Minutes

### Étape 1: Installation des Dépendances (2 min)

```bash
# Installer les packages Python
pip install -r requirements.txt

# Télécharger les stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

### Étape 2: Test de l'Installation (30 sec)

```bash
python test_installation.py
```

✅ Si tous les tests passent, continuez à l'étape 3.
❌ Si des tests échouent, suivez les instructions affichées.

### Étape 3: Exécution du Projet (5-10 min)

```bash
python main.py
```

**C'est tout!** Le pipeline va:
1. Télécharger le dataset depuis Kaggle
2. Nettoyer et filtrer les données
3. Extraire les features TF-IDF
4. Détecter et supprimer les outliers
5. Visualiser avec PCA
6. Entraîner 3 modèles
7. Évaluer avec les 5 métriques

---

## 📁 Où Trouver les Résultats?

Après exécution, consultez le dossier `outputs/`:

### Graphiques
- `genre_distribution_filtered.png` - Distribution des 5 genres
- `outlier_detection.png` - Visualisation des outliers
- `pca_scatter_train.png` - PCA 2D du train set
- `pca_scatter_test.png` - PCA 2D du test set
- `metrics_comparison.png` - **Comparaison des 3 modèles**
- `confusion_matrix_best_model.png` - **Matrice de confusion**

### Rapports
- `metrics_results.csv` - **Tableau des 4 métriques**
- `classification_report_best_model.txt` - Détails par classe
- `outlier_report.txt` - Statistiques outliers
- `pipeline_*.log` - Log d'exécution

---

## 🎯 Les 5 Métriques (IMPORTANT pour l'évaluation)

Le fichier `outputs/metrics_results.csv` contient:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX |
| SVM | 0.XX | 0.XX | 0.XX | 0.XX |
| Naive Bayes | 0.XX | 0.XX | 0.XX | 0.XX |

**5ème métrique**: Matrice de Confusion dans `confusion_matrix_best_model.png`

---

## 📊 Validation des Concepts du Cours

Après exécution, vérifiez que vous avez:

### ✅ Isolation Forest (Slide 155)
- `outputs/outlier_detection.png` - Visualisation
- `outputs/outlier_report.txt` - Statistiques

### ✅ PCA (Slide 129)
- `outputs/pca_scatter_train.png` - Train set
- `outputs/pca_scatter_test.png` - Test set
- `outputs/pca_variance_explained.png` - Variance

### ✅ Random Forest (Méthodes Ensemble)
- `models/random_forest.pkl` - Modèle sauvegardé
- Dans `metrics_results.csv` - Performance

### ✅ Les 5 Métriques
- `metrics_results.csv` - 4 métriques numériques
- `confusion_matrix_best_model.png` - 5ème métrique

---

## 🔧 Personnalisation (Optionnel)

### Changer le nombre de genres
Éditez `main.py`, ligne ~97:
```python
top_genres = preprocessing.get_top_n_genres(df, n=5)  # Changez 5 par 3 ou 7
```

### Changer le taux d'outliers
Éditez `main.py`, ligne ~125:
```python
outlier_mask = outlier_detection.detect_outliers_isolation_forest(
    X_train_tfidf, contamination=0.1  # Changez 0.1 par 0.05 ou 0.15
)
```

### Changer le nombre de features TF-IDF
Éditez `main.py`, ligne ~113:
```python
X_train_tfidf, X_test_tfidf, vectorizer = features.create_tfidf_features(
    X_train, X_test, max_features=5000  # Changez 5000 par 3000 ou 10000
)
```

---

## 🐛 Problèmes Fréquents

### "ModuleNotFoundError: No module named 'kagglehub'"
```bash
pip install kagglehub
```

### "LookupError: NLTK stopwords not found"
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### "MemoryError" pendant l'exécution
Réduisez `max_features` à 3000 dans `main.py`

### Téléchargement Kaggle lent
C'est normal, le dataset fait ~15 MB. Patience!

---

## 📚 Structure des Fichiers Source

Pour comprendre le code:

1. **`src/preprocessing.py`** - Téléchargement et nettoyage
2. **`src/features.py`** - TF-IDF et train/test split
3. **`src/outlier_detection.py`** - Isolation Forest ⭐
4. **`src/visualization.py`** - PCA ⭐
5. **`src/models.py`** - 3 classifiers (dont Random Forest ⭐)
6. **`src/evaluation.py`** - Les 5 métriques ⭐
7. **`main.py`** - Orchestrateur

⭐ = Concepts du cours validés

---

## 📝 Pour la Présentation

Points clés à mentionner:

### 1. Rigueur Académique
- Train/Test split **AVANT** tout processing
- Outliers détectés sur **train uniquement**
- Pas de data leakage

### 2. Les 3 Concepts Validés
- **Isolation Forest** (Slide 155): 10% outliers supprimés
- **PCA** (Slide 129): Visualisation 2D de 5000 features
- **Random Forest**: Méthode ensemble performante

### 3. Les 5 Métriques
- **Accuracy**: Score global
- **Precision** (weighted): Gère déséquilibre
- **Recall** (weighted): Capacité détection
- **F1-Score** (weighted): Équilibre P/R
- **Matrice de Confusion**: Détails par classe

### 4. Résultats
- Comparez les 3 modèles dans `metrics_comparison.png`
- Analysez la matrice de confusion
- Identifiez le meilleur modèle (F1-Score)

---

## ✅ Checklist Avant Soumission

- [ ] `python test_installation.py` réussit
- [ ] `python main.py` s'exécute sans erreur
- [ ] Dossier `outputs/` contient 13+ fichiers
- [ ] Dossier `models/` contient 3 fichiers .pkl
- [ ] `metrics_results.csv` a 3 lignes (3 modèles)
- [ ] `confusion_matrix_best_model.png` est lisible
- [ ] Le meilleur modèle est identifié

---

## 🎓 Critères d'Évaluation (Auto-Vérification)

### Structure Professionnelle ✅
- [x] Code modulaire (7 fichiers .py séparés)
- [x] Pas de Jupyter Notebook
- [x] Architecture claire (src/, data/, outputs/, models/)

### Concepts du Cours ✅
- [x] Isolation Forest (Slide 155) - Code dans `outlier_detection.py`
- [x] PCA (Slide 129) - Code dans `visualization.py`
- [x] Random Forest - Code dans `models.py`

### Les 5 Métriques ✅
- [x] Accuracy - Calculée dans `evaluation.py`
- [x] Precision (weighted) - Calculée dans `evaluation.py`
- [x] Recall (weighted) - Calculée dans `evaluation.py`
- [x] F1-Score (weighted) - Calculée dans `evaluation.py`
- [x] Matrice de Confusion - Générée dans `evaluation.py`

### Rigueur Académique ✅
- [x] Train/Test split stratifié
- [x] Prévention data leakage
- [x] Reproductibilité (random_state=42)
- [x] Gestion déséquilibre (class_weight='balanced')

### Documentation ✅
- [x] Code commenté en français
- [x] Références aux slides du cours
- [x] README.md complet
- [x] Logs d'exécution

---

## 🏆 Objectif Final

**Démontrer la maîtrise des 3 concepts clés:**

1. ✅ Détection d'Anomalies (Isolation Forest)
2. ✅ Réduction de Dimension (PCA)
3. ✅ Méthodes Ensemblistes (Random Forest)

**Avec une évaluation rigoureuse:**

- ✅ Les 5 métriques calculées et comparées
- ✅ Approche académique sans data leakage
- ✅ Résultats reproductibles

---

**Bonne chance! 🚀**

Pour toute question, consultez:
- `README.md` - Documentation complète
- `outputs/pipeline_*.log` - Détails d'exécution
- Le code source dans `src/` - Commentaires détaillés
