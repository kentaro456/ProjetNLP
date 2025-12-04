#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application de prédiction de genres de films
Interface graphique pour tester les modèles entraînés
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib
import pandas as pd
import re
import logging
from pathlib import Path
from datetime import datetime
from nltk.corpus import stopwords
import nltk

# Télécharger les stopwords si nécessaire
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


class MovieGenrePredictor:
    """
    Application GUI pour prédire le genre d'un film
    """

    def __init__(self, root):
        """
        Initialisation de l'interface

        Args:
            root: Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("🎬 Prédicteur de Genres de Films")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.models = {}
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.genres = ['action', 'comedy', 'drama', 'horror', 'unknown']

        # Configuration du logging
        self.setup_logging()

        # Charger les modèles
        self.load_models()

        # Créer l'interface
        self.create_widgets()

    def setup_logging(self):
        """
        Configure le système de logging pour enregistrer les prédictions
        """
        # Créer le dossier logs s'il n'existe pas
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Nom du fichier log avec timestamp
        log_filename = log_dir / "predictions.log"

        # Configuration du logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # Affiche aussi dans la console
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("APPLICATION DÉMARRÉE - Prédicteur de Genres de Films")
        self.logger.info("="*80)

    def load_models(self):
        """
        Charge les 3 modèles entraînés et le vectorizer TF-IDF
        """
        try:
            # Charger les modèles améliorés (undersampled)
            models_dir = Path("models")
            self.models['Naive Bayes'] = joblib.load(models_dir / "naive_bayes_undersampled.pkl")
            self.models['SVM'] = joblib.load(models_dir / "svm_undersampled.pkl")
            self.models['Random Forest'] = joblib.load(models_dir / "random_forest_undersampled.pkl")

            # Charger le vectorizer TF-IDF
            # Note: Le vectorizer n'est pas sauvegardé dans le pipeline actuel
            # On va le recréer à partir du dataset nettoyé
            self.load_vectorizer()

            print("✓ Modèles chargés avec succès")
            self.logger.info("✓ Modèles améliorés (undersampled) chargés avec succès")
            self.logger.info("  - Naive Bayes: naive_bayes_undersampled.pkl")
            self.logger.info("  - SVM: svm_undersampled.pkl")
            self.logger.info("  - Random Forest: random_forest_undersampled.pkl")

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les modèles:\n{e}")
            self.root.quit()

    def load_vectorizer(self):
        """
        Charge le vectorizer TF-IDF sauvegardé
        """
        try:
            vectorizer_path = Path("models/tfidf_vectorizer.pkl")
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                print("✓ Vectorizer TF-IDF chargé avec succès")
                self.logger.info("✓ Vectorizer TF-IDF chargé avec succès")
                self.logger.info(f"  - Vocabulaire: {len(self.vectorizer.vocabulary_)} termes")
            else:
                # Fallback: Recréer si le fichier n'existe pas (déconseillé mais robuste)
                self.logger.warning("⚠️ Vectorizer sauvegardé introuvable, tentative de recréation...")
                self._recreate_vectorizer()

        except Exception as e:
            messagebox.showerror(
                "Erreur",
                f"Impossible de charger le vectorizer:\n{e}"
            )
            self.root.quit()

    def _recreate_vectorizer(self):
        """
        Méthode de secours: Recrée le vectorizer si le fichier est manquant
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Charger le dataset nettoyé
        data_path = Path("data/processed/cleaned_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            
            # Recréer le vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
                strip_accents='unicode',
                lowercase=True
            )
            
            self.vectorizer.fit(df['Plot'])
            print("⚠️ Vectorizer recréé (Attention: risque d'incohérence avec les modèles)")
            self.logger.warning("⚠️ Vectorizer recréé (Attention: risque d'incohérence avec les modèles)")
        else:
            messagebox.showerror(
                "Erreur",
                "Dataset nettoyé introuvable.\nVeuillez d'abord exécuter main.py"
            )
            self.root.quit()

    def clean_text(self, text):
        """
        Nettoie un texte selon le même preprocessing que l'entraînement

        Args:
            text (str): Texte brut

        Returns:
            str: Texte nettoyé
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        # Conversion en minuscules
        text = text.lower()

        # Suppression des caractères spéciaux
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)

        # Suppression des stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]

        return ' '.join(words).strip()

    def create_widgets(self):
        """
        Crée tous les widgets de l'interface
        """
        # En-tête
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🎬 PRÉDICTEUR DE GENRES DE FILMS",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)

        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Section: Entrée du résumé
        input_label = tk.Label(
            main_frame,
            text="📝 Entrez le résumé du film:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0'
        )
        input_label.pack(anchor=tk.W, pady=(0, 5))

        # Zone de texte avec scrollbar
        self.text_input = scrolledtext.ScrolledText(
            main_frame,
            height=10,
            font=('Arial', 11),
            wrap=tk.WORD,
            borderwidth=2,
            relief=tk.GROOVE
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Placeholder
        placeholder = "Exemple: A young wizard discovers his magical powers and attends a school of witchcraft and wizardry, where he makes friends and battles dark forces..."
        self.text_input.insert('1.0', placeholder)
        self.text_input.config(fg='gray')

        # Bind events pour le placeholder
        self.text_input.bind('<FocusIn>', self.on_entry_click)
        self.text_input.bind('<FocusOut>', self.on_focus_out)

        # Frame pour les boutons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=(0, 15))

        # Bouton Prédire
        predict_btn = tk.Button(
            button_frame,
            text="🔮 PRÉDIRE LE GENRE",
            command=self.predict_genre,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10
        )
        predict_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Bouton Effacer
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ EFFACER",
            command=self.clear_text,
            font=('Arial', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10
        )
        clear_btn.pack(side=tk.LEFT)

        # Bouton Exemple
        example_btn = tk.Button(
            button_frame,
            text="💡 EXEMPLE",
            command=self.load_example,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3,
            padx=20,
            pady=10
        )
        example_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Section: Résultats
        results_label = tk.Label(
            main_frame,
            text="🎯 Prédictions des Modèles:",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0'
        )
        results_label.pack(anchor=tk.W, pady=(10, 5))

        # Frame pour les résultats
        self.results_frame = tk.Frame(main_frame, bg='white', relief=tk.GROOVE, borderwidth=2)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Labels pour les résultats (initialement vides)
        self.result_labels = {}
        for i, model_name in enumerate(['Naive Bayes', 'SVM', 'Random Forest']):
            frame = tk.Frame(self.results_frame, bg='white')
            frame.pack(fill=tk.X, padx=10, pady=8)

            model_label = tk.Label(
                frame,
                text=f"{model_name}:",
                font=('Arial', 11, 'bold'),
                bg='white',
                width=15,
                anchor=tk.W
            )
            model_label.pack(side=tk.LEFT)

            result_label = tk.Label(
                frame,
                text="En attente...",
                font=('Arial', 11),
                bg='white',
                fg='gray',
                anchor=tk.W
            )
            result_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.result_labels[model_name] = result_label

        # Footer
        footer_frame = tk.Frame(self.root, bg='#34495e', height=40)
        footer_frame.pack(fill=tk.X)
        footer_frame.pack_propagate(False)

        footer_label = tk.Label(
            footer_frame,
            text="Prédicteur de Genres | Modèles Améliorés (Undersampling) | Naive Bayes • SVM • Random Forest",
            font=('Arial', 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        footer_label.pack(pady=10)

    def on_entry_click(self, event):
        """Supprime le placeholder quand on clique"""
        if self.text_input.get('1.0', 'end-1c') == "Exemple: A young wizard discovers his magical powers and attends a school of witchcraft and wizardry, where he makes friends and battles dark forces...":
            self.text_input.delete('1.0', tk.END)
            self.text_input.config(fg='black')

    def on_focus_out(self, event):
        """Remet le placeholder si vide"""
        if self.text_input.get('1.0', 'end-1c').strip() == "":
            self.text_input.insert('1.0', "Exemple: A young wizard discovers his magical powers and attends a school of witchcraft and wizardry, where he makes friends and battles dark forces...")
            self.text_input.config(fg='gray')

    def clear_text(self):
        """Efface le texte et réinitialise les résultats"""
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', "Exemple: A young wizard discovers his magical powers and attends a school of witchcraft and wizardry, where he makes friends and battles dark forces...")
        self.text_input.config(fg='gray')

        for label in self.result_labels.values():
            label.config(text="En attente...", fg='gray', font=('Arial', 11))

    def load_example(self):
        """Charge un exemple de résumé"""
        examples = [
            "A group of teenagers encounter a masked killer at a summer camp. One by one, they are hunted down in brutal ways. The survivors must find a way to escape before they become the next victims.",

            "A bumbling detective is assigned to solve a series of jewel thefts. Despite his incompetence, he accidentally stumbles upon the real criminals while creating chaos everywhere he goes.",

            "An undercover cop infiltrates a dangerous criminal organization. As he gets deeper into the gang, he must balance his mission with staying alive and protecting his true identity.",

            "A woman struggles with her past trauma while trying to build a new life. She meets someone who helps her heal, but old wounds resurface threatening her happiness.",

            "A team of superheroes must unite to stop an alien invasion threatening Earth. They overcome their differences and personal conflicts to save humanity from total destruction."
        ]

        import random
        example = random.choice(examples)

        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', example)
        self.text_input.config(fg='black')

    def predict_genre(self):
        """
        Prédit le genre avec les 3 modèles
        """
        # Récupérer le texte
        text = self.text_input.get('1.0', 'end-1c').strip()

        # Vérifier si c'est le placeholder
        if text == "Exemple: A young wizard discovers his magical powers and attends a school of witchcraft and wizardry, where he makes friends and battles dark forces..." or len(text) == 0:
            messagebox.showwarning(
                "Attention",
                "Veuillez entrer un résumé de film avant de prédire!"
            )
            return

        try:
            # Log séparateur pour nouvelle prédiction
            self.logger.info("")
            self.logger.info("="*80)
            self.logger.info("NOUVELLE PRÉDICTION")
            self.logger.info("="*80)

            # Log du texte original
            self.logger.info(f"TEXTE ORIGINAL ({len(text)} caractères):")
            self.logger.info(f"  {text[:200]}{'...' if len(text) > 200 else ''}")

            # Nettoyer le texte (même preprocessing que l'entraînement)
            cleaned_text = self.clean_text(text)

            # Log du texte nettoyé
            self.logger.info(f"TEXTE NETTOYÉ ({len(cleaned_text)} caractères):")
            self.logger.info(f"  {cleaned_text[:200]}{'...' if len(cleaned_text) > 200 else ''}")

            if len(cleaned_text) == 0:
                messagebox.showerror(
                    "Erreur",
                    "Le texte nettoyé est vide. Veuillez entrer un résumé valide."
                )
                return

            # Vectoriser le texte
            text_tfidf = self.vectorizer.transform([cleaned_text])

            # Log début des prédictions
            self.logger.info("")
            self.logger.info("PRÉDICTIONS DES MODÈLES:")
            self.logger.info("-" * 80)

            # Prédire avec chaque modèle
            for model_name, model in self.models.items():
                prediction = model.predict(text_tfidf)[0]

                # Obtenir les probabilités si disponible
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(text_tfidf)[0]
                    max_proba = probas.max()
                    confidence = f"({max_proba*100:.1f}% confiance)"

                    # Log détaillé avec toutes les probabilités
                    self.logger.info(f"{model_name}:")
                    self.logger.info(f"  → Prédiction: {prediction.upper()}")
                    self.logger.info(f"  → Confiance: {max_proba*100:.1f}%")
                    self.logger.info(f"  → Distribution complète:")
                    for i, genre in enumerate(sorted(self.genres)):
                        self.logger.info(f"      {genre}: {probas[i]*100:.2f}%")
                else:
                    # Pour SVM, utiliser decision_function
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(text_tfidf)[0]
                        # Normaliser les scores
                        max_score = scores.max()
                        confidence = f"(score: {max_score:.2f})"

                        # Log détaillé avec tous les scores
                        self.logger.info(f"{model_name}:")
                        self.logger.info(f"  → Prédiction: {prediction.upper()}")
                        self.logger.info(f"  → Score max: {max_score:.2f}")
                        self.logger.info(f"  → Scores complets:")
                        for i, genre in enumerate(sorted(self.genres)):
                            self.logger.info(f"      {genre}: {scores[i]:.2f}")
                    else:
                        confidence = ""
                        self.logger.info(f"{model_name}:")
                        self.logger.info(f"  → Prédiction: {prediction.upper()}")

                # Emoji selon le genre
                genre_emojis = {
                    'action': '💥',
                    'comedy': '😂',
                    'drama': '🎭',
                    'horror': '👻',
                    'unknown': '❓'
                }

                emoji = genre_emojis.get(prediction, '🎬')

                # Couleur selon le genre
                genre_colors = {
                    'action': '#e74c3c',
                    'comedy': '#f39c12',
                    'drama': '#9b59b6',
                    'horror': '#34495e',
                    'unknown': '#95a5a6'
                }

                color = genre_colors.get(prediction, '#2c3e50')

                # Mettre à jour le label
                result_text = f"{emoji} {prediction.upper()} {confidence}"
                self.result_labels[model_name].config(
                    text=result_text,
                    fg=color,
                    font=('Arial', 12, 'bold')
                )

            # Log de fin
            self.logger.info("-" * 80)
            self.logger.info("✓ Prédiction terminée avec succès")
            self.logger.info("="*80)

            # Afficher un message de succès
            messagebox.showinfo(
                "Prédiction Terminée",
                "Les 3 modèles ont prédit le genre du film!\n\n"
                f"Texte nettoyé (préprocessing):\n{cleaned_text[:200]}..."
            )

        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la prédiction: {e}")
            messagebox.showerror(
                "Erreur",
                f"Erreur lors de la prédiction:\n{e}"
            )


def main():
    """
    Fonction principale - Lance l'application
    """
    root = tk.Tk()
    app = MovieGenrePredictor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
