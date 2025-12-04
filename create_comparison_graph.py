#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Génération du Graphique Comparatif
Comparaison des 3 approches : Baseline vs Undersampling vs Weighted
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Données des résultats expérimentaux
models = ['Naive Bayes', 'SVM', 'Random Forest']
baseline = [38.62, 35.00, 32.00]
undersampling = [55.00, 52.00, 48.00]
weighted = [20.45, 34.82, 27.73]

# Configuration du graphique
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(models))
width = 0.25

# Barres
bars1 = ax.bar(x - width, baseline, width, label='Baseline', color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x, undersampling, width, label='Undersampling (TP)', color='#27ae60', alpha=0.8)
bars3 = ax.bar(x + width, weighted, width, label='Weighted (Slide 84)', color='#e74c3c', alpha=0.8)

# Ajouter les valeurs sur les barres
def add_value_labels(bars, color='black'):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

add_value_labels(bars1, color='#7f8c8d')
add_value_labels(bars2, color='#229954')
add_value_labels(bars3, color='#c0392b')

# Ligne de référence à 50%
ax.axhline(y=50, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label='Seuil 50% (Acceptable)')

# Annotations des gains
gains = [
    (0, undersampling[0] - baseline[0], '+16.4%'),
    (1, undersampling[1] - baseline[1], '+17.0%'),
    (2, undersampling[2] - baseline[2], '+16.0%')
]

for idx, (pos, gain, label) in enumerate(gains):
    y_pos = max(undersampling[idx], baseline[idx]) + 4
    ax.annotate(label,
                xy=(pos, y_pos),
                xytext=(pos, y_pos + 3),
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f4e6', edgecolor='#27ae60', linewidth=1.5))

# Titre et labels
ax.set_title('Comparaison des Approches de Rééquilibrage\nF1-Score pour le Genre ACTION (Classe Minoritaire)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Modèles', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score ACTION (%)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 70)

# Grille
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Légende
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, shadow=True)

# Annotations explicatives
textstr = '\n'.join([
    '✅ Undersampling (TP Fraudes): Meilleure approche (+16.5% en moyenne)',
    '❌ Weighted (Slide 84): Échec sur ce déséquilibre (ratio 1:5.5)',
    '⚠️  Baseline: Performances faibles (biais vers classes majoritaires)'
])

props = dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, edgecolor='#34495e', linewidth=2)
ax.text(0.98, 0.03, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=props, family='monospace')

# Ajuster layout
plt.tight_layout()

# Sauvegarder
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'comparison_3_approaches.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Graphique sauvegardé: {output_path}")

# Créer aussi un graphique avec tous les modèles côte à côte
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Comparaison Détaillée par Modèle - F1-Score ACTION',
              fontsize=16, fontweight='bold', y=1.02)

approaches = ['Baseline', 'Undersampling\n(TP)', 'Weighted\n(Slide 84)']

# Naive Bayes
data_nb = [baseline[0], undersampling[0], weighted[0]]
colors_nb = ['#95a5a6', '#27ae60', '#e74c3c']
bars_nb = ax1.bar(approaches, data_nb, color=colors_nb, alpha=0.8)
ax1.set_title('Naive Bayes', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1-Score ACTION (%)', fontsize=12)
ax1.set_ylim(0, 70)
ax1.axhline(y=50, color='#3498db', linestyle='--', alpha=0.7)
ax1.grid(axis='y', alpha=0.3)
for bar in bars_nb:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# SVM
data_svm = [baseline[1], undersampling[1], weighted[1]]
colors_svm = ['#95a5a6', '#27ae60', '#e74c3c']
bars_svm = ax2.bar(approaches, data_svm, color=colors_svm, alpha=0.8)
ax2.set_title('SVM', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 70)
ax2.axhline(y=50, color='#3498db', linestyle='--', alpha=0.7)
ax2.grid(axis='y', alpha=0.3)
for bar in bars_svm:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Random Forest
data_rf = [baseline[2], undersampling[2], weighted[2]]
colors_rf = ['#95a5a6', '#27ae60', '#e74c3c']
bars_rf = ax3.bar(approaches, data_rf, color=colors_rf, alpha=0.8)
ax3.set_title('Random Forest', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 70)
ax3.axhline(y=50, color='#3498db', linestyle='--', alpha=0.7)
ax3.grid(axis='y', alpha=0.3)
for bar in bars_rf:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()

output_path2 = output_dir / 'comparison_by_model.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Graphique détaillé sauvegardé: {output_path2}")

# Statistiques résumées
print("\n" + "="*60)
print("STATISTIQUES COMPARATIVES")
print("="*60)

print("\n📊 F1-Score ACTION Moyen:")
print(f"   Baseline:      {np.mean(baseline):.2f}%")
print(f"   Undersampling: {np.mean(undersampling):.2f}% (+{np.mean(undersampling)-np.mean(baseline):.2f}%)")
print(f"   Weighted:      {np.mean(weighted):.2f}% ({np.mean(weighted)-np.mean(baseline):+.2f}%)")

print(f"\n🏆 Meilleur Modèle: Naive Bayes (Undersampling)")
print(f"   F1-Score ACTION: {undersampling[0]:.2f}%")
print(f"   Gain vs Baseline: +{undersampling[0] - baseline[0]:.2f}%")

print(f"\n❌ Pire Approche: Weighted (Naive Bayes)")
print(f"   F1-Score ACTION: {weighted[0]:.2f}%")
print(f"   Perte vs Baseline: {weighted[0] - baseline[0]:.2f}%")

print(f"\n✅ Conclusion: Undersampling est la meilleure approche")
print(f"   Gain moyen: +{np.mean(undersampling)-np.mean(baseline):.2f}%")
print(f"   Tous les modèles améliorés: {all(u > b for u, b in zip(undersampling, baseline))}")

print("\n" + "="*60)
