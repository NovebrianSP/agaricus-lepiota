import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# Set encoding untuk Windows
import sys
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__.detach())

# Set style untuk visualisasi yang lebih menarik
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model dan encoder
with open(os.path.join(BASE_DIR, 'rf_mushroom.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'le_dict.pkl'), 'rb') as f:
    le_dict = pickle.load(f)

# Preprocessing data
df = pd.read_csv('agaricus-lepiota-mapped.csv')
column_mapping = {
    'class': 'kelas',
    'odor': 'bau',
    'spore-print-color': 'warna_spora',
    'gill-color': 'warna_insang',
    'gill-size': 'ukuran_insang',
    'bruises': 'memar',
    'population': 'populasi',
    'habitat': 'habitat'
}
df.rename(columns=column_mapping, inplace=True)
selected_cols = ['bau', 'warna_spora', 'warna_insang', 'ukuran_insang', 'memar', 'populasi', 'habitat', 'kelas']
df = df[selected_cols]

# Encode data
df_encoded = df.copy()
for col in df_encoded.columns:
    le = le_dict[col]
    df_encoded[col] = le.transform(df_encoded[col])

X = df_encoded.drop('kelas', axis=1)
y = df_encoded['kelas']

# Membuat folder untuk hasil visualisasi
results_dir = os.path.join(BASE_DIR, 'evaluation_results')
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("EVALUASI MODEL KLASIFIKASI JAMUR")
print("="*60)

# 1. Cross-Validation Evaluation
print("\n1. CROSS-VALIDATION ANALYSIS")
print("-" * 40)
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualisasi CV Scores
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(range(1, 6), cv_scores, color='skyblue', alpha=0.7)
ax[0].axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax[0].set_xlabel('Fold')
ax[0].set_ylabel('Accuracy Score')
ax[0].set_title('Cross-Validation Scores by Fold')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# Box plot untuk distribusi CV scores
ax[1].boxplot(cv_scores, patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
ax[1].set_ylabel('Accuracy Score')
ax[1].set_title('Cross-Validation Score Distribution')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '1_cross_validation_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Classification Report
print("\n2. CLASSIFICATION REPORT")
print("-" * 40)
y_pred = model.predict(X)
report = classification_report(y, y_pred, target_names=['beracun', 'bisa dimakan'], output_dict=True)
print(classification_report(y, y_pred, target_names=['beracun', 'bisa dimakan']))

# Visualisasi Classification Report
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df = pd.DataFrame(report).iloc[:-1, :-2].T
metrics_df.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Classification Report Metrics', fontsize=16, fontweight='bold')
ax.set_xlabel('Classes', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, '2_classification_report.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrix
print("\n3. CONFUSION MATRIX ANALYSIS")
print("-" * 40)
cm = confusion_matrix(y, y_pred)
print(f"True Negatives (Beracun benar): {cm[0,0]}")
print(f"False Positives (Salah prediksi bisa dimakan): {cm[0,1]}")
print(f"False Negatives (Salah prediksi beracun): {cm[1,0]}")
print(f"True Positives (Bisa dimakan benar): {cm[1,1]}")

# Visualisasi Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix Raw Numbers
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Beracun', 'Bisa Dimakan'],
            yticklabels=['Beracun', 'Bisa Dimakan'])
ax1.set_title('Confusion Matrix (Raw Numbers)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted Class')
ax1.set_ylabel('True Class')

# Confusion Matrix Percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
            xticklabels=['Beracun', 'Bisa Dimakan'],
            yticklabels=['Beracun', 'Bisa Dimakan'])
ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Class')
ax2.set_ylabel('True Class')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '3_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Feature Importance Analysis
print("\n4. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Konversi nama fitur ke bahasa Indonesia untuk visualisasi
feature_names_id = {
    'bau': 'Bau',
    'warna_spora': 'Warna Spora',
    'warna_insang': 'Warna Insang',
    'ukuran_insang': 'Ukuran Insang',
    'memar': 'Memar',
    'populasi': 'Populasi',
    'habitat': 'Habitat'
}

feature_importance['feature_id'] = feature_importance['feature'].map(feature_names_id)
print(feature_importance[['feature_id', 'importance']])

# Visualisasi Feature Importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
bars = ax1.bar(feature_importance['feature_id'], feature_importance['importance'], color=colors)
ax1.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
ax1.set_xlabel('Features')
ax1.set_ylabel('Importance Score')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Pie chart
ax2.pie(feature_importance['importance'], labels=feature_importance['feature_id'], 
        autopct='%1.1f%%', startangle=90, colors=colors)
ax2.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '4_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Probability Distribution Analysis
print("\n5. PROBABILITY DISTRIBUTION ANALYSIS")
print("-" * 40)
probabilities = model.predict_proba(X)
edible_probs = probabilities[:, 1]
poisonous_probs = probabilities[:, 0]

print(f"Mean edible probability: {edible_probs.mean():.4f}")
print(f"Std edible probability: {edible_probs.std():.4f}")
print(f"Min edible probability: {edible_probs.min():.4f}")
print(f"Max edible probability: {edible_probs.max():.4f}")

# Visualisasi Probability Distribution
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Histogram probabilitas edible
ax1.hist(edible_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
ax1.axvline(edible_probs.mean(), color='red', linestyle='--', 
           label=f'Mean: {edible_probs.mean():.3f}')
ax1.set_title('Distribution of Edible Probabilities', fontweight='bold')
ax1.set_xlabel('Probability')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histogram probabilitas poisonous
ax2.hist(poisonous_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(poisonous_probs.mean(), color='blue', linestyle='--', 
           label=f'Mean: {poisonous_probs.mean():.3f}')
ax2.set_title('Distribution of Poisonous Probabilities', fontweight='bold')
ax2.set_xlabel('Probability')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Box plot perbandingan
data_to_plot = [edible_probs, poisonous_probs]
box_plot = ax3.boxplot(data_to_plot, labels=['Edible', 'Poisonous'], 
                       patch_artist=True)
box_plot['boxes'][0].set_facecolor('lightgreen')
box_plot['boxes'][1].set_facecolor('lightcoral')
ax3.set_title('Probability Distribution Comparison', fontweight='bold')
ax3.set_ylabel('Probability')
ax3.grid(True, alpha=0.3)

# Scatter plot untuk melihat separasi
true_labels = le_dict['kelas'].inverse_transform(y)
edible_mask = true_labels == 'bisa dimakan'
poisonous_mask = true_labels == 'beracun'

ax4.scatter(range(sum(edible_mask)), edible_probs[edible_mask], 
           alpha=0.6, c='green', label='True Edible', s=1)
ax4.scatter(range(sum(poisonous_mask)), edible_probs[poisonous_mask], 
           alpha=0.6, c='red', label='True Poisonous', s=1)
ax4.set_title('Probability Distribution by True Class', fontweight='bold')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Edible Probability')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '5_probability_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Threshold Analysis untuk Sistem Scoring (FIXED - Tanpa karakter Unicode)
print("\n6. THRESHOLD ANALYSIS FOR SCORING SYSTEM")
print("-" * 40)
edible_scores = edible_probs * 10  # Convert to 0-10 scale

# Analisis distribusi berdasarkan threshold
high_confidence = (edible_scores >= 8.0).sum()
medium_confidence = ((edible_scores >= 6.0) & (edible_scores < 8.0)).sum()
low_confidence = (edible_scores < 6.0).sum()

# Menggunakan karakter ASCII biasa sebagai pengganti Unicode
print(f"High Confidence (>=8.0): {high_confidence} samples ({high_confidence/len(edible_scores)*100:.1f}%)")
print(f"Medium Confidence (6.0-7.9): {medium_confidence} samples ({medium_confidence/len(edible_scores)*100:.1f}%)")
print(f"Low Confidence (<6.0): {low_confidence} samples ({low_confidence/len(edible_scores)*100:.1f}%)")

# Visualisasi Threshold Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram dengan threshold lines
ax1.hist(edible_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(8.0, color='green', linestyle='--', linewidth=2, label='High Confidence (>=8.0)')
ax1.axvline(6.0, color='orange', linestyle='--', linewidth=2, label='Medium Confidence (>=6.0)')
ax1.set_title('Score Distribution with Thresholds', fontweight='bold')
ax1.set_xlabel('Edible Score (0-10)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Pie chart untuk threshold distribution
threshold_counts = [high_confidence, medium_confidence, low_confidence]
threshold_labels = ['High Confidence\n(>=8.0)', 'Medium Confidence\n(6.0-7.9)', 'Low Confidence\n(<6.0)']
colors_threshold = ['green', 'orange', 'red']

ax2.pie(threshold_counts, labels=threshold_labels, autopct='%1.1f%%', 
        colors=colors_threshold, startangle=90)
ax2.set_title('Threshold-based Classification Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '6_threshold_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Summary Report
print("\n7. EXECUTIVE SUMMARY")
print("=" * 60)
accuracy = accuracy_score(y, y_pred)
print(f"Overall Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Cross-Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Model Stability: {'Excellent' if cv_scores.std() < 0.01 else 'Good' if cv_scores.std() < 0.02 else 'Fair'}")
print(f"Most Important Feature: {feature_importance.iloc[0]['feature_id']}")
print(f"Least Important Feature: {feature_importance.iloc[-1]['feature_id']}")
print(f"High Confidence Predictions: {high_confidence/len(edible_scores)*100:.1f}%")

# Membuat summary plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Model performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
edible_scores_metrics = [report['bisa dimakan']['precision'], 
                        report['bisa dimakan']['recall'], 
                        report['bisa dimakan']['f1-score']]
poisonous_scores_metrics = [report['beracun']['precision'], 
                           report['beracun']['recall'], 
                           report['beracun']['f1-score']]

x = np.arange(len(metrics[1:]))
width = 0.35

ax1.bar(x - width/2, edible_scores_metrics, width, label='Bisa Dimakan', color='green', alpha=0.7)
ax1.bar(x + width/2, poisonous_scores_metrics, width, label='Beracun', color='red', alpha=0.7)
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance by Class', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics[1:])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Feature importance top 5
top_features = feature_importance.head(5)
ax2.barh(top_features['feature_id'], top_features['importance'], color='skyblue')
ax2.set_xlabel('Importance Score')
ax2.set_title('Top 5 Most Important Features', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Cross-validation scores
ax3.plot(range(1, 6), cv_scores, marker='o', linewidth=2, markersize=8, color='purple')
ax3.fill_between(range(1, 6), cv_scores, alpha=0.3, color='purple')
ax3.set_xlabel('Fold')
ax3.set_ylabel('Accuracy')
ax3.set_title('Cross-Validation Performance', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Threshold distribution
ax4.pie(threshold_counts, labels=threshold_labels, autopct='%1.1f%%', 
        colors=colors_threshold, startangle=90)
ax4.set_title('Prediction Confidence Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, '7_executive_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📊 All visualizations saved to: {results_dir}")
print("📋 Generated files:")
for i in range(1, 8):
    filename = f"{i}_*.png"
    print(f"   - {filename}")

print("\n✅ Evaluation complete! Check the 'evaluation_results' folder for all PNG files.")