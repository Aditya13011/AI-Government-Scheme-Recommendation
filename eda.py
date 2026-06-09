# ============================================================
# FILE: src/eda.py
# PURPOSE: Visualize data to understand patterns
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle, os


df = pd.read_csv('data/government_schemes_dataset.csv')



os.makedirs('reports/figures', exist_ok=True)
# Create folder to save all charts

sns.set_style("whitegrid")
# Sets a clean white background for all plots

# ── Plot 1: Age Distribution ───────────────────────────────────
plt.figure(figsize=(8,4))
# figsize=(width, height) in inches

sns.histplot(df['age'], bins=20, kde=True, color='steelblue')
# histplot: bar chart showing frequency of age ranges
# bins=20: divide age range into 20 bars
# kde=True: draw a smooth curve on top

plt.title('Age Distribution of Citizens')
plt.xlabel('Age'); plt.ylabel('Count')
plt.tight_layout()
# tight_layout() prevents labels from being cut off

plt.savefig('reports/figures/age_distribution.png', dpi=150)
plt.show()

# ── Plot 2: Income Distribution (log scale) ────────────────────
plt.figure(figsize=(8,4))
sns.histplot(df['income_annual'], bins=30, kde=True, color='coral', log_scale=True)
# log_scale=True: useful when data spans many orders of magnitude (₹30k to ₹10L)
plt.title('Annual Income Distribution')
plt.xlabel('Income (₹, log scale)'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig('reports/figures/income_distribution.png', dpi=150)
plt.show()

# ── Plot 3: Occupation Count ───────────────────────────────────
plt.figure(figsize=(8,4))
df['occupation'].value_counts().plot(kind='bar', color='mediumseagreen', edgecolor='black')
# value_counts(): counts frequency of each category
# .plot(kind='bar'): bar chart
plt.title('Occupation Distribution')
plt.xlabel('Occupation'); plt.ylabel('Count')
plt.xticks(rotation=30)   # Rotate x-axis labels 30 degrees for readability
plt.tight_layout()
plt.savefig('reports/figures/occupation_distribution.png', dpi=150)
plt.show()

# ── Plot 4: Scheme Eligibility Counts ─────────────────────────
LABEL_COLS = ['PM_KISAN','PMAY','Ayushman_Bharat','PM_Scholarship',
              'Kisan_Credit_Card','PMEGP','Disability_Pension','Minority_Scholarship']

scheme_counts = df[LABEL_COLS].sum().sort_values(ascending=False)
# .sum() on binary columns = total number of eligible people per scheme

plt.figure(figsize=(10,5))
scheme_counts.plot(kind='barh', color='dodgerblue', edgecolor='black')
# barh = horizontal bar chart (easier to read long labels)
plt.title('Number of Citizens Eligible per Scheme')
plt.xlabel('Count'); plt.ylabel('Scheme')
plt.tight_layout()
plt.savefig('reports/figures/scheme_distribution.png', dpi=150)
plt.show()

# ── Plot 5: Correlation Heatmap ────────────────────────────────
plt.figure(figsize=(14,8))
numeric_df = df.select_dtypes(include=[np.float64, np.int64])
# select_dtypes: keep only numeric columns (heatmap needs numbers)

corr = numeric_df.corr()
# .corr() computes pairwise correlation: -1 (opposite) to +1 (same direction)

sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, annot_kws={'size':7})
# annot=True: show numbers inside each cell
# fmt='.2f': show 2 decimal places
# cmap='coolwarm': red=positive, blue=negative correlation
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('reports/figures/correlation_heatmap.png', dpi=150)
plt.show()

# ── Plot 6: Income vs Scheme Eligibility (Boxplot) ────────────
fig, axes = plt.subplots(2, 4, figsize=(16,8))
# subplots(rows, cols): create a grid of plots
axes = axes.flatten()   # Convert 2D array of axes to 1D for easy looping

for i, scheme in enumerate(LABEL_COLS):
    sns.boxplot(x=df[scheme], y=df['income_annual'],
                palette=['salmon','lightgreen'], ax=axes[i])
    # boxplot shows income spread for eligible (1) vs not eligible (0)
    axes[i].set_title(scheme)
    axes[i].set_xlabel('Eligible (1=Yes)')
    axes[i].set_ylabel('Income (₹)')

plt.suptitle('Income Distribution by Scheme Eligibility', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/income_by_scheme.png', dpi=150)
plt.show()

# ── Plot 7: Category vs Scheme Eligibility ─────────────────────
cat_scheme = df.groupby('category')[LABEL_COLS].mean()
# groupby('category'): group rows by category (General, OBC, SC, ST)
# [LABEL_COLS].mean(): average eligibility rate per category per scheme

ax = cat_scheme.T.plot(kind='bar', figsize=(12,5), colormap='Set2')
# .T = transpose (schemes on x-axis, categories as bars)
plt.title('Average Scheme Eligibility by Category')
plt.xlabel('Scheme'); plt.ylabel('Eligibility Rate')
plt.xticks(rotation=30); plt.legend(title='Category')
plt.tight_layout()
plt.savefig('reports/figures/category_vs_scheme.png', dpi=150)
plt.show()
plt.close()

print("✅ EDA complete. All figures saved to reports/figures/")