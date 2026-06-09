# ============================================================
# FILE: src/train_model.py
# PURPOSE: Train Decision Tree, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings('ignore')
# Suppress non-critical warnings to keep output clean

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.multioutput   import MultiOutputClassifier
# MultiOutputClassifier: wraps any classifier to handle multiple labels simultaneously

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Optional XGBoost (install if available) ────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Skipping. Run: pip install xgboost")

# ── Load preprocessed data ─────────────────────────────────────
X = pd.read_csv('data/X_processed.csv')
y = pd.read_csv('data/y_labels.csv')
# X: feature matrix (inputs), y: label matrix (outputs)

LABEL_COLS = y.columns.tolist()
# tolist() converts pandas Index to Python list

# ── Train-Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20% data for testing, 80% for training
    random_state=42    # Fix randomness for reproducibility
)

print(f"Train size: {X_train.shape[0]} rows")
print(f"Test  size: {X_test.shape[0]} rows")

# ── Evaluation Helper Function ─────────────────────────────────
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """Train a model and return evaluation metrics."""

    model.fit(X_tr, y_tr)
    # .fit(X, y): train the model — it learns patterns from training data

    y_pred = model.predict(X_te)
    # .predict(X): use trained model to predict labels for test data

    # Convert predictions and true values to arrays for metric calculation
    y_te_arr   = np.array(y_te)
    y_pred_arr = np.array(y_pred)

    # Accuracy: What % of ALL label predictions are correct?
    acc = accuracy_score(y_te_arr.ravel(), y_pred_arr.ravel())
    # .ravel() flattens 2D array to 1D for overall accuracy

    # Precision: Of all predicted positives, how many are actually positive?
    prec = precision_score(y_te_arr, y_pred_arr, average='macro', zero_division=0)
    # average='macro': calculate metric per label, then take unweighted mean

    # Recall: Of all actual positives, how many did we correctly predict?
    rec  = recall_score(y_te_arr, y_pred_arr, average='macro', zero_division=0)

    # F1 Score: Harmonic mean of Precision and Recall
    # Best when you need balance between precision and recall
    f1   = f1_score(y_te_arr, y_pred_arr, average='macro', zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Model : {name}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*50}")

    return {'Model':name, 'Accuracy':acc, 'Precision':prec,
            'Recall':rec, 'F1':f1, 'trained_model':model}

# ── Model 1: Decision Tree ─────────────────────────────────────
dt = MultiOutputClassifier(
    DecisionTreeClassifier(
        max_depth=10,       # Limit tree depth to prevent overfitting
        min_samples_split=5,# Node must have at least 5 samples to split
        random_state=42
    )
)
# MultiOutputClassifier trains one DecisionTree per label (8 trees total)

dt_results = evaluate_model("Decision Tree", dt,
                             X_train, X_test, y_train, y_test)

# ── Model 2: Random Forest ─────────────────────────────────────
rf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,   # Train 100 trees and combine predictions
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1           # Use all CPU cores for faster training
    )
)
# RandomForest = many DecisionTrees voting together (ensemble method)
# More trees = more robust predictions, less overfitting

rf_results = evaluate_model("Random Forest", rf,
                             X_train, X_test, y_train, y_test)

# ── Model 3: XGBoost (if available) ───────────────────────────
if XGBOOST_AVAILABLE:
    xgb = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,  # How fast model learns; lower = more careful
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            verbosity=0         # Silent mode
        )
    )
    xgb_results = evaluate_model("XGBoost", xgb,
                                  X_train, X_test, y_train, y_test)

# ── Collect Results ────────────────────────────────────────────
all_results = [dt_results, rf_results]
if XGBOOST_AVAILABLE:
    all_results.append(xgb_results)

results_df = pd.DataFrame([{k:v for k,v in r.items() if k!='trained_model'}
                            for r in all_results])
# Build summary table, excluding the model object itself

print("\n📊 Model Comparison Summary:")
print(results_df.to_string(index=False))

# ── Save Best Model (Random Forest usually wins) ───────────────
best = max(all_results, key=lambda r: r['F1'])
# max() with key= finds the result dict with highest F1 score

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best['trained_model'], f)
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_results['trained_model'], f)
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_results['trained_model'], f)

# Save label column names for later use
with open('models/label_cols.pkl', 'wb') as f:
    pickle.dump(LABEL_COLS, f)

print(f"\n✅ Best model: {best['Model']} (F1={best['F1']:.4f}) saved.")

# ── Visualization: Model Comparison Bar Chart ──────────────────
metrics = ['Accuracy','Precision','Recall','F1']
x       = np.arange(len(metrics))   # [0, 1, 2, 3]
width   = 0.25                       # Bar width

fig, ax = plt.subplots(figsize=(10,5))
for i, row in results_df.iterrows():
    # iterrows(): loop through DataFrame row by row
    offset = (i - len(results_df)/2 + 0.5) * width
    ax.bar(x + offset, row[metrics], width, label=row['Model'])
    # x + offset: shifts bars slightly so they don't overlap

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.legend()
plt.tight_layout()
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/model_comparison.png', dpi=150)
plt.show()
print("✅ Comparison chart saved.")