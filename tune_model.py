# ============================================================
# FILE: src/tune_model.py
# PURPOSE: Find best hyperparameters using GridSearchCV
# ============================================================

import pandas as pd
import numpy as np
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier
from sklearn.multioutput     import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics         import f1_score, make_scorer

X_train = pd.read_csv('data/X_processed.csv').iloc[:4000]
y_train = pd.read_csv('data/y_labels.csv').iloc[:4000]
X_test  = pd.read_csv('data/X_processed.csv').iloc[4000:]
y_test  = pd.read_csv('data/y_labels.csv').iloc[4000:]
# Using iloc to slice — first 4000 for train, rest for test

# ── Define hyperparameter grid ─────────────────────────────────
param_grid = {
    'estimator__n_estimators' : [50, 100, 200],
    # Number of trees: try 50, 100, 200
    # 'estimator__' prefix is needed because of MultiOutputClassifier wrapper

    'estimator__max_depth'    : [8, 12, None],
    # None = trees grow until all leaves are pure (can overfit)

    'estimator__min_samples_split': [2, 5, 10],
    # Minimum samples needed to split a node
}
# Total combinations: 3 × 3 × 3 = 27 configurations

# ── Custom F1 scorer for multi-label ──────────────────────────
def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

scorer = make_scorer(macro_f1)
# make_scorer wraps a custom function so sklearn can use it

# ── Run Grid Search ────────────────────────────────────────────
rf_base = MultiOutputClassifier(
    RandomForestClassifier(random_state=42, n_jobs=-1)
)

grid_search = GridSearchCV(
    estimator  = rf_base,
    param_grid = param_grid,
    cv         = 3,           # 3-fold cross-validation per config
    scoring    = scorer,
    verbose    = 2,           # Print progress
    n_jobs     = -1           # Parallelize across CPU cores
)
# GridSearchCV tries every combination of params and picks the best
# cross-validation: split training data 3 ways, train on 2/test on 1, rotate

print("🔍 Starting Grid Search (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Best Parameters: {grid_search.best_params_}")
print(f"   Best CV F1 Score: {grid_search.best_score_:.4f}")

# ── Evaluate tuned model on test set ──────────────────────────
best_model = grid_search.best_estimator_
y_pred     = best_model.predict(X_test)

f1 = f1_score(np.array(y_test), np.array(y_pred),
              average='macro', zero_division=0)
print(f"   Test F1 Score (tuned): {f1:.4f}")

# ── Save tuned model ───────────────────────────────────────────
with open('models/tuned_random_forest.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Tuned model saved → models/tuned_random_forest.pkl")

# ── Cross-Validation Score of Tuned Model ─────────────────────
# (Optional deeper validation)
cv_scores = cross_val_score(best_model, X_train, y_train,
                             cv=5, scoring=scorer)
# cv=5: 5-fold cross-validation
# Each fold: train on 4/5 of data, test on 1/5, repeat 5 times

print(f"\n📊 5-Fold CV Scores: {cv_scores.round(4)}")
print(f"   Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# ± std: shows how consistent the model is across different data splits