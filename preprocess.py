# ============================================================
# FILE: src/preprocess.py
# PURPOSE: Clean data, encode categories, scale numbers
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle, os

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv('data/government_schemes_dataset.csv')
# read_csv loads CSV file into a DataFrame (like an Excel table in Python)

print("Shape:", df.shape)          # (rows, columns)
print(df.dtypes)                   # Shows data type of each column
print(df.isnull().sum())           # Count missing values per column
# isnull() returns True where value is missing; .sum() counts Trues

# ── Define feature and label columns ──────────────────────────
FEATURE_COLS = ['age','gender','income_annual','occupation','education',
                'category','state','is_farmer','has_disability',
                'land_holding_acres','has_bpl_card','family_size']

LABEL_COLS   = ['PM_KISAN','PMAY','Ayushman_Bharat','PM_Scholarship',
                'Kisan_Credit_Card','PMEGP','Disability_Pension',
                'Minority_Scholarship']

CATEGORICAL  = ['gender','occupation','education','category','state']
NUMERICAL    = ['age','income_annual','land_holding_acres','family_size']

X = df[FEATURE_COLS].copy()   # Features (inputs)
y = df[LABEL_COLS].copy()     # Labels  (outputs)
# .copy() ensures we don't accidentally modify the original df

# ── Encode categorical columns ─────────────────────────────────
encoders = {}   # Dictionary to store one encoder per column

for col in CATEGORICAL:
    le = LabelEncoder()
    # LabelEncoder converts text → numbers
    # e.g. ['Male','Female','Other'] → [1, 0, 2]

    X[col] = le.fit_transform(X[col])
    # fit_transform: learns the mapping AND applies it in one step

    encoders[col] = le
    # Save encoder so we can use it later on new user inputs

# ── Scale numerical columns ────────────────────────────────────
scaler = MinMaxScaler()
# MinMaxScaler squishes values to 0–1 range
# Formula: (x - min) / (max - min)
# Helps models treat all features equally (income vs age)

X[NUMERICAL] = scaler.fit_transform(X[NUMERICAL])
# Apply scaling only to numerical columns

# ── Save preprocessed data + tools ────────────────────────────
os.makedirs('models', exist_ok=True)

X.to_csv('data/X_processed.csv', index=False)
y.to_csv('data/y_labels.csv',    index=False)

with open('models/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
# pickle.dump saves Python objects to a binary file
# 'wb' = write binary mode

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Preprocessing complete. Files saved.")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")