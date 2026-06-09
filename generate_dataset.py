# ============================================================
# FILE: src/generate_dataset.py
# PURPOSE: Generate a realistic synthetic dataset
# ============================================================

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 5000  # Number of records

states = ['Maharashtra','Uttar Pradesh','Bihar','Rajasthan',
          'Madhya Pradesh','Gujarat','West Bengal','Karnataka',
          'Tamil Nadu','Andhra Pradesh','Punjab','Haryana']
occupations   = ['Farmer','Student','Salaried','Business','Unemployed']
education_lvl = ['Illiterate','Primary','Secondary','Graduate','Postgraduate']
categories    = ['General','OBC','SC','ST']
genders       = ['Male','Female','Other']

# ── Generate features ──────────────────────────────────────────
age        = np.random.randint(18, 80, N)
# randint(18,80,N) → N random integers from 18 to 79

gender     = np.random.choice(genders, N, p=[0.50, 0.48, 0.02])
# choice(list, N, p) → picks N items with given probabilities

income     = np.random.randint(30000, 1000000, N)
# Annual income in ₹

occupation = np.random.choice(occupations, N, p=[0.35,0.20,0.25,0.12,0.08])
education  = np.random.choice(education_lvl, N, p=[0.15,0.25,0.30,0.20,0.10])
category   = np.random.choice(categories, N, p=[0.30,0.40,0.20,0.10])
state      = np.random.choice(states, N)

occ = np.array(occupation)  # Convert to numpy for fast boolean indexing
cat = np.array(category)

is_farmer = np.where(occ == 'Farmer', 1,
                     np.random.choice([0,1], N, p=[0.85,0.15]))
# np.where(condition, if_true, if_false)

has_disability = np.random.choice([0,1], N, p=[0.93,0.07])

land_holding = np.where(is_farmer==1,
                        np.random.uniform(0.5,10,N),
                        np.random.uniform(0,1,N)).round(2)
# uniform(low,high,N) → continuous decimal values

has_bpl_card = np.where(income < 120000,
                        np.random.choice([0,1], N, p=[0.2,0.8]),
                        np.random.choice([0,1], N, p=[0.9,0.1]))

family_size = np.random.randint(1, 10, N)

# ── Scheme eligibility rules ───────────────────────────────────
pm_kisan      = ((is_farmer==1)&(income<200000)&(land_holding<5)).astype(int)
pmay          = ((has_bpl_card==1)&(income<300000)).astype(int)
ayushman      = ((has_bpl_card==1)&(income<500000)).astype(int)
pm_scholarship= ((occ=='Student')&(income<250000)).astype(int)
kisan_credit  = ((is_farmer==1)&(land_holding>0)).astype(int)
pmegp         = (((occ=='Unemployed')|(occ=='Business'))&(age>=18)&(age<=45)).astype(int)
disability_pen= (has_disability==1).astype(int)
minority_sch  = ((cat!='General')&(occ=='Student')).astype(int)

# ── Add 5% noise to simulate real-world imperfection ──────────
def add_noise(arr, rate=0.05):
    # Flip ~5% of labels randomly
    noisy = arr.copy()
    idx   = np.random.choice(len(arr), int(len(arr)*rate), replace=False)
    noisy[idx] = 1 - noisy[idx]  # 0→1 or 1→0
    return noisy

pm_kisan       = add_noise(pm_kisan)
pmay           = add_noise(pmay)
ayushman       = add_noise(ayushman)
pm_scholarship = add_noise(pm_scholarship)
kisan_credit   = add_noise(kisan_credit)
pmegp          = add_noise(pmegp)
disability_pen = add_noise(disability_pen)
minority_sch   = add_noise(minority_sch)

# ── Build DataFrame ────────────────────────────────────────────
df = pd.DataFrame({
    'age':age, 'gender':gender, 'income_annual':income,
    'occupation':occupation, 'education':education,
    'category':category, 'state':state,
    'is_farmer':is_farmer, 'has_disability':has_disability,
    'land_holding_acres':land_holding, 'has_bpl_card':has_bpl_card,
    'family_size':family_size,
    # Labels
    'PM_KISAN':pm_kisan, 'PMAY':pmay,
    'Ayushman_Bharat':ayushman, 'PM_Scholarship':pm_scholarship,
    'Kisan_Credit_Card':kisan_credit, 'PMEGP':pmegp,
    'Disability_Pension':disability_pen,
    'Minority_Scholarship':minority_sch
})

os.makedirs('data', exist_ok=True)
# exist_ok=True → no error if folder already exists

df.to_csv('data/government_schemes_dataset.csv', index=False)
# index=False → don't write row numbers

print(f"✅ Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))