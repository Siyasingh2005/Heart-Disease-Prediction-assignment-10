"""
generate_dataset.py
--------------------
Generates heart.csv in the same schema as the Kaggle "Heart Disease Dataset"
(johnsmith88/heart-disease-dataset), which is the standard 14-column
UCI Cleveland Heart Disease dataset:

age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
slope, ca, thal, target

NOTE: This environment has no internet access, so the real Kaggle file
could not be downloaded directly. This script builds a large synthetic
dataset that follows the real dataset's column definitions, value ranges,
and realistic clinical correlations (e.g. higher age/cholesterol/oldpeak
and exercise-induced angina increase heart-disease risk). Before final
submission, replace heart.csv with the actual file downloaded from Kaggle
(same column names are used here, so no other code needs to change).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1025  # matches size of the popular Kaggle version of this dataset

age = np.random.randint(29, 78, N)
sex = np.random.randint(0, 2, N)  # 1 = male, 0 = female
cp = np.random.randint(0, 4, N)   # chest pain type (0-3)
trestbps = np.random.randint(94, 201, N)   # resting blood pressure
chol = np.random.randint(126, 565, N)      # serum cholesterol
fbs = np.random.binomial(1, 0.15, N)       # fasting blood sugar > 120 mg/dl
restecg = np.random.randint(0, 3, N)       # resting ECG results
thalach = np.random.randint(71, 203, N)    # max heart rate achieved
exang = np.random.binomial(1, 0.33, N)     # exercise induced angina
oldpeak = np.round(np.random.exponential(1.0, N).clip(0, 6.2), 1)  # ST depression
slope = np.random.randint(0, 3, N)         # slope of peak exercise ST segment
ca = np.random.randint(0, 4, N)            # number of major vessels colored
thal = np.random.randint(0, 3, N)          # thalassemia

# Build target with realistic clinical signal + noise
risk_score = (
    0.03 * (age - 50)
    + 0.9 * sex
    + 0.5 * (cp == 2).astype(int)
    + 0.01 * (chol - 240)
    + 0.02 * (trestbps - 130)
    - 0.02 * (thalach - 150)
    + 1.1 * exang
    + 0.5 * oldpeak
    + 0.6 * ca
    + 0.4 * (thal == 2).astype(int)
    + np.random.normal(0, 1.2, N)
)
target = (risk_score > np.median(risk_score)).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
})

df.to_csv("heart.csv", index=False)
print(f"heart.csv created with shape {df.shape}")
print(df["target"].value_counts())
