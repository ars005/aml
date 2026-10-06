# Perform Data Loading, Feature Selection (Principal Component Analysis) and Feature Scoring and Ranking.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

breast = load_breast_cancer()

X = breast.data
y = breast.target

print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])


# --------------------------------------------------
# 2. Standardize the Data
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# 3. Apply PCA
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Completed")
print("PCA Shape:", X_pca.shape)


# --------------------------------------------------
# 4. Feature Scoring and Ranking
# --------------------------------------------------

scores = np.abs(pca.components_).sum(axis=0)

feature_names = breast.feature_names

feature_ranking = pd.DataFrame({
    "Feature": feature_names,
    "Score": scores
})

feature_ranking = feature_ranking.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Ranking:")
print(feature_ranking)


# --------------------------------------------------
# 5. Explained Variance
# --------------------------------------------------

print("\nExplained Variance:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal Variance Explained:",
    sum(pca.explained_variance_ratio_) * 100,
    "%"
)


# --------------------------------------------------
# 6. PCA Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Malignant"
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Benign"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Breast Cancer Dataset")

plt.legend()
plt.show()