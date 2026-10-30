
"""
Week 3 Internship Project: Unsupervised Learning & Model Evaluation
Run: python main.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# -------------------- PART A: CLUSTERING --------------------
wine = load_wine()
X_cluster = pd.DataFrame(wine.data, columns=wine.feature_names)
X_scaled = StandardScaler().fit_transform(X_cluster)

# K-Means: select a useful number of clusters using silhouette scores.
silhouette_by_k = {}
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    silhouette_by_k[k] = silhouette_score(X_scaled, labels)

best_k = max(silhouette_by_k, key=silhouette_by_k.get)
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Hierarchical (agglomerative) clustering with the same cluster count.
hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_labels = hierarchical.fit_predict(X_scaled)

def clustering_metrics(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels)
    }

cluster_results = {
    "K-Means": clustering_metrics(X_scaled, kmeans_labels),
    "Hierarchical": clustering_metrics(X_scaled, hier_labels)
}

# PCA reduces the 13 original wine features to 2 components for visualization.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=35)
plt.title(f"Wine Dataset: K-Means Clusters (k={best_k}) shown using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("outputs/kmeans_pca_clusters.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(list(silhouette_by_k.keys()), list(silhouette_by_k.values()), marker="o")
plt.title("K-Means Silhouette Score for Different k Values")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.xticks(list(silhouette_by_k.keys()))
plt.tight_layout()
plt.savefig("outputs/kmeans_silhouette_scores.png", dpi=160)
plt.close()

# -------------------- PART B: MODEL EVALUATION --------------------
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cv_f1 = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cancer.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=160)
plt.close()

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Logistic Regression")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=160)
plt.close()

# Save readable metrics for the report and future review.
with open("outputs/results.txt", "w", encoding="utf-8") as f:
    f.write("WEEK 3: UNSUPERVISED LEARNING & MODEL EVALUATION\n\n")
    f.write(f"Best K-Means k: {best_k}\n")
    f.write(f"Silhouette scores by k: {silhouette_by_k}\n\n")
    f.write("Clustering metrics (higher silhouette and Calinski-Harabasz are generally better; lower Davies-Bouldin is better):\n")
    for name, values in cluster_results.items():
        f.write(f"{name}: {values}\n")
    f.write("\nClassification: Breast Cancer Wisconsin dataset\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best cross-validation F1: {grid.best_score_:.4f}\n")
    f.write(f"5-fold CV F1 scores: {cv_f1}\n")
    f.write(f"Mean 5-fold CV F1: {cv_f1.mean():.4f}\n")
    f.write(f"Test precision: {precision:.4f}\n")
    f.write(f"Test recall: {recall:.4f}\n")
    f.write(f"Test F1-score: {f1:.4f}\n")
    f.write(f"Test ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Confusion matrix:\n{cm}\n\n")
    f.write("Classification report:\n")
    f.write(classification_report(y_test, y_pred, target_names=cancer.target_names))

print("Project completed. See outputs/ for charts and results.txt.")
print(f"Best K-Means k: {best_k}")
print("Clustering metrics:", cluster_results)
print("Best classification parameters:", grid.best_params_)
print(f"CV F1: {cv_f1.mean():.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
print("Confusion matrix:\n", cm)
