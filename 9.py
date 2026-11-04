# ============================================================
# 9.py — DBSCAN (Density-Based Spatial Clustering)
#         Clustering the Iris dataset with noise detection
# ============================================================

# --- Import libraries ---
import matplotlib.pyplot as plt                            # For creating plots
from sklearn.datasets import load_iris                     # Iris flower dataset
from sklearn.preprocessing import StandardScaler           # For feature normalization
from sklearn.cluster import DBSCAN                         # Density-Based Spatial Clustering of Applications with Noise
from sklearn.decomposition import PCA                      # For dimensionality reduction (visualization)

# --- Load the Iris dataset ---
# Load the 4-feature Iris dataset (150 samples)
data = load_iris()
X = data.data

# --- Scale the data ---
# Standardize features to mean=0, std=1
# Critical for DBSCAN because it relies on distance-based neighborhood calculations
X = StandardScaler().fit_transform(X)

# --- Apply DBSCAN clustering ---
# eps=0.8: maximum distance between two points to be considered neighbors
# min_samples=5: minimum number of points required to form a dense cluster
# Points that don't belong to any cluster are labeled as noise (-1)
model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X)

# --- Count clusters and noise points ---
# Get unique cluster labels, excluding -1 (noise)
clusters = set(labels) - {-1}

# Count how many points were classified as noise
noise = list(labels).count(-1)

# Display clustering summary
print("Number of clusters:", len(clusters))
print("Number of noise points:", noise)
print("Cluster labels:")
print(labels)

# --- Reduce to 2D for visualization ---
# PCA reduces the 4 features to 2 principal components for 2D plotting
X_2d = PCA(n_components=2).fit_transform(X)

# --- Plot the clusters ---
# Scatter plot colored by cluster label (noise points shown in a distinct color)
# 'viridis' colormap maps cluster labels to different colors
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering")
plt.show()
