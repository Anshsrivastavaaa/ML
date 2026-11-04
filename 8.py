# ============================================================
# 8.py — K-Means Clustering with Elbow Method
#         Finding optimal clusters in the Iris dataset
# ============================================================

# --- Import libraries ---
import matplotlib.pyplot as plt                            # For creating plots
from sklearn.datasets import load_iris                     # Iris flower dataset
from sklearn.preprocessing import StandardScaler           # For feature normalization
from sklearn.cluster import KMeans                         # K-Means clustering algorithm

# --- Load the Iris dataset ---
# Load features only (no target labels needed — clustering is unsupervised)
data = load_iris()
X = data.data

# --- Scale the data ---
# Standardize features to mean=0, std=1
# Important for K-Means because it uses Euclidean distance (sensitive to feature scale)
X = StandardScaler().fit_transform(X)

# --- Elbow Method: find the optimal number of clusters ---
# WCSS (Within-Cluster Sum of Squares) = total distance of points from their cluster center
# As K increases, WCSS decreases; the "elbow" point suggests the optimal K
wcss = []

for k in range(1, 11):
    # Create K-Means model with k clusters
    # n_init=10: run the algorithm 10 times with different random centers, pick the best
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    # model.inertia_ is the WCSS value for this K
    wcss.append(model.inertia_)

# --- Plot the Elbow curve ---
# The x-axis shows K values (1–10), y-axis shows WCSS
# Look for the "elbow" — the point where WCSS stops dropping sharply
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# --- Apply K-Means with the optimal K ---
# Based on the elbow plot, K=3 is chosen (Iris has 3 natural species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# fit_predict: fits the model and returns the cluster label for each data point
clusters = kmeans.fit_predict(X)

# Display the cluster assignment for each sample (0, 1, or 2)
print("Cluster labels:")
print(clusters)

# Display the coordinates of the 3 cluster centers in the feature space
print("Cluster centers:")
print(kmeans.cluster_centers_)
