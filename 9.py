import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
X = data.data

# Scale data
X = StandardScaler().fit_transform(X)

# Apply DBSCAN
model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X)

# Count clusters and noise
clusters = set(labels) - {-1}
noise = list(labels).count(-1)

print("Number of clusters:", len(clusters))
print("Number of noise points:", noise)
print("Cluster labels:")
print(labels)

# Reduce dimensions for visualization
X_2d = PCA(n_components=2).fit_transform(X)

# Plot clusters
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering")
plt.show()


 
