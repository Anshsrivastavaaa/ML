"""Find the optimal number of clusters in the supplied Iris dataset.

The species column is kept for reference only and is not used to fit K-Means.
The number of clusters is selected using the highest silhouette score.
"""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]


def load_iris_data() -> pd.DataFrame:
    """Load Iris.csv from the project directory or from iris.zip."""
    csv_candidates = (
        BASE_DIR / "Iris.csv",
        BASE_DIR / "iris.csv",
        BASE_DIR / "_iris_inspect" / "Iris.csv",
    )
    for csv_path in csv_candidates:
        if csv_path.exists():
            return pd.read_csv(csv_path)

    archive_path = BASE_DIR / "iris.zip"
    if archive_path.exists():
        with ZipFile(archive_path) as archive:
            csv_members = [
                member for member in archive.namelist()
                if Path(member).name.lower() == "iris.csv"
            ]
            if csv_members:
                with archive.open(csv_members[0]) as csv_file:
                    return pd.read_csv(BytesIO(csv_file.read()))

    raise FileNotFoundError(
        "Could not find Iris.csv or iris.zip next to ML 8.py."
    )


def main() -> None:
    data = load_iris_data()
    missing_columns = set(FEATURE_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

    features = data[FEATURE_COLUMNS]
    scaled_features = StandardScaler().fit_transform(features)

    cluster_counts = range(2, 11)
    inertias = []
    silhouette_scores = []
    models = {}

    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)
        models[cluster_count] = model
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, labels))

    best_index = silhouette_scores.index(max(silhouette_scores))
    optimal_clusters = list(cluster_counts)[best_index]
    best_model = models[optimal_clusters]
    data["Cluster"] = best_model.labels_

    metrics = pd.DataFrame(
        {
            "Clusters": list(cluster_counts),
            "Inertia": inertias,
            "SilhouetteScore": silhouette_scores,
        }
    )
    metrics.to_csv(BASE_DIR / "kmeans_metrics.csv", index=False)
    data.to_csv(BASE_DIR / "iris_clustered.csv", index=False)

    figure, (elbow_axis, silhouette_axis) = plt.subplots(1, 2, figsize=(12, 5))
    elbow_axis.plot(list(cluster_counts), inertias, marker="o")
    elbow_axis.set_title("Elbow Method")
    elbow_axis.set_xlabel("Number of clusters (k)")
    elbow_axis.set_ylabel("Inertia")
    elbow_axis.grid(True, alpha=0.3)

    silhouette_axis.plot(list(cluster_counts), silhouette_scores, marker="o")
    silhouette_axis.axvline(optimal_clusters, color="tab:red", linestyle="--")
    silhouette_axis.set_title("Silhouette Scores")
    silhouette_axis.set_xlabel("Number of clusters (k)")
    silhouette_axis.set_ylabel("Silhouette score")
    silhouette_axis.grid(True, alpha=0.3)

    figure.suptitle(
        f"K-Means result: optimal number of clusters = {optimal_clusters}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(BASE_DIR / "kmeans_cluster_selection.png", dpi=150)

    print(f"Optimal number of clusters: {optimal_clusters}")
    print(
        f"Best silhouette score: "
        f"{silhouette_scores[best_index]:.4f}"
    )
    print(f"Saved metrics to: {BASE_DIR / 'kmeans_metrics.csv'}")
    print(f"Saved clustered data to: {BASE_DIR / 'iris_clustered.csv'}")
    print(f"Saved plot to: {BASE_DIR / 'kmeans_cluster_selection.png'}")

    # Display the Elbow and Silhouette charts when the script is run.
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()