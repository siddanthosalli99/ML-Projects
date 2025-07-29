import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generate clustered data
X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=42)
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

# Add outliers
outliers = np.random.uniform(-10, 10, size=(20, 2))
df = pd.concat([df, pd.DataFrame(outliers, columns=["Feature1", "Feature2"])], ignore_index=True)

# Scale features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(scaled_data)
unique_labels = np.unique(labels)

# Plot before and after clustering
plt.figure(figsize=(12, 5))

# --- Before Preprocessing ---
plt.subplot(1, 2, 1)
plt.scatter(df["Feature1"], df["Feature2"], c='blue', alpha=0.6)
plt.title("Before Preprocessing")
plt.xlabel("Feature1")
plt.ylabel("Feature2")

# --- After DBSCAN Clustering with Legend ---
plt.subplot(1, 2, 2)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    cluster_data = scaled_data[labels == label]
    cluster_label = f"Cluster {label + 1}" if label != -1 else "Noise"
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=cluster_label, alpha=0.7, color=color)

plt.title("After DBSCAN Clustering")
plt.xlabel("Feature1 (scaled)")
plt.ylabel("Feature2 (scaled)")
plt.legend()
plt.tight_layout()
plt.show()
