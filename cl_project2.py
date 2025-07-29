import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate 2D points for 4 parabolas
x = np.linspace(-10, 10, 200)
parabola1 = np.column_stack((x, x**2))
parabola2 = np.column_stack((x, -x**2 + 100))
parabola3 = np.column_stack((x, 0.5 * x**2))
parabola4 = np.column_stack((x, -0.5 * x**2 + 50))

# Generate 2D points for 2 circles
theta = np.linspace(0, 2 * np.pi, 200)
circle1 = np.column_stack((10 * np.cos(theta) + 30, 10 * np.sin(theta) + 30))
circle2 = np.column_stack((7 * np.cos(theta) - 30, 7 * np.sin(theta) - 30))

# Combine all data
all_shapes = np.vstack([parabola1, parabola2, parabola3, parabola4, circle1, circle2])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_shapes)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Plotting
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap("tab10", len(unique_labels))

for cluster_id in unique_labels:
    cluster_data = X_scaled[labels == cluster_id]
    label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=label, s=20, alpha=0.7)

plt.title("DBSCAN Clustering on Parabolas and Circles")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
