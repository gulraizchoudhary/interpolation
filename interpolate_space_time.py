# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:17:07 2025

@author: Gulraiz.Iqbal
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def interpolate_space_time(traj, M=20):
    traj = np.array(traj)
    
    # Calculate cumulative spatial arc length (x, y only)
    spatial_diffs = np.diff(traj[:, :2], axis=0)
    spatial_dists = np.linalg.norm(spatial_diffs, axis=1)
    arc_lengths = np.insert(np.cumsum(spatial_dists), 0, 0)

    # Uniform arc lengths
    uniform_arc = np.linspace(0, arc_lengths[-1], M)

    # Interpolate x, y, t over uniform arc length
    x_interp = interp1d(arc_lengths, traj[:, 0],  kind='linear')(uniform_arc)
    y_interp = interp1d(arc_lengths, traj[:, 1],  kind='linear')(uniform_arc)
    t_interp = interp1d(arc_lengths, traj[:, 2],  kind='linear')(uniform_arc)

    return np.stack([x_interp, y_interp, t_interp], axis=1)


def kmeans(X, K=2, num_iters=10):
    np.random.seed(0)
    indices = np.random.choice(len(X), K, replace=False)
    centroids = X[indices]

    for _ in range(num_iters):
        # Assign points to the closest centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        for k in range(K):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)

    return labels, centroids

def normalize_trajectories(trajectories):
    normalized_trajectories = []
    for traj in trajectories:
        min_vals = traj.min(axis=0)
        max_vals = traj.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        norm_traj = (traj - min_vals) / range_vals
        normalized_trajectories.append(norm_traj)
    return normalized_trajectories

# Helper: generate a wavy or random trajectory
def generate_random_trajectory(length=5, noise_level=0.3):
    t = np.linspace(0, length, num=length+1)  # time steps [0, 1, ..., length]
    x = np.cumsum(np.random.randn(length+1))  # random walk in x
    y = np.cumsum(np.random.randn(length+1))  # random walk in y
    # Add small noise
    x += noise_level * np.random.randn(length+1)
    y += noise_level * np.random.randn(length+1)
    return np.stack([x, y, t], axis=1)

# Define some fixed (original) trajectories
tr1 = np.array([(0, 0, 0), (1, 2, 1), (2, 4, 2), (3, 5, 3), (4, 7, 4)])
tr2 = np.array([(0, 0, 0), (1, 2, 1), (2, 4, 2), (2, 5, 3), (2, 6, 4), (3, 5, 5), (4, 7, 6)])

# Generate synthetic more diverse trajectories
np.random.seed(1)  # for reproducibility
tr3 = generate_random_trajectory(length=6, noise_level=0.2)
tr4 = generate_random_trajectory(length=7, noise_level=0.2)
tr5 = generate_random_trajectory(length=9, noise_level=0.4)
tr6 = generate_random_trajectory(length=8, noise_level=0.3)
tr7 = generate_random_trajectory(length=10, noise_level=0.5)

# Collect all trajectories
trajectories = normalize_trajectories([tr1, tr2, tr3, tr4, tr5, tr6, tr7])

# === 3D Plot ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 7 colors
labels = [f"tr{i+1}" for i in range(len(trajectories))]

for traj_interp, color, label in zip(trajectories, colors, labels):
    ax.plot(traj_interp[:, 0], traj_interp[:, 1], traj_interp[:, 2], 'o-', color=color, label=label)

ax.set_title('3D Trajectories (X, Y, Time)', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()
plt.tight_layout()
plt.show()


M=6
flattened_trajectories = []
interpolated_trajectories = []  # Save interpolated ones separately for plotting

for traj in trajectories:
    traj_interp = interpolate_space_time(traj, M=M)
    interpolated_trajectories.append(traj_interp)
    flattened = traj_interp.flatten()
    flattened_trajectories.append(flattened)

# Stack into single dataset
X = np.stack(flattened_trajectories)

# === 3D Plot ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 7 colors
labels = [f"tr{i+1}" for i in range(len(flattened_trajectories))]

for traj_interp, color, label in zip(interpolated_trajectories, colors, labels):
    ax.plot(traj_interp[:, 0], traj_interp[:, 1], traj_interp[:, 2], 'o-', color=color, label=label)

ax.set_title('3D Interpolation of Trajectories (X, Y, Time)', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()
plt.tight_layout()
plt.show()


# Run K-means
labels, centroids = kmeans(X, K=4, num_iters=10)

# Print cluster labels
print("Cluster labels:", labels)

centroids_reshape = []

for centroid in centroids:
   centroids_reshape.append(centroid.reshape(-1,3))


# === 3D Plot ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

c_labels = [f"c{i+1}" for i in range(len(centroids_reshape))]
for cent, color, label in zip(centroids_reshape, colors, c_labels):
    ax.plot(cent[:, 0], cent[:, 1], cent[:, 2], 'o-', color=color, label=label)

ax.set_title('Centroids (X, Y, Time)', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()
plt.tight_layout()
plt.show()


# Make sure your cluster labels and colors are consistent
unique_clusters = np.unique(labels)
# Use your predefined colors list (7 colors in your code)
# but only pick as many colors as clusters
cluster_colors = colors[:len(unique_clusters)]

# 3D plot for trajectories, colored by cluster
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for traj_idx, traj in enumerate(interpolated_trajectories):
    cluster_id = labels[traj_idx]  # cluster index of this trajectory
    color = cluster_colors[cluster_id]
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'o-', color=color, label=f"Trajectory {traj_idx+1} (Cluster {cluster_id+1})")

# Plot centroids with same color palette, for reference
for cent, color, cluster_id in zip(centroids_reshape, cluster_colors, range(len(centroids_reshape))):
    ax.plot(cent[:, 0], cent[:, 1], cent[:, 2], 's--', color=color, label=f"Centroid Cluster {cluster_id+1}")

ax.set_title('3D Trajectories Colored by Cluster', fontsize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
plt.tight_layout()
plt.show()


