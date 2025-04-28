# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:17:07 2025

@author: Gulraiz.Iqbal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define input trajectories with time as third dimension
tr1 = np.array([(0, 0, 0), (1, 2, 1), (2, 4, 2), (3, 5, 3), (4, 7, 4)])
tr2 = np.array([(0, 0, 0), (1, 2, 1), (2, 4, 2), (2, 5, 3), (2, 6, 4), (3, 5, 5), (4, 7, 6)])

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



# Interpolate both trajectories
tr1_interp = interpolate_space_time(tr1, M=6)
tr2_interp = interpolate_space_time(tr2, M=6)

# 3D plot of interpolated trajectories
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(tr1_interp[:, 0], tr1_interp[:, 1], tr1_interp[:, 2], 's-', label='Interpolated tr1')
ax.plot(tr2_interp[:, 0], tr2_interp[:, 1], tr2_interp[:, 2],'s-',  label='Interpolated tr2')
ax.set_title('3D Interpolation of Trajectories (X, Y, Time)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()
plt.tight_layout()
plt.show()

# Step 4: Flatten (x,y,t) into a 1D vector
flat1 = tr1_interp.flatten()
flat2 = tr2_interp.flatten()


X = np.vstack([flat1, flat2])

# Step 6: Manual K-means
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

# Step 7: Run manual K-means
labels, centroids = kmeans(X, K=2, num_iters=10)

# Step 8: Print cluster labels
print("Cluster labels:", labels)