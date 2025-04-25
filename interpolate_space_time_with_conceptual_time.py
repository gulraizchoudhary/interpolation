# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:39:49 2025

@author: Gulraiz.Iqbal
"""

# Re-import necessary libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D



def interpolate_by_distance_3d(traj, M=20):
    traj = np.array(traj)
    diffs = np.diff(traj, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumdist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumdist[-1]

    # Uniform distance steps along the path
    dist_new = np.linspace(0, total_length, M)
    interp = interp1d(cumdist, traj, axis=0, kind='linear')
    interpolated_xy = interp(dist_new)

    # Add normalized time based on spatial progression
    time_z = np.linspace(0, 1, M).reshape(-1, 1)
    return np.hstack((interpolated_xy, time_z))

# Interpolation function with time as 3rd dimension
def interpolate_with_time_3d(traj, M=20):
    traj = np.array(traj)
    t = np.linspace(0, 1, len(traj))  # normalized time
    t_new = np.linspace(0, 1, M)
    x_interp = interp1d(t, traj[:, 0], kind='linear')(t_new)
    y_interp = interp1d(t, traj[:, 1], kind='linear')(t_new)
    return np.stack([x_interp, y_interp, t_new], axis=1)  # Add time as 3rd dimension



# Re-define original trajectories
tr1 = np.array([(0, 0), (1, 2), (2, 4), (3, 5), (4, 7)])
tr2 = np.array([(0, 0), (1, 2), (2, 4), (2, 5), (2, 6), (3, 5), (4, 7)])

# Interpolated 3D trajectories (with time)
tr1_interp_3d = interpolate_with_time_3d(tr1, M=20)
tr2_interp_3d = interpolate_with_time_3d(tr2, M=20)

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot original trajectories in 3D (using original time)
#ax.plot(tr1[:, 0], tr1[:, 1], np.linspace(0, 1, len(tr1)), 'o--', label='Original tr1')
#ax.plot(tr2[:, 0], tr2[:, 1], np.linspace(0, 1, len(tr2)), 'o--', label='Original tr2')

# Plot interpolated 3D trajectories
ax.plot(tr1_interp_3d[:, 0], tr1_interp_3d[:, 1], tr1_interp_3d[:, 2], 's-', label='Interpolated tr1 (3D)')
ax.plot(tr2_interp_3d[:, 0], tr2_interp_3d[:, 1], tr2_interp_3d[:, 2], 's-', label='Interpolated tr2 (3D)')

ax.set_title('3D Trajectories (with Time as Z)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
ax.legend()
plt.tight_layout()
plt.show()




# Apply equidistant interpolation
tr1_dist_3d = interpolate_by_distance_3d(tr1, M=20)
tr2_dist_3d = interpolate_by_distance_3d(tr2, M=20)

# 3D Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot equidistantly interpolated trajectories
ax.plot(tr1_dist_3d[:, 0], tr1_dist_3d[:, 1], tr1_dist_3d[:, 2], 's-', label='tr1 (equi-distance)')
ax.plot(tr2_dist_3d[:, 0], tr2_dist_3d[:, 1], tr2_dist_3d[:, 2], 's-', label='tr2 (equi-distance)')

ax.set_title('3D Trajectories (Equidistant in Space, Normalized Time)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Normalized Time')
ax.legend()
plt.tight_layout()
plt.show()
