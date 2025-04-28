# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:28:34 2025

@author: Gulraiz.Iqbal
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Function to subsample trajectory to M equidistant points in space
def subsample_trajectory(trajectory, M=20):
    # Calculate cumulative distance along the trajectory
    cum_distances = np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    distances = np.insert(cum_distances, 0, 0)  # Insert zero for the first point
    
    # Generate target distances (equally spaced)
    target_distances = np.linspace(0, distances[-1], M)
    
    # Interpolate the trajectory to find points at target distances
    interpolator = interp1d(distances, trajectory, axis=0, kind='linear')
    return interpolator(target_distances), distances[-1]



def interpolate_with_time(traj, M=20):
    traj = np.array(traj)
    t = np.linspace(0, 1, len(traj))  # normalized time
    t_new = np.linspace(0, 1, M)
    x_interp = interp1d(t, traj[:, 0], kind='linear')(t_new)
    y_interp = interp1d(t, traj[:, 1], kind='linear')(t_new)
    return np.stack([x_interp, y_interp], axis=1)

# Original trajectories
tr1 = np.array([(0, 0), (1, 2), (2, 4), (3, 5), (4, 7)])
tr2 = np.array([(0, 0), (1, 2), (2, 4), (2, 5), (2, 6), (3, 5), (4, 7)])

# Interpolated trajectories
tr1_interp = interpolate_with_time(tr1, 20)
tr2_interp = interpolate_with_time(tr2, 20)
tr3, d1 = subsample_trajectory(tr1)

# Plotting
plt.figure(figsize=(10, 6))

# Original trajectories
#plt.plot(tr1[:, 0], tr1[:, 1], 'o--', label='Original tr1')
#plt.plot(tr2[:, 0], tr2[:, 1], 'o--', label='Original tr2')

# Interpolated trajectories
plt.plot(tr1_interp[:, 0], tr1_interp[:, 1], 's-', label='Interpolated tr1 (time-based)', alpha=0.7)
plt.plot(tr2_interp[:, 0], tr2_interp[:, 1], 's-', label='Interpolated tr2 (time-based)', alpha=0.7)
#plt.plot(tr3[:, 0], tr3[:, 1], 's-', label='Interpolated tr3', alpha=0.7)

plt.title('Trajectory Interpolation with Time Preservation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()


