import numpy as np

def scale_linear_bycolumn(matrix, high=1.0, low=0.0):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - matrix)) / rng)