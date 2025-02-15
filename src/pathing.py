import numpy as np
def snake_fill(polygon:np.ndarray, num_waypoints:int):
    pass


def path_to_angles(waypoints):
    """
    Convert a sequence of 2D waypoints into a compressed representation using angles.
    Assumes waypoints are equidistant.
    
    Parameters:
        waypoints (np.ndarray): Nx2 array of (x, y) waypoints.
    
    Returns:
        angles (np.ndarray): (N-1,) array of angles in radians.
    """
    vectors = np.diff(waypoints, axis=0)  # Compute direction vectors
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])  # Compute angles
    return angles

def angles_to_path(start_point, step_size, angles):
    """
    Reconstruct a sequence of 2D waypoints from a starting point, step size, and angles.
    
    Parameters:
        start_point (tuple): (x, y) starting coordinate.
        step_size (float): Distance between consecutive waypoints.
        angles (np.ndarray): Sequence of angles in radians.
    
    Returns:
        waypoints (np.ndarray): Reconstructed Nx2 array of (x, y) waypoints.
    """
    num_points = len(angles) + 1
    waypoints = np.zeros((num_points, 2))
    waypoints[0] = start_point
    
    for i in range(1, num_points):
        waypoints[i, 0] = waypoints[i-1, 0] + step_size * np.cos(angles[i-1])
        waypoints[i, 1] = waypoints[i-1, 1] + step_size * np.sin(angles[i-1])
    
    return waypoints

def pack_constant_alt(path):
    packed = np.concat([path[:,:2].flatten(), [path[0,2]]])
    return packed

def unpack_constant_alt(x):
    path = x[:-1].reshape((-1,2))
    altitude = x[-1]
    path = np.vstack([path.T, altitude*np.ones(len(path))]).T
    return path

def pack_constant_segments(path):
    distance = np.linalg.norm(path[0,:2] - path[1,:2])
    return np.concat([path[0,:2], [distance], path_to_angles(path[:,:2]), [path[0,2]]])

def unpack_constant_segments(x):
    altitude = x[-1]
    path = angles_to_path(x[:2], x[2], x[3:-1])
    path = np.vstack([path.T, altitude*np.ones(len(path))]).T
    return path

def zigzag_fill(polygon:np.ndarray, num_waypoints:int):
    """
    Expect a (4,2) numpy array of corner point that this drone has to cover in clockwise/counter clockwise order
    """
    waypoints = np.zeros((num_waypoints,polygon.shape[1]))
    waypoints[0] = polygon[0]
  
    for i in range(num_waypoints):
        percent = i / num_waypoints
        edge1 = (1.0-percent)*polygon[0] +  percent*polygon[1]
        edge2 = (1.0-percent)*polygon[3] +  percent*polygon[2]

        waypoints[i] = edge1 if i % 2 == 0 else edge2
    return waypoints

def spiral_fill(polygon:np.ndarray, num_waypoints:int):
    return 


if __name__ == "__main__":
    polygon = np.array([[0,0],[0,10],[10,10],[10,0]])
    num_waypoints = 10
    waypoints = snake_fill(polygon, num_waypoints)
    print(waypoints)

    waypoints = zigzag_fill(polygon, num_waypoints)
    print(waypoints)

    waypoints = spiral_fill(polygon, num_waypoints)
    print(waypoints)