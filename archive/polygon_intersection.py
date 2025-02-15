import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

def simulate_camera_view(map_shape, pixel_size, drone_pos, angle, view_width, view_height, max_distance):
    """
    Simulate a drone camera view over a flat world.
    
    Parameters:
        map_shape   : tuple of ints (rows, cols) defining the size of the coverage map.
        pixel_size  : world size per pixel (assumes square pixels).
        drone_pos   : (x, y) world coordinates of the drone (assumed to be the bottom center of the view).
        angle       : camera orientation in radians (0 means pointing along the positive x-axis).
        view_width  : width of the rectangular view (in world units).
        view_height : height (depth) of the view (in world units).
        max_distance: maximum distance for detection probability computation.
        
    Returns:
        coverage: 2D NumPy array (of shape map_shape) with values in [0, 1] representing detection probability.
    """
    # Initialize the coverage map (all zeros)
    coverage = np.zeros(map_shape)
    
    # Define the camera view rectangle in the camera coordinate system.
    # Here, we assume the drone is at the bottom-center of the view.
    half_width = view_width / 2.0
    # Corners: bottom-left, bottom-right, top-right, top-left in camera coordinates
    corners_cam = np.array([
        [0, -half_width],
        [0,  half_width],
        [view_height,  half_width],
        [view_height, -half_width]
    ])
    
    # Create the 2D rotation matrix for the given angle.
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Transform the rectangle corners into world coordinates:
    # world_point = drone_pos + R @ camera_point
    corners_world = (R @ corners_cam.T).T + np.array(drone_pos)
    
    # Convert world coordinates to pixel indices.
    # (Assume the world origin (0,0) aligns with the lower-left of the coverage map.)
    poly_cols = corners_world[:, 0] / pixel_size  # x -> column
    poly_rows = corners_world[:, 1] / pixel_size  # y -> row

    # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
    rr, cc = polygon(poly_rows, poly_cols, shape=map_shape)
    
    # For each pixel in the polygon, compute its world coordinate (center of the pixel).
    xs = cc * pixel_size + pixel_size / 2.0
    ys = rr * pixel_size + pixel_size / 2.0
    
    # Transform these world coordinates into the camera coordinate system.
    # Subtract the drone position and apply a rotation by -angle.
    coords = np.vstack([xs - drone_pos[0], ys - drone_pos[1]])
    # Rotation by -angle:
    R_inv = np.array([
        [np.cos(angle),  np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    cam_coords = R_inv @ coords
    x_cam = cam_coords[0]  # distance along the camera's view direction
    
    # Compute detection probability as a function of distance.
    # For example, using a linear falloff:
    #   p = 1 if x_cam == 0 and falls off to 0 at max_distance.
    p = np.clip(1 - (x_cam / max_distance), 0, 1)
    
    # (Optional) You might want to ignore pixels behind the camera:
    p[x_cam < 0] = 0
    
    # Write the computed probabilities back into the coverage map.
    coverage[rr, cc] = p
    
    return coverage

# Example usage:
if __name__ == '__main__':
    # Define simulation parameters.
    map_shape   = (500, 500)      # Coverage map of 500x500 pixels.
    pixel_size  = 0.1             # Each pixel is 0.1 world units.
    drone_pos   = (25, 25)        # Drone located at (25, 25) in world coordinates.
    angle       = np.deg2rad(45)    # Camera rotated 45 degrees (in radians).
    view_width  = 10              # Camera view width (world units).
    view_height = 20              # Camera view height (world units).
    max_distance = 20             # Maximum distance for detection probability falloff.
    
    # Compute the coverage map.
    coverage = simulate_camera_view(map_shape, pixel_size, drone_pos, angle, view_width, view_height, max_distance)
    
    # Visualize the coverage map.
    plt.figure(figsize=(8, 8))
    plt.imshow(coverage, origin='lower',
               extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
               cmap='viridis')
    plt.colorbar(label='Detection Probability')
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X')
    plt.ylabel('World Y')
    plt.show()
