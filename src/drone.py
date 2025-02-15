import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from skimage.draw import polygon

class Drone:
    def __init__(self, path, velocity=1.0, camera_elevation=0.0, camera_azimuth=0.0, camera_fov=np.deg2rad(90)):
        """
        Initialize the Drone.

        Parameters:
        - path: list of numpy arrays representing points ([(x,y,z), (x,y,z), ...]).
        - velocity: constant speed per move call (distance moved per move call).
        - camera_elevation: initial elevation angle of the camera in radians. 0.0 means the camera is level
        - camera_azimuth: initial azimuth angle of the camera in radians. 0.0 means the camera is looking ahead
        - camera_fov: initial horizontal field of view in radians
        """
        self.path = path
        self.velocity = velocity
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.camera_fov = camera_fov 
        
        self.current_path_idx = 0
        self.position = path[self.current_path_idx]
        self.direction = path[self.current_path_idx+1] - self.position
        self.distance_to_target = np.linalg.norm(self.direction)
        self.direction /= self.distance_to_target

        def __str__(self):
            return (f"Drone(position={self.position}, "
                    f"camera_elevation={self.camera_elevation}, "
                    f"camera_azimuth={self.camera_azimuth})")

    def move(self, dt):
        """
        Move the drone forward by a distance equal to its velocity toward the next point.
        If the drone reaches a point, it continues to the subsequent point.

        parameters:
        - dt: duration of a single time step
        """

        if self.current_path_idx == len(self.path) - 1: return
        
        # If already at the target, update the index.
        if (np.all(self.position == self.path[self.current_path_idx + 1])):
            self.current_path_idx += 1
            self.direction = path[self.current_path_idx+1] - self.position
            self.distance_to_target = np.linalg.norm(self.direction)
            self.direction /= self.distance_to_target

        # update position
        self.distance_to_target -= self.velocity * dt
        next_point = self.path[self.current_path_idx + 1]
        if self.distance_to_target < 0: # we overshot the target
            self.position = self.path[self.current_path_idx + 1]
        else:
            self.position = next_point + self.distance_to_target * (-self.direction)


    def adjust_camera(self, elevation=None, azimuth=None, fov=None):
        """
        Adjust the camera's elevation and azimuth angles.
        """
        if elevation is not None: self.camera_elevation = elevation
        if azimuth is not None: self.camera_azimuth = azimuth
        if fov is not None: self.camera_fov = fov


    def view_coverage(self): 
        """
        calculate current view coverage of the camera
        """

        # Convert FOV from degrees to radians and compute half-angle.
        fov_vertical = self.camera_fov
        fov_horizontal = self.camera_fov 

        tan_half_horizontal = np.tan(fov_horizontal / 2)
        tan_half_vertical = np.tan(fov_vertical / 2)

        # define corners in camera space i.e. with the camera looking in the positive y direction
        # camera corners have distance 1 from origin (drone)
        camera_corners = np.array([ # x,y,z
            [-tan_half_horizontal, 1, -tan_half_vertical],  # bottom-left
            [ tan_half_horizontal, 1, -tan_half_vertical],  # bottom-right
            [ tan_half_horizontal, 1,  tan_half_vertical],  # top-right
            [-tan_half_horizontal, 1,  tan_half_vertical]   # top-left
        ])

        # Rotate the camera in camera space 
        R_z = np.array([
            [np.cos(self.camera_azimuth), -np.sin(self.camera_azimuth), 0],
            [np.sin(self.camera_azimuth),  np.cos(self.camera_azimuth), 0],
            [0, 0, 1]
        ])
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(-self.camera_elevation), -np.sin(-self.camera_elevation)],
            [0, np.sin(-self.camera_elevation),  np.cos(-self.camera_elevation)]
        ])
        cam_rotation = R_z @ R_x # first pitch then yaw
        camera_corners = (cam_rotation @ camera_corners.T)  

        # rotate the camera in world space
        theta = -np.arctan2(self.direction[0], self.direction[1])
        angle_cos = np.cos(theta)
        angle_sin = np.sin(theta)
        world_rotation = np.array([
            [angle_cos, -angle_sin, 0],
            [angle_sin,  angle_cos, 0], 
            [0, 0, 1]
        ])
        camera_rays = (world_rotation @ camera_corners).T
        
        # calcualte ray intersection with ground
        x = - self.position[2] / camera_rays[:, 2]
        camera_world_corners = camera_rays * x[:, np.newaxis] + self.position
        return camera_world_corners[:, :2]
    

    def detection_coverage(self, terrain, pixel_size, certain_detection_distance, max_detection_distance):
        """
        Calculate a detection probability for every pixel in the terrain that lies within the camera's view.
        
        For now, the detection probability p is set to be the inverse of the distance from the drone to the pixel.
        Parameters:
          terrain: 2D numpy array representing the terrain features.
          pixel_size: The size (in world units) of one pixel.
          certain_detection_distance: (Not used in the basic inverse distance model.)
          max_detection_distance: (Not used in the basic inverse distance model.)
        
        Returns:
          detection_coverage: 2D numpy array of the same shape as terrain containing the detection probabilities.
        """
      
        detection_coverage = np.zeros(terrain.shape[:2], dtype=float)

        # Convert world coordinates to pixel indices.
        view_extent = self.view_coverage()
        poly_cols = view_extent[:, 0] / pixel_size  # x -> column
        poly_rows = view_extent[:, 1] / pixel_size  # y -> row

        # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
        rr, cc = polygon(poly_rows, poly_cols, shape=terrain.shape[:2])

        xyzcoordinates = (np.array([cc, rr, np.zeros_like(rr)]).T * pixel_size)

        diffs = self.position - xyzcoordinates
        distances = np.linalg.norm(diffs, axis=1, keepdims=True)
        normalized_diffs = diffs/distances 

        def detection_confidence(distances, terrain):
            P = 1 - (distances - certain_detection_distance) / (max_detection_distance - certain_detection_distance)
            P = P.flatten() * terrain
            return np.clip(P, 0, 1)

        height_thresh = 0.7
        is_upper_quadrant = normalized_diffs[:, 0] > 0
        is_right_quadrant = normalized_diffs[:, 1] > 0
        is_top_quadrant = normalized_diffs[:, 2] > height_thresh

        masks = [
            is_upper_quadrant  & is_right_quadrant  & ~is_top_quadrant,
            is_upper_quadrant  & ~is_right_quadrant & ~is_top_quadrant,
            ~is_upper_quadrant & ~is_right_quadrant & ~is_top_quadrant,
            ~is_upper_quadrant & is_right_quadrant  & ~is_top_quadrant,
            is_upper_quadrant  & is_right_quadrant  & is_top_quadrant ,
            is_upper_quadrant  & ~is_right_quadrant & is_top_quadrant ,
            ~is_upper_quadrant & ~is_right_quadrant & is_top_quadrant ,
            ~is_upper_quadrant & is_right_quadrant  & is_top_quadrant ,
        ]

        for i in range(terrain.shape[2]):
            rrr = rr[masks[i]]
            ccc = cc[masks[i]]
            detection_coverage[rrr, ccc] = detection_confidence(distances[masks[i]], terrain[rrr, ccc, i])

        return detection_coverage
       





def plot_drone(drone, terrain, pixel_size, dt):
    """
    Create an animation of the drone's flight with updated detection coverage.

    Parameters:
      drone      : an instance of Drone.
      terrain    : 3D numpy array representing the terrain (shape: [rows, cols, channels]).
      pixel_size : scalar, size of a pixel in world units.
      dt         : time step duration for each frame.
    """
    # Set up the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the entire path (assuming drone.path is a list or array of [x,y,z] points).
    path_arr = np.array(drone.path)

    # Compute and show the initial detection coverage.
    initial_coverage = drone.detection_coverage(terrain, pixel_size, certain_detection_distance=4, max_detection_distance=10)
    # Set the extent using terrain dimensions and pixel_size.
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]
    im = ax.imshow(initial_coverage, origin='lower', extent=extent, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, label='Detection Probability')

    # Create a marker for the drone (initially empty, but we'll update its position).
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    
    # Create an arrow (using quiver) to show the drone's direction.
    arrow_scale = 0.2  # adjust as needed
    quiver = ax.quiver(drone.position[0], drone.position[1],
                       drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='k')

    # Set axis labels and title.
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_title('Drone Camera Coverage Map')
    ax.legend()
    ax.plot(path_arr[:, 0], path_arr[:, 1], 'k--', label="Path")
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        # Move the drone and update camera orientation (e.g., oscillate azimuth).
        drone.move(dt)
        drone.adjust_camera(azimuth=0.6*np.sin(frame*0.1))
        
        # Recalculate detection coverage.
        coverage = drone.detection_coverage(terrain, pixel_size, certain_detection_distance=4, max_detection_distance=10)
        im.set_data(coverage)
        
        # Update the drone marker position:
        # Provide the position as sequences/lists.
        drone_marker.set_data([drone.position[0]], [drone.position[1]])
        
        # Update the quiver arrow:
        quiver.set_offsets([drone.position[0], drone.position[1]])
        quiver.set_UVC(drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale)
        
        return im, drone_marker, quiver

    # Create and show the animation.
    ani = FuncAnimation(fig, update, frames=300, interval=dt*500, blit=True)
    plt.show()





# Example usage:
if __name__ == "__main__":
    # Define a simple 2D path.
    # path = np.array([[0,0,10], [0, 5,10], [5,7,10], [10,10,10]], dtype=float)
    path = np.array([[0,0,3], [5, 5,3], [5,7,3], [10,10,3]], dtype=float)
    
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length = np.sum(segment_lengths) 

    dt = 0.02
    drone = Drone(path, velocity=2.0, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(40), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 0.025
    terrain = np.ones(map_shape , dtype=float)
    terrain[:, :, 4:] *= 0.9
    terrain[:, :, :4] *= 0.5
    plot_drone(drone, terrain, pixel_size, dt)


    # for _ in range(300):
    #     drone.move(dt)
    #     detection_coverage = drone.detection_coverage(terrain, pixel_size, 4, 10)

    #     arrow_scale = 0.2  # Adjust arrow length as needed
    #     plt.figure(figsize=(8, 8))

    #     plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    #     plt.imshow(detection_coverage[:, :], origin='lower',
    #             extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
    #             cmap='viridis')
    #     plt.colorbar(label='Detection Probability')
    #     plt.arrow(drone.position[0], drone.position[1],
    #             drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
    #             head_width=0.2, head_length=0.1, fc='k', ec='k', width=0.05)
    #     plt.title('Drone Camera Coverage Map')
    #     plt.xlabel('World X')
    #     plt.ylabel('World Y')
    #     plt.axis('equal')
    #     plt.show()

   
