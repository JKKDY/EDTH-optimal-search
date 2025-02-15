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

    def move(self, dt, velocity=None):
        """
        Move the drone forward by a distance equal to its velocity toward the next point.
        If the drone reaches a point, it continues to the subsequent point.

        parameters:
        - dt: duration of a single time step
        - velocity: (optional) updated velocity of the drone
        """

        if self.current_path_idx == len(self.path) - 1: return
        if velocity is not None: self.velocity = velocity 

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
    

    def detection_coverage(self, enviornment_map, pixel_size, certain_detection_distance, max_detection_distance):
        view_extent = self.view_coverage()
        detection_coverage = np.zeros(enviornment_map.shape, dtype=float)
        print(view_extent)

        # Convert world coordinates to pixel indices.
        poly_cols = view_extent[:, 0] / pixel_size  # x -> column
        poly_rows = view_extent[:, 1] / pixel_size  # y -> row

        # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
        rr, cc = polygon(poly_rows, poly_cols, shape=enviornment_map.shape)
        xyzcoordinates = (np.array([cc, rr, np.zeros_like(rr)]).T * pixel_size)
        distances = np.linalg.norm(xyzcoordinates - self.position, axis=1)
        detection_coverage[rr, cc] = np.clip(1 - (distances - certain_detection_distance) / (max_detection_distance - certain_detection_distance), 0, 1)

        return detection_coverage




       




def plot_drone(drone, dt):
    # Set up the 2D plot (bird's-eye view) showing only x and y.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the full path (x and y only).
    ax.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    
    # Create a marker for the drone.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")

    # Create a polygon patch for the camera view corners.
    polygon_patch = Polygon(np.zeros((4, 2)), closed=True, color='cyan', ec='b', alpha=0.5)
    ax.add_patch(polygon_patch)
    
    # Set the limits of the plot.
    ax.set_xlim(np.min(path[:, 0]) - 5, np.max(path[:, 0]) + 5)
    ax.set_ylim(np.min(path[:, 1]) - 5, np.max(path[:, 1]) + 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Bird's-Eye View of Drone Flight")
    ax.legend()

    ax.set_aspect('equal', adjustable='box')

    # Set the time step (seconds) for the simulation.
    def update(frame):
        if drone.current_path_idx >= len(drone.path) - 1:
            return drone_marker,
        drone.move(dt)
        drone.adjust_camera(azimuth=0.6*np.sin(frame*0.1))
        corners = drone.view_coverage()
        drone_marker.set_data([drone.position[0]], [drone.position[1]])
        polygon_patch.set_xy(corners)
        return drone_marker,polygon_patch

    # Create the animation.
    ani = FuncAnimation(fig, update, frames=300, interval=dt*500, blit=True)
    # ani.save('drone_animation.gif', fps=30)
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
    drone = Drone(path, velocity=2.0, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(40), camera_azimuth=np.deg2rad(20))

    # plot_drone(drone, dt)


    # for _ in range(100):
    #     drone.move(dt)

    corners = drone.view_coverage()
    # print(corners)
    # print(drone.position)


    map_shape = (500, 500)
    environment =  np.zeros((500, 500) , dtype=float)
    coverage = drone.detection_coverage(environment, 0.05, 4, 10)
    plt.imshow(coverage, origin='lower')

    # coverage = drone.view_coverage(environment)
    arrow_scale = 0.2  # Adjust arrow length as needed
    pixel_size = 0.02
    plt.figure(figsize=(8, 8))

    x, y = corners[:, 0], corners[:, 1]
    plt.fill(x, y, color='cyan', edgecolor='b', alpha=0.5)

    plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    # plt.imshow(coverage, origin='lower',
    #            extent=[-1, map_shape[1]*pixel_size+1, -1, map_shape[0]*pixel_size+1],
    #            cmap='viridis')
    # plt.colorbar(label='Detection Probability')
    plt.arrow(drone.position[0], drone.position[1],
            drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
            head_width=0.2, head_length=0.1, fc='k', ec='k', width=0.05)
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X')
    plt.ylabel('World Y')
    plt.axis('equal')
    plt.show()

   
