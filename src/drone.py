import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from skimage.draw import polygon

epsilon = 1e-6

class Drone:
    def __init__(self, path, camera_elevation=0.0, camera_azimuth=0.0, camera_fov=np.deg2rad(90),
                 num_timesteps = 100):
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
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.camera_fov = camera_fov 
        
        self.current_path_idx = 0
        distances = np.cumsum(np.concat([[0.0], np.linalg.norm(np.diff(path, axis=0), axis=1)]))
        
        interp_points = np.linspace(0.0, distances[-1], num_timesteps)
        x_interp = np.interp(interp_points, distances, path[:,0])
        y_interp = np.interp(interp_points, distances, path[:,1])
        z_interp = np.interp(interp_points, distances, path[:,2])
        
        self.positions = np.array([x_interp, y_interp, z_interp]).T
        dirs = np.diff(self.positions, axis=0) 
        self.directions = np.concat([[dirs[0]], dirs], axis=0)
        # x_interp = np.interp(interp_points, distances, dirs[:,0])
        # y_interp = np.interp(interp_points, distances, dirs[:,1])
        # z_interp = np.interp(interp_points, distances, dirs[:,2])
        # self.directions = np.array([x_interp, y_interp, z_interp]).T
        # print(self.positions)
        # print(self.directions)
        self.distances = distances

        def __str__(self):
            return (f"Drone(position={self.positions}, "
                    f"camera_elevation={self.camera_elevation}, "
                    f"camera_azimuth={self.camera_azimuth})")


    def adjust_camera(self, elevation=None, azimuth=None, fov=None):
        """
        Adjust the camera's elevation and azimuth angles.
        """
        if elevation is not None: self.camera_elevation = elevation
        if azimuth is not None: self.camera_azimuth = azimuth
        if fov is not None: self.camera_fov = fov


    def view_coverage(self, timestep_idx): 
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

        # clamp the z coordinate
        camera_corners[2, camera_corners[2] > -epsilon] = -epsilon 

        # rotate the camera in world space
        direc = self.directions[timestep_idx] / np.linalg.norm(self.directions[timestep_idx])
        theta = -np.arctan2(direc[0], direc[1])
        angle_cos = np.cos(theta)
        angle_sin = np.sin(theta)
        world_rotation = np.array([
            [angle_cos, -angle_sin, 0],
            [angle_sin,  angle_cos, 0], 
            [0, 0, 1]
        ])
        camera_rays = (world_rotation @ camera_corners).T
        
        # calcualte ray intersection with ground
        pos = self.positions[timestep_idx]
        x = - pos[2] / camera_rays[:, 2]
        camera_world_corners = camera_rays * x[:, np.newaxis] + pos
        return camera_world_corners[:, :2]
    

    def detection_coverage(self, timestep_idx, enviornment_map, pixel_size, certain_detection_distance, max_detection_distance):
        view_extent = self.view_coverage(timestep_idx)
        detection_coverage = np.zeros(enviornment_map.shape, dtype=float)

        # Convert world coordinates to pixel indices.
        poly_cols = view_extent[:, 0] / pixel_size  # x -> column
        poly_rows = view_extent[:, 1] / pixel_size  # y -> row

        # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
        rr, cc = polygon(poly_rows, poly_cols, shape=enviornment_map.shape[:2])
        xyzcoordinates = (np.array([cc, rr, np.zeros_like(rr)]).T * pixel_size)
        distances = np.linalg.norm(xyzcoordinates - self.positions[timestep_idx], axis=1)
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

   
