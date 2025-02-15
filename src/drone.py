import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from skimage.draw import polygon
from skimage.draw import disk

epsilon = 1e-6

class Drone:
    def __init__(self, path, camera_elevation=0.0, camera_azimuth=0.0, camera_fov=np.deg2rad(90),
                 num_timesteps = 100, certain_detection_distance = 1500, max_detection_distance=8000):
        """
        Initialize the Drone.

        Parameters:
        - path: list of numpy arrays representing points ([(x,y,z), (x,y,z), ...]).
        - velocity: constant speed per move call (distance moved per move call).
        - camera_elevation: initial elevation angle of the camera in radians. 0.0 means the camera is level
        - camera_azimuth: initial azimuth angle of the camera in radians. 0.0 means the camera is looking ahead
        - camera_fov: initial horizontal field of view in radians
        """
        assert certain_detection_distance < max_detection_distance


        self.path = path
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.camera_fov = camera_fov 
        self.certain_detection_distance = certain_detection_distance
        self.max_detection_distance = max_detection_distance
        
        self.num_timesteps = num_timesteps
        distances = np.cumsum(np.concat([[0.0], np.linalg.norm(np.diff(path, axis=0), axis=1)]))
        
        interp_points = np.linspace(0.0, distances[-1], self.num_timesteps)
        x_interp = np.interp(interp_points, distances, path[:,0])
        y_interp = np.interp(interp_points, distances, path[:,1])
        z_interp = np.interp(interp_points, distances, path[:,2])
        
        self.positions = np.array([x_interp, y_interp, z_interp]).T
        dirs = np.diff(self.positions, axis=0) 
        self.directions = np.concat([[dirs[0]], dirs], axis=0)

        def __str__(self):
            return (f"Drone(position={self.positions}, "
                    f"camera_elevation={self.camera_elevation}, "
                    f"camera_azimuth={self.camera_azimuth})")
        
    def position(self, step):
        return self.positions[step]


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
    

    def detection_confidence(self, distances, terrain):
        assert distances.flatten().shape == terrain.shape, f"{distances.shape} vs {terrain.shape}"
        terrain2 = 0.5 + (terrain)*0.5
        P = 1 - (distances.flatten()/terrain2 - self.certain_detection_distance) / (self.max_detection_distance - self.certain_detection_distance)
        P = P
        return np.clip(P, 0, 1)


    def detection_coverage(self, timestep_idx, terrain, pixel_size):
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
        assert terrain.shape[2] == 8

        view_extent = self.view_coverage(timestep_idx)
        
        # Convert world coordinates to pixel indices.
        poly_cols = view_extent[:, 0] / pixel_size  # x -> column
        poly_rows = view_extent[:, 1] / pixel_size  # y -> row

        # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
        rr, cc = polygon(poly_rows, poly_cols, shape=terrain.shape[:2])

        return self.evaluate_coverage(cc, rr, timestep_idx, terrain, pixel_size)
    
    def evaluate_coverage(self, cc, rr, timestep_idx, terrain, pixel_size):
        detection_coverage = np.zeros(terrain.shape)
        xyzcoordinates = (np.array([cc, rr, np.zeros_like(rr)]).T * pixel_size)

        diffs = self.positions[timestep_idx] - xyzcoordinates
        distances = np.linalg.norm(diffs, axis=1, keepdims=True)
        
        normalized_diffs = diffs/distances 

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
            detection_coverage[rrr, ccc, i] = self.detection_confidence(distances[masks[i]], terrain[rrr, ccc, i])

        return detection_coverage

    
    def detection_sphere(self, timestep_idx, terrain, pixel_size): 
        center = self.positions[timestep_idx][:2].astype(float) / pixel_size
        radius = int(self.max_detection_distance / pixel_size)

        rr, cc = disk(center, radius, shape=terrain.shape[:2])
        half_angle = 120  # in degrees
    
        # Normalize the sensor's direction vector.
        sensor_dir = self.directions[timestep_idx][:2].astype(float)
        norm = np.linalg.norm(sensor_dir)
        unit_dir = sensor_dir / norm
        
        # Note: skimage.draw.disk returns rr (rows) and cc (columns).
        # In image coordinates, (row, col) corresponds to (y, x).
        # Here we assume that self.direction is given as (x, y).
        # Compute the vector from the center to each point:
        dx = cc - center[1]  # x difference
        dy = rr - center[0]  # y difference
        
        # Compute the norm of these vectors.
        vec_norm = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero at the center.
        vec_norm[vec_norm == 0] = 1
        
        # Compute the cosine of the angle between each vector and sensor direction.
        # Dot product: (dx, dy) · (unit_dir_x, unit_dir_y)
        cos_angle = (dx * unit_dir[0] + dy * unit_dir[1]) / vec_norm
        
        # Points are within the wedge if the angle between them and sensor_dir is <= half_angle.
        # That is: angle = arccos(cos_angle) <= half_angle, or cos_angle >= cos(half_angle)
        wedge_mask = cos_angle >= np.cos(np.deg2rad(half_angle))
        
        # Apply the mask to keep only points within the 120° wedge.
        rr = rr[wedge_mask]
        cc = cc[wedge_mask]
        
        return self.evaluate_coverage(rr, cc, timestep_idx, terrain, pixel_size) 


    def total_coverage(self, terrain, pixel_size):
        observation = np.zeros(terrain.shape)
        for step in range(self.num_timesteps):
            # detectable object size at max zoom range = sin(1.5deg (fov at max zoom) ) * 8km / 1080px * 20px (minimal detection size)
            observation = np.maximum(observation, self.detection_sphere(step, terrain, pixel_size))
            # observation = np.maximum(observation, self.detection_coverage(step, terrain, pixel_size))
            
        return observation
    
    def detect_target(self, target, terrain, pixel_size):
        detection_coverage = np.zeros(terrain.shape)

        rr, cc = target 

        coverages = []
        for timestep_idx in range(1, self.num_timesteps):
            coverage = self.evaluate_coverage(np.array([cc]), np.array([rr]), timestep_idx, terrain, pixel_size)
            coverages.append(np.max(coverage[rr, cc]))

        in_region = False
        regions = []
        region_start = 0
        region_end = 0
        for i, x in enumerate(coverages):
            if in_region is False and x > 0:
                in_region = True
                region_start = i
            if in_region is True and x==0:
                in_region=False
                region_end = i
                regions.append((region_start, region_end))
        
        trials = []
        for r in regions:
            max_prob = 0
            max_idx = r[0]
            for i in range(r[0], r[1]):
                max_prob = max(max_prob, coverages[i])
                max_idx = i
            trials.append((i, max_prob))

        remaining_probabilities = []
        current_probability = 0.0
        for trial in trials:
            remaining_prob = trial[1]*(1.0 - current_probability)
            remaining_probabilities.append((trial[0], remaining_prob))
            current_probability += remaining_prob
            
        exp_value = sum([x*y for x,y in remaining_probabilities])
        return exp_value
        # print(remaining_probabilities)
        # print(exp_value)
        # plt.plot(coverages)
        # plt.show()
       








def plot_drone(drone, terrain, pixel_size):
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
        detection_coverage = np.sum(detection_coverage, axis=2)

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
    # path = np.array([[0,0,10], [0, 5,10], [5,7,10], [10,10,10]])
    # path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])
    path = np.array([[0,0,2000], [9000, 2000, 2000], [1000,4000,2000], [9000,6000,2000]])
    # path = np.array([[0,0,2000], [9000, 2000, 2000]])

    # path = np.array([[    0.         ,    0.          ,2000.        ],
    #         [  758.39689188 ,12355.15657072  ,2000.        ],
    #         [ 4297.30421253 ,12012.41540355  ,2000.        ],
    #         [ 1360.76381815 ,11456.55349753  ,2000.        ],
    #         [ 6755.7283386  , 8313.93550828  ,2000.        ],
    #         [ 6170.46952714 , 7199.88101115  ,2000.        ],
    #         [ 5738.92107382 , 5044.74214392  ,2000.        ],
    #         [ 8662.49657626 , 4878.35260546  ,2000.        ],
    #         [ 4321.28359208 , 2511.67533032  ,2000.        ],
    #         [ 6886.04182764 , 3353.1616541   ,2000.        ],
    #         [12385.89356639 , 7112.86843122  ,2000.        ],
    #         [11065.90886356 , 5036.19977054  ,2000.        ],
    #         [10214.17312873 , 3885.18706779  ,2000.        ],
    #         [ 3280.99859883 ,  672.40107673  ,2000.        ],
    #         [10222.09499517 , 2922.45958431  ,2000.        ],
    #         [ 3688.67067126 , 6667.29793658  ,2000.        ],
    #         [ 5348.96450971 , 7985.99971339  ,2000.        ],
    #         [ 4791.45512612 , 7976.31462321  ,2000.        ],
    #         [ 4586.03381588 , 3880.21916045  ,2000.        ],
    #         [ 2035.08895372 , 4724.44871408  ,2000.        ],
    #         [ 6193.47496205 , 3020.03695658  ,2000.        ],
    #         [ 6625.89944532 , 6705.56955045  ,2000.        ],
    #         [ 9141.86418116 , 8280.19290297  ,2000.        ],
    #         [ 2061.34773299 , 7639.48760113  ,2000.        ],
    #         [ 2094.63983458 , 1865.18095454  ,2000.        ],
    #         [ 3289.80512123 , 6397.08940158  ,2000.        ],
    #         [ 1663.17006065 , 2417.13863163  ,2000.        ],
    #         [ 1750.85485309 , 2463.40191922  ,2000.        ],
    #         [ 5322.25325236 , 2361.01461692  ,2000.        ],
    #         [ 8429.04370041 , 1179.08907689  ,2000.        ]])

    # path
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length = np.sum(segment_lengths) 

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)
    # terrain = np.flip(terrain)
    # terrain[:, :, 4:] *= 0.9
    # terrain[:, :, :4] *= 0.5
    # plot_drone(drone, terrain, pixel_size, dt)

    # plot_drone(drone terrain, pixel_size)


    # arrow_scale = 0.1
    # detection_coverage = drone.total_coverage(terrain, pixel_size)
    # # detection_coverage = 1 - (np.prod(1-detection_coverage, axis=2))
    # detection_coverage = np.max(detection_coverage, axis=2)
    # # detection_coverage = np.sum(detection_coverage, axis=2)
    # detection_coverage = np.clip(detection_coverage[:, :], 0, 1)
    # plt.figure(figsize=(8, 8))

    # plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    # plt.imshow(detection_coverage, origin="lower",
    #         extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
    #         cmap='bone')
    # plt.colorbar(label='Detection Probability')
    # # plt.arrow(drone.position[0], drone.position[1],
    # #         drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
    # #         head_width=0.2, head_length=0.1, fc='k', ec='k', width=0.05)
    # plt.title('Drone Camera Coverage Map')
    # plt.xlabel('World X')
    # plt.ylabel('World Y')
    # plt.axis('equal')
    # plt.show()

   
