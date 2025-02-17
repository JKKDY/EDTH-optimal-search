import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from skimage.draw import polygon
from skimage.draw import disk

epsilon = 1e-6

class Drone:
    def __init__(self, flight_path, camera_elevation=0.0, camera_azimuth=0.0, camera_fov=np.deg2rad(60),
                 num_timesteps = 100, certain_detection_distance = 1500, max_detection_distance=8000):
        """
        Initialize the Drone.

        Parameters:
        - flight_path: list of numpy arrays representing points ([(x,y,z), (x,y,z), ...]).
        - camera_elevation: initial elevation angle of the camera in radians. 0.0 means the camera is level
        - camera_azimuth: initial azimuth angle of the camera in radians. 0.0 means the camera is looking ahead
        - camera_fov: initial horizontal field of view in radians
        - num_timesteps: number of position points after linearly interpolating the path 
        - certain_detection_distance: maximum distance at which the drone will have 100% detection capability
        - max_detection_distance: maximum distance at which the drone will any detection capability
        """
        assert certain_detection_distance < max_detection_distance


        self.path = flight_path
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.camera_fov = camera_fov 
        self.certain_detection_distance = certain_detection_distance
        self.max_detection_distance = max_detection_distance
        
        self.num_timesteps = num_timesteps
        distances = np.cumsum(np.concat([[0.0], np.linalg.norm(np.diff(self.path, axis=0), axis=1)]))
        
        interp_points = np.linspace(0.0, distances[-1], self.num_timesteps)
        x_interp = np.interp(interp_points, distances, self.path[:,0])
        y_interp = np.interp(interp_points, distances, self.path[:,1])
        z_interp = np.interp(interp_points, distances, self.path[:,2])
        
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


    def camera_view(self, timestep_idx): 
        """
        calculate current view coverage of the camera

        Parameters:
            - timestep_idx: used to specify current position and bearing of the drone
        
        Returns:
            nd.array (4,2) containing the 4 corners of the cameras view
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
    

    def calculate_detection_confidence(self, distances, terrain):
        """
        given the terrain and the distance of each point of the terrain to the drone, return the detection confidence of each point
        """
        assert distances.flatten().shape == terrain.shape, f"{distances.shape} vs {terrain.shape}"
        terrain2 = 0.5 + (terrain)*0.5
        P = 1 - (distances.flatten()/terrain2 - self.certain_detection_distance) / (self.max_detection_distance - self.certain_detection_distance)
        return np.clip(P, 0, 1)
    
    def calculate_detection_coverage(self, rr, cc, timestep_idx, terrain, pixel_size):
        """
        given an area of interest (columns:cc, rows:rr) and a time index, calculate the detection confidence for under consideration of the view direction
        view directions are binned into 8 classes/octants
        """
        # this is a helper function to avoid code duplication
        detection_coverage = np.zeros(terrain.shape)
        xyzcoordinates = (np.array([cc, rr, np.zeros_like(rr)]).T * pixel_size)

        diffs = self.positions[timestep_idx] - xyzcoordinates
        distances = np.linalg.norm(diffs, axis=1, keepdims=True)
        
        normalized_diffs = diffs/distances 

        height_thresh = 0.7
        is_northern_octant = normalized_diffs[:, 0] > 0
        is_eastern_octant = normalized_diffs[:, 1] > 0
        is_top_octant = normalized_diffs[:, 2] > height_thresh

        # masks for selecting a single octant 
        masks = [
            is_northern_octant  & is_eastern_octant  & ~is_top_octant,
            is_northern_octant  & ~is_eastern_octant & ~is_top_octant,
            ~is_northern_octant & ~is_eastern_octant & ~is_top_octant,
            ~is_northern_octant & is_eastern_octant  & ~is_top_octant,
            is_northern_octant  & is_eastern_octant  & is_top_octant ,
            is_northern_octant  & ~is_eastern_octant & is_top_octant ,
            ~is_northern_octant & ~is_eastern_octant & is_top_octant ,
            ~is_northern_octant & is_eastern_octant  & is_top_octant ,
        ]   

        # for each mask evaluate coverage
        for i in range(terrain.shape[2]):
            rrr = rr[masks[i]]
            ccc = cc[masks[i]]
            detection_coverage[rrr, ccc, i] = self.calculate_detection_confidence(distances[masks[i]], terrain[rrr, ccc, i])

        return detection_coverage


    def coverage_of_camera(self, timestep_idx, terrain, pixel_size):
        """
        Calculate a detection probability for every pixel in the terrain that lies within the camera's view.
        
        Parameters:
            - timestep_idx: used to specify current position and bearing of the drone
            - terrain: (n,m,8) numpy array representing the terrain features for each octant
            - pixel_size: The size (in world units) of one pixel.
        
        Returns:
          detection_coverage: 3D numpy array of the same shape as terrain containing the detection probabilities.
        """
        assert terrain.shape[2] == 8

        view_extent = self.camera_view(timestep_idx)
        
        # Convert world coordinates to pixel indices.
        poly_cols = view_extent[:, 0] / pixel_size  # x -> column
        poly_rows = view_extent[:, 1] / pixel_size  # y -> row

        # Use skimage.draw.polygon to get indices of all pixels inside the polygon.
        rr, cc = polygon(poly_rows, poly_cols, shape=terrain.shape[:2])

        return self.calculate_detection_coverage(rr, cc, timestep_idx, terrain, pixel_size)
    
    
    def coverage_of_drone(self, timestep_idx, terrain, pixel_size): 
        """
        Calculate a detection probability for every pixel in the terrain that lies within the drones view.
        
        Parameters:
            - timestep_idx: used to specify current position and bearing of the drone
            - terrain: (n,m,8) numpy array representing the terrain features for each octant
            - pixel_size: The size (in world units) of one pixel.
        
        Returns:
          detection_coverage: 3D numpy array of the same shape as terrain containing the detection probabilities.
        """
        pos = self.positions[timestep_idx]
        center = self.positions[timestep_idx][:2].astype(float) / pixel_size
        center = (center[1], center[0]) # no clue why we need to permutate the coordinates
        radius = int(np.sqrt(self.max_detection_distance**2 + pos[2]**2) / pixel_size)

        # skimage.draw.disk returns rr (rows) and cc (columns).
        rr, cc = disk(center, radius, shape=terrain.shape[:2])
        # plt.show()
        half_angle = 120  # in degrees
    
        # Normalize the sensor's direction vector.
        sensor_dir = self.directions[timestep_idx][:2].astype(float)
        unit_dir = sensor_dir / np.linalg.norm(sensor_dir)
        
        # Compute the vector from the center to each point:
        dx = cc - center[1]  # x difference
        dy = rr - center[0]  # y difference
        
        # Compute distance of each point to the center
        vec_norm = np.sqrt((dx)**2 + dy**2)
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
        
        return self.calculate_detection_coverage(rr, cc, timestep_idx, terrain, pixel_size) 


    def total_coverage(self, terrain, pixel_size):
        """
        calcualte the total terrain coverage over the entire flight path of the drone
        """
        observation = np.zeros(terrain.shape)
        for step in range(self.num_timesteps):
            # detectable object size at max zoom range = sin(1.5deg (fov at max zoom) ) * 8km / 1080px * 20px (minimal detection size)
            observation = np.maximum(observation, self.coverage_of_drone(step, terrain, pixel_size))
            # observation = np.maximum(observation, self.coverage_of_camera(step, terrain, pixel_size))
        return observation
    
    def expected_target_detection_time(self, target, terrain, pixel_size):
        cc, rr = target 
        coverages = []
        for timestep_idx in range(1, self.num_timesteps):
            coverage = self.calculate_detection_coverage(np.array([cc]), np.array([rr]), timestep_idx, terrain, pixel_size)
            coverages.append(np.max(coverage[cc, rr]))

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
            trials.append((max_idx, max_prob))

        remaining_probabilities = []
        current_probability = 0.0
        for trial in trials:
            remaining_prob = trial[1]*(1.0 - current_probability)
            remaining_probabilities.append((trial[0], remaining_prob))
            current_probability += remaining_prob

        exp_value = sum([x*y for x,y in remaining_probabilities])
        # print(remaining_probabilities)
        # print(exp_value)
        # plt.plot(coverages)
        # plt.show()
        return exp_value
       



if __name__ == "__main__":
    path = np.array([[0,0,2000], [9000, 2000, 2000], [1000,4000,2000], [9000,6000,2000]])

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    arrow_scale = 0.1
    detection_coverage = drone.total_coverage(terrain, pixel_size)
    detection_coverage = np.max(detection_coverage, axis=2)
    detection_coverage = np.clip(detection_coverage[:, :], 0, 1)

    plt.figure(figsize=(8, 8))

    plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='bone')
    plt.colorbar(label='Detection Probability')
    # plt.arrow(drone.position[0], drone.position[1],
    #         drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
    #         head_width=0.2, head_length=0.1, fc='k', ec='k', width=0.05)
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X')
    plt.ylabel('World Y')
    plt.axis('equal')
    plt.show()

   
