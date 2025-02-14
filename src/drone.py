import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Drone:
    def __init__(self, path, velocity=1.0, camera_elevation=0.0, camera_azimuth=0.0, camera_fov=90):
        """
        Initialize the Drone.

        Parameters:
        - path: list of numpy arrays representing points ([(x,y,z), (x,y,z), ...]).
        - velocity: constant speed per move call (distance moved per move call).
        - camera_elevation: initial elevation angle of the camera in radians. 0.0 means the camera is level
        - camera_azimuth: initial azimuth angle of the camera in radians. 0.0 means the camera is looking ahead
        - camera_fov: initial field of view in degrees
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


    def adjust_camera(self, elevation, azimuth, fov):
        """
        Adjust the camera's elevation and azimuth angles.
        """
        self.camera_elevation = elevation
        self.camera_azimuth = azimuth
        self.camera_fov = fov

    def look_at(self, map=None): 
        max_elevation = self.camera_elevation + math.radians(self.camera_fov/2)
        min_elevation = self.camera_elevation - math.radians(self.camera_fov/2)




    def __str__(self):
        return (f"Drone(position={self.position}, "
                f"camera_elevation={self.camera_elevation}, "
                f"camera_azimuth={self.camera_azimuth})")



def plot_drone(drone, dt):
    # Set up the 2D plot (bird's-eye view) showing only x and y.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the full path (x and y only).
    ax.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    
    # Create a marker for the drone.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    
    # Set the limits of the plot.
    ax.set_xlim(np.min(path[:, 0]) - 5, np.max(path[:, 0]) + 5)
    ax.set_ylim(np.min(path[:, 1]) - 5, np.max(path[:, 1]) + 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Bird's-Eye View of Drone Flight")
    ax.legend()

    # Set the time step (seconds) for the simulation.
    def update(frame):
        if drone.current_path_idx >= len(drone.path) - 1:
            return drone_marker,
        drone.move(dt)
        drone_marker.set_data([drone.position[0]], [drone.position[1]])
        return drone_marker,

    # Create the animation.
    ani = FuncAnimation(fig, update, frames=300, interval=dt*500, blit=True)
    
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Define a simple 2D path.
    path = np.array([[0,0,10], [5,5,10], [5,7,10], [10,10,10]], dtype=float)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length = np.sum(segment_lengths) 

    dt = 0.02
    drone = Drone(path, velocity=2.0, )


    plot_drone(drone, dt)


   
