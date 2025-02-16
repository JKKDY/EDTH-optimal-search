import numpy as np
import scipy.optimize as opt
from pyswarm import pso

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 121760.8016296049
num_timesteps = 160
pixel_size = 260 # in [m]
num_waypoints = 120

free_altitude = True
free_camera = False

map_shape = (50, 50, 8)
limits = np.array([[0,0,2e3], [0,14e3,2e3], [13e3,13e3,2e3], [13e3,0,2e3]])
limits[:,:2] *= 1
starting_point = np.array([0,0,2e3])

if __name__ == "__main__":
    terrain = np.load("terrain/Kursk_4_50x50.npy", allow_pickle=True).reshape(map_shape)
    roads = np.load("terrain/Kursk_4_50x50_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    prior = scoring.compute_prior(roads)
    
    path0 = pathing.lawnmower_fill(starting_point, 12500, 12500, 4, num_waypoints)

    plt.subplot(121)
    plt.imshow(prior, extent=(0, pixel_size*prior.shape[0], 0, pixel_size*prior.shape[1]), origin='lower') 
    plt.colorbar()
    plt.title("Prior probability of targets")
    plt.subplot(122)
    drone = Drone(path0, 
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps)  
    discovery = drone.total_coverage(terrain, pixel_size)
    plt.imshow(scoring.discovery_score_map(discovery), extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1])
    plt.colorbar()
    plt.title("Drone's view coverage")
    discovered_percentage = scoring.discovery_score(discovery)
    distance_penalty = scoring.total_path_length(path0) - max_distance
    out_of_bounds_penalty = np.sum((path0 - np.clip(path0, np.zeros(3), np.max(limits, axis=0)))**2)/num_waypoints
    
    score = 1e2*(1.0 - discovered_percentage) + 1e-3*distance_penalty**2 + 1e-3*out_of_bounds_penalty
    print("Total Path length", scoring.total_path_length(path0))
    print(discovered_percentage, distance_penalty, out_of_bounds_penalty)
    print("score", score)
        
    plt.show()
