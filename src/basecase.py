import numpy as np
import scipy.optimize as opt
from pyswarm import pso

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 60e3 * 1 
num_timesteps = 20
pixel_size = 0.1
num_waypoints = 6

free_altitude = True
free_camera = False


if __name__ == "__main__":
    map_shape = (50, 50, 8)
    limits = np.array([[0,0,2e3], [0,12e3,2e3], [12e3,12e3,2e3], [12e3,0,2e3]])
    limits[:,:2] *= 2
    pixel_size = 260*2 # in [m]
    terrain = np.load("terrain/Kursk_3_50x50.npy", allow_pickle=True).reshape(map_shape)
    roads = np.load("terrain/Kursk_3_50x50_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    prior = scoring.compute_prior(roads)
    
    path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)

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
    plt.imshow((discovery[:,:,2]))
    plt.show()
    discovered_percentage = scoring.discovery_score(discovery)
    distance_penalty = scoring.total_path_length(path0) 
    plt.imshow(scoring.discovery_score_map(discovery), extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1])
    plt.colorbar()
    plt.title("Drone's view coverage")
    print("Total Path length",distance_penalty)
    plt.show()
