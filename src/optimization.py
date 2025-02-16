import numpy as np
import scipy.optimize as opt
from pyswarm import pso

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 121760.8016296049 
num_timesteps = 100
pixel_size = 0.1
num_waypoints = 24

free_altitude = True
compressed_representation = False 

limits = np.array([[0,0,2e3], [0,13e3,2e3], [13e3,13e3,2e3], [13e3,0,2e3]])    
starting_point = np.array([0,0,2e3])

def assemble_x(path:np.ndarray, camera:np.ndarray):
    path_flat = pathing.pack_constant_alt(path[1:])
    # path_flat = pathing.pack_constant_segments(path)
    cam_flat = camera.flatten()
    return np.concat([path_flat, cam_flat]), np.array([len(path_flat), len(cam_flat)])

def unpack_x(x:np.ndarray, offsets:np.ndarray):
    path = pathing.unpack_constant_alt(x[:offsets[0]])
    # path = pathing.unpack_constant_segments(x[:offsets[0]])
    path = np.concat([[starting_point], path])
    camera = x[offsets[0]:offsets[1]]
    return path, camera


counter = 0
if __name__ == "__main__":
    path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)
    path0 = pathing.lawnmower_fill(starting_point, 12500, 12500, 4, num_waypoints)
    camera0 = np.array([])
    assembledx = assemble_x(path0, camera0)
    unpacked_path, _ = unpack_x(*assembledx)
    assert np.allclose(unpacked_path, path0)
    assert np.allclose(unpack_x(*assembledx)[1], camera0)

    # Maybe the drone wants to leave the area
    bounds_min, offsets = assemble_x(np.zeros_like(path0), -np.pi*np.ones_like(camera0))
    bounds_max, _ = assemble_x(np.ones_like(path0) * np.max(limits) , np.pi*np.ones_like(camera0))
    if compressed_representation:
        # Manual bound on height
        bounds_min[:-1] = -1.9*np.pi 
        bounds_max[:-1] =  1.9*np.pi 
        bounds_min[:2] = -1
        bounds_max[:2] = np.max(limits)
        bounds_min[2] = 500
        bounds_max[2] = assembledx[0][2]+10

        assembledx[0][2] *= max_distance / scoring.total_path_length(path0)
        path0 = unpack_x(*assembledx)[0]
    bounds_min[-1] = 2e3
    bounds_max[-1] = limits[0,2]

    bounds = np.vstack((bounds_min, bounds_max)).T
    map_shape = (50, 50, 8)
    pixel_size = 260 # in [m]
    terrain = np.load("terrain/Kursk_4_50x50.npy", allow_pickle=True).reshape(map_shape)
    roads = np.load("terrain/Kursk_4_50x50_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    
    prior = scoring.compute_prior(roads)

    def f(x):
        path, camera = unpack_x(x, offsets)
        drone = Drone(path, 
                      camera_elevation=np.deg2rad(60), 
                      camera_azimuth=np.deg2rad(0), 
                      camera_fov=np.deg2rad(40),
                      num_timesteps=num_timesteps)  
        discovery = drone.total_coverage(terrain, pixel_size)
        discovered_percentage = scoring.discovery_score(discovery)
        distance_penalty = scoring.total_path_length(path) - max_distance
        out_of_bounds_penalty = np.sum((path - np.clip(path, np.zeros(3), np.max(limits, axis=0)))**2)/num_waypoints
        #     print("out of bounds penalty", out_of_bounds_penalty)
        #     print(path)
        #     print(path - np.clip(path, np.zeros(3), np.max(limits, axis=0)))
        score = 1e2*(1.0 - discovered_percentage) + 1e-3*distance_penalty**2 + 1e-3*out_of_bounds_penalty
        # print(score)
        global counter
        # counter+=1
        # if counter % (num_waypoints+1) == 0:
        #     print("score", discovered_percentage, distance_penalty)

        #     plt.imshow(np.sum(discovery, axis=-1), extent=(0,pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
        #     plt.plot(drone.positions[:,0], drone.positions[:,1])
        #     plt.colorbar()
        #     plt.show()
        return score
    x0 = assemble_x(path0, camera0)[0]
    res = opt.differential_evolution(f, bounds=bounds,
                                     x0=x0, 
                                     workers=4, disp=True, maxiter=20)
    # res = opt.minimize(f, x0=assemble_x(path0, camera0)[0], bounds=bounds, )
    x = res.x
    # x, _ = pso(f, lb=bounds_min, ub=bounds_max, maxiter=20, debug=True)

    path, camera = unpack_x(x, offsets)
    drone = Drone(path,
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps) 
    discovery = drone.total_coverage(terrain, pixel_size)
    print(path)

    plt.imshow(scoring.discovery_score_map(discovery), extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1])
    plt.colorbar()
    plt.show()
