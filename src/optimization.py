import numpy as np
import scipy.optimize as opt
from skopt import gp_minimize

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 30
num_timesteps = 80
pixel_size = 0.1
num_waypoints = 16

limits = np.array([[0,0,3], [0,5,3], [5,5,3], [5,0,3]])  


def assemble_x(path:np.ndarray, camera:np.ndarray):
    # path_flat = pathing.pack_constant_alt(path)
    path_flat = pathing.pack_constant_segments(path)
    cam_flat = camera.flatten()
    return np.concat([path_flat, cam_flat]), np.array([len(path_flat), len(cam_flat)])

def unpack_x(x:np.ndarray, offsets:np.ndarray):
    # path = pathing.unpack_constant_alt(x[:offsets[0]])
    path = pathing.unpack_constant_segments(x[:offsets[0]])
    camera = x[offsets[0]:offsets[1]]
    return path, camera


counter = 0
if __name__ == "__main__":
    path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)
    camera0 = np.array([0])
    unpacked_path, _ = unpack_x(*assemble_x(path0, camera0))
    assert np.allclose(unpacked_path, path0)
    assert np.allclose(unpack_x(*assemble_x(path0, camera0))[1], camera0)

    # Maybe the drone wants to leave the area
    bounds_min, offsets = assemble_x(np.ones_like(path0) * (np.min(limits)-0.2* np.max(limits)) , -np.pi*np.ones_like(camera0))
    bounds_max, _ = assemble_x(np.ones_like(path0) * (1.2*np.max(limits)) , np.pi*np.ones_like(camera0))
    # Manual bound on height
    bounds_min[:-2] = -1.9*np.pi 
    bounds_max[:-2] =  1.9*np.pi 
    bounds_min[:2] = 0
    bounds_max[:2] = 4
    bounds_min[2] = 0.5
    bounds_max[2] = 6
    bounds_min[-2] = 0.4
    bounds_max[-2] = 5

    print(bounds_min)
    
    bounds = np.vstack((bounds_min, bounds_max)).T

    terrain = np.ones((100, 100, 8))

    def f(x):
        path, camera = unpack_x(x, offsets)
        drone = Drone(path, 
                      camera_elevation=np.deg2rad(60), 
                      camera_azimuth=np.deg2rad(0), 
                      camera_fov=np.deg2rad(40),
                      num_timesteps=num_timesteps)  
        discovery = drone.total_coverage(terrain, pixel_size)
        discovered_percentage = scoring.discovery_score(discovery)
        distance_penalty = scoring.total_path_length(path)-max_distance
        score = (1.0 - discovered_percentage) + distance_penalty**2
        
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
                                     workers=4, disp=True, maxiter=10)
    # res = opt.minimize(f, x0=assemble_x(path0, camera0)[0], bounds=bounds)
    # print([tuple(elm) for elm in bounds])
    # res = gp_minimize(f, dimensions=[tuple(elm) for elm in bounds], x0=x0, n_calls=200, 
    #                   n_initial_points=30, random_state=42, verbose=1, n_jobs=4)
    
    path, camera = unpack_x(res.x, offsets)
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
    