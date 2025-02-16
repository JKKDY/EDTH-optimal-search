import numpy as np
import scipy.optimize as opt
from pyswarm import pso

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing

plt.ion()


max_distance = 121760.8016296049 
num_timesteps = 100
pixel_size = 0.1
num_waypoints = 26

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
    bounds_min[-1] = 1.99e3
    bounds_max[-1] = limits[0,2]

    bounds = np.vstack((bounds_min, bounds_max)).T
    n = 50
    i = 4
    map_shape = (n, n, 8)
    pixel_size = 260 # in [m]
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    
    roads = np.load(f"terrain/Kursk_{i}_{n}x{n}_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    buildings = np.load(f"terrain/Kursk_{i}_{n}x{n}_buildings.npy", allow_pickle=True).reshape(map_shape[:2])
    trees = np.load(f"terrain/Kursk_{i}_{n}x{n}_trees.npy", allow_pickle=True).reshape(map_shape[:2])
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    prior = scoring.compute_prior(0.2 * roads + 0.35 * buildings + 0.45* trees)
    im = plt.imshow(roads, extent=(0,pixel_size*terrain.shape[0], 0, pixel_size*terrain.shape[1]), origin='lower', cmap="bone")
    line, = plt.gca().plot(path0[:,0], path0[:,1], color="k")
    plt.colorbar()
            
    def f(x):
        path, camera = unpack_x(x, offsets)
        drone = Drone(path, 
                      camera_elevation=np.deg2rad(60), 
                      camera_azimuth=np.deg2rad(0), 
                      camera_fov=np.deg2rad(40),
                      num_timesteps=num_timesteps)  
        discovery = drone.total_coverage(terrain, pixel_size)
        discovered_percentage = scoring.discovery_score(discovery, prior)
        distance_penalty = scoring.total_path_length(path) - max_distance
        out_of_bounds_penalty = np.sum((path - np.clip(path, np.zeros(3), np.max(limits, axis=0)))**2)/num_waypoints
        #     print("out of bounds penalty", out_of_bounds_penalty)
        #     print(path)
        #     print(path - np.clip(path, np.zeros(3), np.max(limits, axis=0)))
        score = 1e2*(1.0 - discovered_percentage) + 1e-3*distance_penalty**2 + 1e-3*out_of_bounds_penalty
        # print(score)
        # print(discovered_percentage, 1e-4*distance_penalty)
        global counter
        counter+=1
        if counter % (num_waypoints//2) == 0:
            im.set_data(scoring.discovery_score_map(discovery, prior))
            line.set_ydata(drone.positions[:,1])
            line.set_xdata(drone.positions[:,0])
            # plt.plot(drone.positions[:,0], drone.positions[:,1], color="k")
            plt.title(f"Intermediate optimization score: {score:.3e}\nExplored {counter*4} possible paths")
            plt.show()
            plt.pause(0.05)
        return score
    x0 = assemble_x(path0, camera0)[0]
    # res = opt.differential_evolution(f, bounds=bounds,
    #                                  x0=x0, #popsize=50,
    #                                  workers=4, disp=True, maxiter=20)
    # res = opt.minimize(f, x0=assemble_x(path0, camera0)[0], bounds=bounds, )
    # x = res.x
    x, _ = pso(f, lb=bounds_min, ub=bounds_max, maxiter=20, debug=True) 

    path, camera = unpack_x(x, offsets)
    drone = Drone(path,
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps) 
    discovery = drone.total_coverage(terrain, pixel_size)
    print(path)

    plt.ioff()  # Turn off interactive mode
    plt.imshow(scoring.discovery_score_map(discovery, prior), extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1],"k--")
    plt.colorbar()
    plt.show()
