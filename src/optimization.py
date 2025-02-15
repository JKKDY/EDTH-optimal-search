import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 30
num_timesteps = 40
pixel_size = 0.1
num_waypoints = 8

limits = np.array([[0,0,3], [0,5,3], [5,5,3], [5,0,3]])  

def total_discovery(drone: Drone): 
  terrain = np.ones((70, 70, 8)) 
  observation = np.zeros(terrain.shape[:2])
  for step in range(num_timesteps):
    observation = np.max([observation, drone.detection_coverage(step, terrain, pixel_size, 4, 10)], axis=0)
  # plt.imshow(observation, extent=(0,pixel_size*observation.shape[0], 0, pixel_size*observation.shape[1]), origin='lower')
  # plt.plot(drone.positions[:,0], drone.positions[:,1])
  # plt.show()
  return observation



def assemble_x(path:np.ndarray, camera:np.ndarray):
  path_flat = pathing.pack_constant_alt(path)
  cam_flat = camera.flatten()
  return np.concat([path_flat, cam_flat]), np.array([len(path_flat), len(cam_flat)])

def unpack_x(x:np.ndarray, offsets:np.ndarray):
  path = pathing.unpack_constant_alt(x[:offsets[0]])
  camera = x[offsets[0]:offsets[1]]
  return path, camera

if __name__ == "__main__":
    path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)
    camera0 = np.array([0])
    assert np.allclose(unpack_x(*assemble_x(path0, camera0))[0], path0)
    assert np.allclose(unpack_x(*assemble_x(path0, camera0))[1], camera0)

    # Maybe the drone wants to leave the area
    bounds_min, offsets = assemble_x(np.ones_like(path0) * (np.min(limits)-0.2* np.max(limits)) , -np.pi*np.ones_like(camera0))
    bounds_max, _ = assemble_x(np.ones_like(path0) * (1.2*np.max(limits)) , np.pi*np.ones_like(camera0))
    # Manual bound on height
    bounds_min[-2] = 0.4
    bounds_max[-2] = 5
    bounds = np.vstack((bounds_min, bounds_max)).T

    def f(x):
        # offsets = args[0]
        path, camera = unpack_x(x, offsets)
        drone = Drone(path, 
                      camera_elevation=np.deg2rad(60), 
                      camera_azimuth=np.deg2rad(0), 
                      camera_fov=np.deg2rad(40),
                      num_timesteps=num_timesteps)  
        discovery = total_discovery(drone)
        discovered_percentage = scoring.discovery_score(discovery)
        distance_penalty = scoring.total_path_length(path)-max_distance
        score = (1.0 - discovered_percentage) + distance_penalty**2
        # score = np.linalg.norm(score)
        print("score", discovered_percentage, distance_penalty)
        # plt.imshow(discovery, extent=(0,pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
        # plt.plot(drone.positions[:,0], drone.positions[:,1])
        # plt.colorbar()
        # plt.show()
        return score

    print("bounds", bounds)
    print("x0", assemble_x(path0, camera0)[0])
    res = opt.differential_evolution(f, bounds=bounds, 
                                     x0=assemble_x(path0, camera0)[0], 
                                    #  args=(offsets), 
                                     workers=4, disp=True, maxiter=10)
    # res = opt.dual_annealing(f, x0=assemble_x(path0, camera0), bounds=bounds, maxiter=70)
    # res = opt.minimize(f, x0=assemble_x(path0, camera0)[0], bounds=bounds)
    
    path, camera = unpack_x(res.x, offsets)
    drone = Drone(path,
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps)  
    discovery = total_discovery(drone)
    print(path)

    plt.imshow(discovery, extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1])
    plt.colorbar()
    plt.show()
    