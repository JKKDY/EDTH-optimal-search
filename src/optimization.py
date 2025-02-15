import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


flight_time = 3*60*60 # 3hours
flight_speed = 0.01#40/3.6 # 40 km/h
max_distance = flight_speed * flight_time
print("max_distance",max_distance)
max_distance = 50
num_timesteps = 40
pixel_size = 0.1
num_waypoints = 8

limits = np.array([[0,0,3], [0, 10,3], [5,10,3], [10,0,3]], dtype=float)  

def total_discovery(drone: Drone): 
  observation = np.zeros((50, 50)) 
  for step in range(num_timesteps):
    observation = np.max([observation, drone.detection_coverage(step, observation, pixel_size, 4, 10)], axis=0)
  # plt.imshow(observation, extent=(0,pixel_size*observation.shape[0], 0, pixel_size*observation.shape[1]), origin='lower')
  # plt.plot(drone.positions[:,0], drone.positions[:,1])
  # plt.show()
  return observation



def assemble_x(path, camera):
  path_flat = pathing.pack_constant_alt(path)
  cam_flat = camera.flatten()
  return np.concat([path_flat, cam_flat]), np.array([len(path_flat), len(cam_flat)])

def unpack_x(x, offsets):
  path = pathing.unpack_constant_alt(x[:offsets[0]])
  camera = x[offsets[0]:offsets[1]]
  return path, camera

def f(x, *args):
  path, camera = unpack_x(x, args[0])
  drone = Drone(path, 
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps)  
  discovery = total_discovery(drone)

  score = np.linalg.norm(
              np.array([
                1.0 - scoring.discovery_score(discovery),
                scoring.total_path_length(path)-max_distance,
            ]))
  print("score", score)
  # plt.imshow(discovery, extent=(0,pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
  # plt.plot(drone.positions[:,0], drone.positions[:,1])
  # plt.colorbar()
  # plt.show()
  return score

if __name__ == "__main__":
    path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)
    camera0 = np.array([0])
    assert np.allclose(unpack_x(*assemble_x(path0, camera0))[0], path0)

    # Maybe the drone wants to leave the area
    bounds_min, offsets = assemble_x(np.ones_like(path0) * np.min(limits) , -np.ones_like(camera0)*np.pi)
    bounds_min[-1] = 0.4
    bounds_max, _ = assemble_x(np.ones_like(path0) * np.max(limits) , np.ones_like(camera0)*np.pi)
    bounds_max[-1] = 5
    bounds = np.vstack((bounds_min, bounds_max)).T
    # res = opt.differential_evolution(f, x0=assemble_x(path0, camera0), bounds=bounds, 
    #                                  disp=True, workers=4, maxiter=2)
    # res = opt.dual_annealing(f, x0=assemble_x(path0, camera0), bounds=bounds, maxiter=70)
    res = opt.minimize(f, x0=assemble_x(path0, camera0)[0], args=(assemble_x(path0, camera0)[1]), bounds=bounds)
    
    path, camera = unpack_x(res.x, offsets)
    drone = Drone(path,
                camera_elevation=np.deg2rad(60), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40),
                num_timesteps=num_timesteps)  
    discovery = total_discovery(drone)
    print(path)

    plt.imshow(discovery, extent=(0,pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1])
    plt.colorbar()
    plt.show()
    