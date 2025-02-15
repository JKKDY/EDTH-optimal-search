import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


flight_time = 3*60*60 # 3hours
flight_speed = 40/3.6 # 40 km/h
max_distance = flight_speed * flight_time
dt = 1
pixel_size = 0.1
num_waypoints = 6

limits = np.array([[0,0], [0, 10], [10,10], [10,0]], dtype=float) 

def total_discovery(drone: Drone): 
  observation = np.zeros((50, 50))
  for step in range(int(flight_time/dt)):
    observation = np.max([observation, drone.detection_coverage(step, observation, pixel_size, 4, 10)], axis=0)
    plt.imshow(observation)
    plt.show()



def assemble_x(path, camera, altitude):
  return np.concat([path.flatten(), camera.flatten(), np.array([altitude])])

def unpack_x(x):
  path = x[:num_waypoints*2].reshape((num_waypoints,2))
  camera = x[num_waypoints*2:]
  altitude = x[num_waypoints*2+1]
  return path, camera, float(altitude)

def f(x):
  path, camera, altitude = unpack_x(x)
  path = np.vstack([path.T, altitude*np.ones(len(path))]).T
  drone = Drone(path, dt=dt, 
                camera_elevation=np.deg2rad(0), 
                camera_azimuth=np.deg2rad(0), 
                camera_fov=np.deg2rad(40))  
  discovery = total_discovery(drone)

  score = np.array([scoring.discovery_score(discovery) - 1.0,
                    scoring.total_path_length(path)-max_distance,
                    ])
  return score
  
  
if __name__ == "__main__":
  path0 = pathing.zigzag_fill(limits, num_waypoints=num_waypoints)
  print(path0)
  camera0 = np.array([0])
  # Maybe the drone wants to leave the area
  bounds_min = assemble_x(np.ones_like(path0) * np.min(limits) , np.ones_like(camera0)*np.pi, 1)
  bounds_max = assemble_x(np.ones_like(path0) * np.max(limits) , np.ones_like(camera0)*np.pi, 6)
  res=opt.least_squares(f, x0=assemble_x(path0, camera0, 3))

  print(res.x)