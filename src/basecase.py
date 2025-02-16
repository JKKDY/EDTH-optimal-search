import numpy as np
import scipy.optimize as opt
from pyswarm import pso

import matplotlib.pyplot as plt

from drone import Drone
import scoring
import pathing


max_distance = 121760.8016296049
num_timesteps = 160
pixel_size = 260/4 # in [m]
num_waypoints = 120

free_altitude = True
free_camera = False

map_shape = (200, 200, 8)
limits = np.array([[0,0,2e3], [0,14e3,2e3], [13e3,13e3,2e3], [13e3,0,2e3]])
limits[:,:2] *= 1
starting_point = np.array([0,0,2e3])

if __name__ == "__main__":
    n = 200
    map_shape = (n, n, 8)
    i = 4
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    
    roads = np.load(f"terrain/Kursk_{i}_{n}x{n}_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    buildings = np.load(f"terrain/Kursk_{i}_{n}x{n}_buildings.npy", allow_pickle=True).reshape(map_shape[:2])
    trees = np.load(f"terrain/Kursk_{i}_{n}x{n}_trees.npy", allow_pickle=True).reshape(map_shape[:2])
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    prior = scoring.compute_prior(roads)
    
    path0 = pathing.lawnmower_fill(starting_point, 12500, 12500, 4, num_waypoints)

    path0 = np.array(
[[    0.        ,     0.  ,        2000.,        ],
 [ 4032.25806452,     0.  ,        2000.,        ],
 [ 8064.51612903,     0.  ,        2000.,        ],
 [12096.77419355,     0.  ,        2000.,        ],
 [10433.46774194,  1562.5 ,        2000.,        ],
 [ 6401.20967742,  1562.5 ,        2000.,        ],
 [ 2368.9516129 ,  1562.5 ,        2000.,        ],
 [  100.80645161,  3125.  ,        2000.,        ],
 [ 4133.06451613,  3125.  ,        2000.,        ],
 [ 8165.32258065,  3125.  ,        2000.,        ],
 [12197.58064516,  3125.  ,        2000.,        ],
 [10332.66129032,  4687.5 ,        2000.,        ],
 [ 6300.40322581,  4687.5 ,        2000.,        ],
 [ 2268.14516129,  4687.5 ,        2000.,        ],
 [  201.61290323,  6250.  ,        2000.,        ],
 [ 4233.87096774,  6250.  ,        2000.,        ],
 [ 8266.12903226,  6250.  ,        2000.,        ],
 [12298.38709677,  6250.  ,        2000.,        ],
 [10231.85483871,  7812.5 ,        2000.,        ],
 [ 6199.59677419,  7812.5 ,        2000.,        ],
 [ 2167.33870968,  7812.5 ,        2000.,        ],
 [  302.41935484,  9375.  ,        2000.,        ],
 [ 4334.67741935,  9375.  ,        2000.,        ],
 [ 8366.93548387,  9375.  ,        2000.,        ],
 [12399.19354839,  9375.  ,        2000.,        ],
 [10131.0483871 , 10937.5 ,        2000.,        ],
 [ 6098.79032258, 10937.5 ,        2000.,        ],
 [ 2066.53225806, 10937.5 ,        2000.,        ],
 [  403.22580645, 12500.  ,        2000.,        ],
 [ 4435.48387097, 12500.  ,        2000.,        ],
 [ 8467.74193548, 12500.  ,        2000.,        ],
 [12500.        , 12500.  ,        2000.,        ],])
    path0 = np.array([[    0.         ,    0.         , 2000.        ],
 [  494.30067394 , 3930.7353755  , 1999.48310038],
 [ 4653.2720684  ,  816.40658973 , 1999.48310038],
 [ 5599.05091405 , 3449.39359557 , 1999.48310038],
 [ 7746.60886458 , 4157.64519347 , 1999.48310038],
 [ 9183.59765641 , 3469.94151056 , 1999.48310038],
 [12102.4053737  , 4707.397645   , 1999.48310038],
 [10230.47251374 , 6819.75381084 , 1999.48310038],
 [10427.59283039 , 7607.86075399 , 1999.48310038],
 [ 9530.71324152 , 5779.98394408 , 1999.48310038],
 [ 6985.33508135 , 2820.85109097 , 1999.48310038],
 [ 4839.58170301 , 3274.76572834 , 1999.48310038],
 [ 3522.21655658 , 3709.36571602 , 1999.48310038],
 [ 2723.92735744 ,   68.20755947 , 1999.48310038],
 [ 9400.6008869  ,    0.         , 1999.48310038],
 [ 8530.57916224 , 3560.53065989 , 1999.48310038],
 [ 5156.59371089 , 7580.88111349 , 1999.48310038],
 [ 3218.48527256 , 7001.87655603 , 1999.48310038],
 [ 3325.85656189 , 7519.58208499 , 1999.48310038],
 [ 6274.97485994 , 8645.05260488 , 1999.48310038],
 [ 6225.92241969 , 9030.91104813 , 1999.48310038],
 [ 6163.62245978 , 9129.05191027 , 1999.48310038],
 [ 5351.38437872 , 9713.96431716 , 1999.48310038],
 [ 1717.6985254  ,11880.71544885 , 1999.48310038],
 [ 8547.48244329 ,10461.73070696 , 1999.48310038],
 [12149.99972415 ,12007.98195311 , 1999.48310038],]
)



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
    plt.imshow(scoring.discovery_score_map(discovery, prior), extent=(0, pixel_size*discovery.shape[0], 0, pixel_size*discovery.shape[1]), origin='lower')
    plt.plot(drone.positions[:,0], drone.positions[:,1], "k--")
    plt.colorbar()
    plt.title("Drone's view coverage")
    discovered_percentage = scoring.discovery_score(discovery, prior)
    distance_penalty = scoring.total_path_length(path0) - max_distance
    out_of_bounds_penalty = np.sum((path0 - np.clip(path0, np.zeros(3), np.max(limits, axis=0)))**2)/num_waypoints
    
    score = 1e2*(1.0 - discovered_percentage) + 1e-3*distance_penalty**2 + 1e-3*out_of_bounds_penalty
    score = 1e2*(1.0 - discovered_percentage) + 1e-4*scoring.total_path_length(path0) + 1e-3*out_of_bounds_penalty
    print("Total Path length", scoring.total_path_length(path0))
    print("score", score)
        
    plt.show()
