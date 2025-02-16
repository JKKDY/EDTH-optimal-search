import scoring
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from pathing import lawnmower_fill
from scoring import activation_function

def benchmark(drone, prior):
   
    prob_array  = prior / np.sum(prior)
    flat_probs = prob_array.flatten()
    sampled_index = np.random.choice(len(flat_probs), p=flat_probs)
    sampled_point = np.unravel_index(sampled_index, prob_array.shape)


   
    


def lawnmower_path(start, x_dist, y_dist, num_zizags):
    """
    create a lawnmower path
    Parameters:
        start: (x,y,z) starting point of the drone
        x_dist: extent in x direction
        y_dist: extent in y direction 
    """
    curr = np.array(start)
    path = [curr]
    y_dist = y_dist/num_zizags/2
    for _ in range(num_zizags):
        path.append(curr:=curr+[x_dist, 0, 0])
        path.append(curr:=curr+[0, y_dist, 0])
        path.append(curr:=curr-[x_dist, 0, 0])
        path.append(curr:=curr+[0, y_dist, 0])
    path.append(curr:=curr+[x_dist, 0, 0])
    path = np.array(path)
    
    return path
    


if __name__ == "__main__":


    n =200
    pixel_size = 13000/n # in [m]
    map_shape = (n, n, 8)
    i = 4
    roads = np.load(f"terrain/Kursk_{i}_{n}x{n}_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    buildings = np.load(f"terrain/Kursk_{i}_{n}x{n}_buildings.npy", allow_pickle=True).reshape(map_shape[:2])
    trees = np.load(f"terrain/Kursk_{i}_{n}x{n}_trees.npy", allow_pickle=True).reshape(map_shape[:2])
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    prior = scoring.compute_prior(0.2 * roads + 0.35 * buildings + 0.45* trees)

    path = lawnmower_path((100, 100, 2000), 12500, 12500, 4)

    plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
        # detection_coverage = 1 - (np.prod(1-detection_coverage, axis=2))

    drone = Drone(path, num_timesteps=300)

    detection_coverage = drone.total_coverage(terrain, pixel_size)
    detection_coverage = np.max(detection_coverage, axis=2)

    plt.imshow(detection_coverage, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='viridis')
    plt.colorbar(label='Detection Probability')
    plt.axis('equal')
    plt.show()

    # benchmark(None)