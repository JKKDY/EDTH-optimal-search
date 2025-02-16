import scoring
import numpy as np
import matplotlib.pyplot as plt

def benchmark(drone):
    map_shape = (50, 50, 8)
    pixel_size = 260 # in [m]


    n = 50
    i = 4
    roads = np.load(f"terrain/Kursk_{i}_{n}x{n}_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    buildings = np.load(f"terrain/Kursk_{i}_{n}x{n}_buildings.npy", allow_pickle=True).reshape(map_shape[:2])
    trees = np.load(f"terrain/Kursk_{i}_{n}x{n}_trees.npy", allow_pickle=True).reshape(map_shape[:2])
    

    prior = scoring.compute_prior(0.2 * roads + 0.35 * buildings + 0.45* trees)

    prob_array  = prior / np.sum(prior)
    flat_probs = prob_array.flatten()
    sampled_index = np.random.choice(len(flat_probs), p=flat_probs)
    sampled_point = np.unravel_index(sampled_index, prob_array.shape)

    xs = []
    ys = []

    for _ in range(1000):

        xs.append(sampled_point[0])
        ys.append(sampled_point[1])


    print("Sampled point (row, col):", xs,)
    
    # plt.imshow(prior)
    plt.scatter(xs, ys)
    plt.show()

    # plt.imshow(0.2 * roads + 0.35 * buildings + 0.45* trees)
    # plt.show()


    

    


if __name__ == "__main__":

    benchmark(None)