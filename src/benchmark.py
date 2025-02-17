import scoring
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from pathing import lawnmower_fill
from scoring import activation_function
import multiprocessing as mp



def benchmark_worker(args):
    """Worker function for parallel execution."""
    drone, terrain, prior, pixel_size, num_samples = args
    exp_value = 0

    for _ in range(num_samples):
        prob_array = prior / np.sum(prior)
        flat_probs = prob_array.flatten()
        sampled_index = np.random.choice(len(flat_probs), p=flat_probs)
        target = np.unravel_index(sampled_index, prob_array.shape)
        exp_value += drone.expected_target_detection_time(target, terrain, pixel_size)
    
    return exp_value  #return the sum, averaging is done later

def benchmark(drone, terrain, prior, pixel_size, n=1000, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()  # Use all available cores
    num_samples_per_worker = n // num_workers  # Divide workload evenly
    extra_samples = n % num_workers  # Distribute remaining samples

    # Create argument tuples for each worker
    args_list = [(drone, terrain, prior, pixel_size, num_samples_per_worker + (1 if i < extra_samples else 0))
                 for i in range(num_workers)]

    # Start parallel processing
    np.random.seed(42)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(benchmark_worker, args_list)

    # Aggregate results: Sum over all workers, then normalize
    total_exp_value = sum(results)
    return total_exp_value / n


   
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
    

def benchmark_path(path:np.array, plot=True):
    n = 200
    num_benchmarks = 1000
    pixel_size = 13000/n # in [m]
    map_shape = (n, n, 8)
    i = 4
    roads = np.load(f"terrain/Kursk_{i}_{n}x{n}_roads.npy", allow_pickle=True).reshape(map_shape[:2])
    buildings = np.load(f"terrain/Kursk_{i}_{n}x{n}_buildings.npy", allow_pickle=True).reshape(map_shape[:2])
    trees = np.load(f"terrain/Kursk_{i}_{n}x{n}_trees.npy", allow_pickle=True).reshape(map_shape[:2])
    terrain = np.load(f"terrain/Kursk_{i}_{n}x{n}.npy", allow_pickle=True).reshape(map_shape)
    prior = scoring.compute_prior(0.2 * roads + 0.35 * buildings + 0.45* trees)

    drone = Drone(path, num_timesteps=300)
    print("Benchmark (expected time to detect target):", benchmark(drone, terrain, prior, pixel_size, num_benchmarks))
    detection_coverage = drone.total_coverage(terrain, pixel_size)
    detection_coverage = np.max(detection_coverage, axis=2)

    if plot: 
        plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
        plt.imshow(detection_coverage, origin="lower",
                extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
                cmap='bone')
        plt.colorbar(label='Detection Probability')
        plt.axis('equal')
        plt.savefig("lawnmower.png")
        plt.show()


def path_distance(path):
    dist = 0
    for i in range(len(path)-1):
        x1 = path[i]
        x2 = path[i+1]
        dist += np.linalg.norm(x1[:2] - x2[:2])

    return dist

if __name__ == "__main__":
    lawnmower = lawnmower_path((100, 100, 2000), 12500, 12500, 4)
    print("lawnmower Path distance:", path_distance(lawnmower))
   
    optimized_path = np.array([
        [    0.,             0. ,         2000.        ],
        [11226.24524245,  4305.77929347 , 1998.63854871],
        [10765.15791932,  1154.42651301 , 1998.63854871],
        [ 9210.50199422,  6381.51823291 , 1998.63854871],
        [ 5060.72873017,  4826.66645017 , 1998.63854871],
        [ 1220.43853144,  4198.63160928 , 1998.63854871],
        [ 6302.84147666,  8014.22556769 , 1998.63854871],
        [ 1211.19811833,  5194.52750229 , 1998.63854871],
        [ 6508.7079487 ,  6431.1564223  , 1998.63854871],
        [ 5145.88962206,  1069.59956887 , 1998.63854871],
        [ 6931.79847361,  5418.73500009 , 1998.63854871],
        [ 6376.34422367,  6718.89330616 , 1998.63854871],
        [ 3921.36300132,   900.23905856 , 1998.63854871],
        [ 3434.13672436,  2373.14080902 , 1998.63854871],
        [ 1097.52521524,  8775.79918046 , 1998.63854871],
        [ 1538.18032331,  1632.45094846 , 1998.63854871],
        [ 2843.22195901,  4921.31600798 , 1998.63854871],
        [ 1998.90358232,  1079.77572469 , 1998.63854871],
        [ 2634.90446054,  4022.44962501 , 1998.63854871],
        [ 4399.5347394 ,  8068.28845177 , 1998.63854871],
        [ 3395.17592065,  7370.04008304 , 1998.63854871],
        [ 4624.54687109, 10436.42043672 , 1998.63854871],
        [ 1361.4803444 , 10318.39051224 , 1998.63854871],
        [ 5264.47844395,  4450.84350771 , 1998.63854871],
        [ 9295.39977705,  9492.45068894 , 1998.63854871],
        [ 3983.05423792, 10854.64561088 , 1998.63854871],])
    benchmark_path(lawnmower)
   
    print("optimized_path Path distance:", path_distance(optimized_path))
    benchmark_path(optimized_path)