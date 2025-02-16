import scoring
import numpy as np
import matplotlib.pyplot as plt
from drone import Drone
from pathing import lawnmower_fill
from scoring import activation_function

def benchmark(drone, terrain, prior, pixel_size):
    
    n = 200
    exp_value = 0
    for _ in range(n):
        prob_array  = prior / np.sum(prior)
        flat_probs = prob_array.flatten()
        # np.random.seed(1)
        sampled_index = np.random.choice(len(flat_probs), p=flat_probs)
        target = np.unravel_index(sampled_index, prob_array.shape)

        exp_value += drone.detect_target(target, terrain, pixel_size)

    return exp_value / n


   
    


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
    # path = np.array([[    0.         ,    0.          ,2000.        ],
    #         [  758.39689188 ,12355.15657072  ,2000.        ],
    #         [ 4297.30421253 ,12012.41540355  ,2000.        ],
    #         [ 1360.76381815 ,11456.55349753  ,2000.        ],
    #         [ 6755.7283386  , 8313.93550828  ,2000.        ],
    #         [ 6170.46952714 , 7199.88101115  ,2000.        ],
    #         [ 5738.92107382 , 5044.74214392  ,2000.        ],
    #         [ 8662.49657626 , 4878.35260546  ,2000.        ],
    #         [ 4321.28359208 , 2511.67533032  ,2000.        ],
    #         [ 6886.04182764 , 3353.1616541   ,2000.        ],
    #         [12385.89356639 , 7112.86843122  ,2000.        ],
    #         [11065.90886356 , 5036.19977054  ,2000.        ],
    #         [10214.17312873 , 3885.18706779  ,2000.        ],
    #         [ 3280.99859883 ,  672.40107673  ,2000.        ],
    #         [10222.09499517 , 2922.45958431  ,2000.        ],
    #         [ 3688.67067126 , 6667.29793658  ,2000.        ],
    #         [ 5348.96450971 , 7985.99971339  ,2000.        ],
    #         [ 4791.45512612 , 7976.31462321  ,2000.        ],
    #         [ 4586.03381588 , 3880.21916045  ,2000.        ],
    #         [ 2035.08895372 , 4724.44871408  ,2000.        ],
    #         [ 6193.47496205 , 3020.03695658  ,2000.        ],
    #         [ 6625.89944532 , 6705.56955045  ,2000.        ],
    #         [ 9141.86418116 , 8280.19290297  ,2000.        ],
    #         [ 2061.34773299 , 7639.48760113  ,2000.        ],
    #         [ 2094.63983458 , 1865.18095454  ,2000.        ],
    #         [ 3289.80512123 , 6397.08940158  ,2000.        ],
    #         [ 1663.17006065 , 2417.13863163  ,2000.        ],
    #         [ 1750.85485309 , 2463.40191922  ,2000.        ],
    #         [ 5322.25325236 , 2361.01461692  ,2000.        ],
    #         [ 8429.04370041 , 1179.08907689  ,2000.        ]])

    path = np.array([[    0.,             0. ,         2000.        ],
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
        # detection_coverage = 1 - (np.prod(1-detection_coverage, axis=2))
    drone = Drone(path, num_timesteps=300)
    # print(benchmark(drone, terrain, prior, pixel_size))
    # 113.17235490908399
    # 88
    detection_coverage = drone.total_coverage(terrain, pixel_size)
    detection_coverage = np.max(detection_coverage, axis=2)
    print(np.sum(detection_coverage))

    plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='bone')
    # plt.scatter([target[0]*pixel_size], [target[1]*pixel_size], c="r", s=50)
    plt.colorbar(label='Detection Probability')
    plt.axis('equal')
    plt.savefig("lawnmower.png")
    plt.show()

    # benchmark(None)