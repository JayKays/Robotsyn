import numpy as np
from estimate_E import *
from epipolar_distance import *
from F_from_E import *

def estimate_E_ransac(xy1, xy2, K, distance_threshold = 4, num_trials = 20000, load = False):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    if load:
        print("Loaded E and inliers from file")
        return np.loadtxt("E_ransac.txt"), np.loadtxt("inliers.txt")
    uv1 = K@xy1
    uv2 = K@xy2
    max_inliers = -np.inf
    best_E = None
    inliers = None
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E = estimate_E(xy1[:,sample], xy2[:,sample])
        F = F_from_E(E,K)
        e = epipolar_distance(F,uv1,uv2)

        num_inliers = np.count_nonzero(np.abs(e) < distance_threshold)

        if num_inliers > max_inliers:
            best_E = E
            max_inliers = num_inliers
            inliers = np.where(np.abs(e) < distance_threshold)
            print(f"New best E found! Number of inliers: {max_inliers}")
        if not i % 100:
            print(f"Iteration: {i} \t inliers: {max_inliers}")

    return best_E, inliers # Placeholder, replace with your implementation


if __name__ == '__main__':

    K = np.loadtxt('../data/K.txt')
    matches = np.loadtxt('../data/task4matches.txt') # Part 4
    
    uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
    uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])
    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    p = 0.5
    P = 0.99
    k = 8

    S = int(np.log(1-P)/np.log(1-p**k))
    print(f"S = {S}")
    print(f"Total number of matches: {uv1.shape[1]}")

    E, inliers = estimate_E_ransac(xy1, xy2, K, num_trials = S)

    inliers_stored = np.loadtxt("inliers.txt")

    if inliers[0].shape[0] > inliers_stored.shape[0]:
        print(f"Found new better E, saving to file with {inliers.shape[0]} inliers")

        f = open("E_ransac.txt", 'w')
        np.savetxt(f,E)
        f.close()

        f = open("inliers.txt", 'w')
        np.savetxt(f,inliers)
        f.close()

