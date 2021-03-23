import matplotlib.pyplot as plt
import numpy as np
from figures import *
from estimate_E import *
from decompose_E import *
from triangulate_many import *
from epipolar_distance import *
from F_from_E import *
from estimate_E_ransac import *

K = np.loadtxt('../data/K.txt')
I1 = plt.imread('../data/image1.jpg')/255.0
I2 = plt.imread('../data/image2.jpg')/255.0
matches = np.loadtxt('../data/matches.txt')
matches = np.loadtxt('../data/task4matches.txt') # Part 4

uv1 = np.vstack([matches[:,:2].T, np.ones(matches.shape[0])])
uv2 = np.vstack([matches[:,2:4].T, np.ones(matches.shape[0])])
xy1 = np.linalg.inv(K) @ uv1
xy2 = np.linalg.inv(K) @ uv2


# Task 2: Estimate E
# E = estimate_E(xy1, xy2)
E, inliers = estimate_E_ransac(xy1, xy2, K, load = True)
# Task 3: Triangulate 3D points
inliers = inliers.astype(int)
uv1 = uv1[:,inliers]
uv2 = uv2[:,inliers]
xy1 = xy1[:,inliers]
xy2 = xy2[:,inliers]

xy01 = np.vstack((xy1[:2,:], np.zeros(xy1.shape[1]), xy1[2,:]))
xy02 = np.vstack((xy2[:2,:], np.zeros(xy2.shape[1]), xy2[2,:]))

P11,P12,P21,P22 = decompose_E(E)
P1 = np.hstack((np.eye(3), np.zeros((3,1))))
P2 = choose_pose([P11,P12,P21,P22], xy01, xy02)

X = triangulate_many(xy1, xy2, P1, P2) 

# Uncomment in Task 2
np.random.seed(123) # Leave as commented out to get a random selection each time
draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K), sample_size=8)
plt.savefig("Ransac_corr")
#
# Uncomment in Task 3
#
draw_point_cloud(X, I1, uv1, xlim=[-1,+1], ylim=[-1,+1], zlim=[1,3])
plt.savefig("Ransac_cloud")
# plt.hist(epipolar_distance(F_from_E(E,K), uv1, uv2), bins = 100)
# plt.savefig("task41")
plt.show()
