import numpy as np
from matplotlib import pyplot as plt
from common import *

img = plt.imread('../data/quanser.jpg')
T = np.loadtxt('../data/platform_to_camera.txt')
K = np.loadtxt('../data/heli_K.txt')
marker = np.loadtxt('../data/heli_points.txt').T


# 4.2
L = 11.45*1e-2 

X = np.array([[0,0,0],[0,L,0],[L,0,0],[L,L,0]]).T
X = np.vstack((X,np.ones((1,4))))

u,v = project(K,T @ X)
plt.imshow(img)
plt.xlim(220,450)
plt.ylim(540,440)
plt.scatter(u, v, c='red', marker='.', s=150, edgecolors='black')
plt.savefig("screwPlot")
plt.show()

# Task 4.3 - 4.7
T_base_plat = translate(L/2,L/2,0) @ rotate(0,0, np.deg2rad(11.6))
T_hinge_base = translate(0,0, 32.5*1e-2) @ rotate(0,np.deg2rad(28.9),0)
T_arm_hinge = translate(0,0,-5*1e-2) @ rotate(0,0,0)
T_rot_arm = translate(65*1e-2,0,-3*1e-2) @ rotate(0,0,0)

mark_hinge = T @ T_base_plat @ T_hinge_base @ T_arm_hinge @ marker[:,:3]
mark_rotor = T @ T_base_plat @ T_hinge_base @ T_arm_hinge @ T_rot_arm @ marker[:,3:]

u, v = project(K, np.hstack((mark_hinge ,mark_rotor)))

plt.imshow(img)

draw_frame(K, T)
draw_frame(K, T@T_base_plat)
draw_frame(K,T @ T_base_plat @ T_hinge_base)
draw_frame(K,T @ T_base_plat @ T_hinge_base @ T_arm_hinge)
draw_frame(K,T @ T_base_plat @ T_hinge_base @ T_arm_hinge @ T_rot_arm)

plt.scatter(u, v, c='yellow', marker='.', s=150, edgecolors='black')

plt.savefig("FramePlots")





# plt.show()

