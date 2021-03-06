import numpy as np
import matplotlib.pyplot as plt

#
# Tip: Define functions to create the basic 4x4 transformations
#
# def translate_x(x): Translation along X-axis
# def translate_y(y): Translation along Y-axis
# def translate_z(z): Translation along Z-axis
# def rotate_x(radians): Rotation about X-axis
# def rotate_y(radians): Rotation about Y-axis
# def rotate_z(radians): Rotation about Z-axis
#
# For example:
def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


#
# Note that you should use np.array, not np.matrix,
# as the latter can cause some unintuitive behavior.
#
# translate_x/y/z could alternatively be combined into
# a single function.

def rotate(x,y,z):
    Rx = np.array(
        [[1,0,0], [0, np.cos(x), -np.sin(x)],[0, np.sin(x), np.cos(x)]]
    )

    Ry = np.array(
        [[np.cos(y), 0, np.sin(y)], [0,1,0],[-np.sin(y), 0, np.cos(y)]]
    )

    Rz = np.array(
        [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0],[0,0,1]]
    )

    R = Rx @ Ry @ Rz

    return np.vstack((np.hstack((R, np.zeros((3,1)))), [0,0,0,1]))

def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """

    # Tip: Use the @ operator for matrix multiplication, the *
    # operator on arrays performs element-wise multiplication!

    #
    # Placeholder code (replace with your implementation)

    P = np.hstack((np.eye(3), np.zeros((3,1))))

    if X.shape[0] == 3:
        hom = K @ X
    elif X.shape[0] == 4:
        hom = K @ P @ X

    uv = hom[:-1][:] / hom[-1][:]
    return uv

def draw_frame(K, T, scale=0.05):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    This uses your project function, so implement it first.

    Control the length of the axes using 'scale'.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X) # If you get an error message here, you should modify your project function to accept 4xN arrays of homogeneous vectors, instead of 3xN.
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red', linewidth = 2.5) # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green',linewidth = 2.5) # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue',linewidth = 2.5) # Z-axis
