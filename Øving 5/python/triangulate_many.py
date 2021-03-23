import numpy as np

def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4,n)) # Placeholder, replace with your implementation

    for i in range(n):
        x1, y1 = xy1[0,i], xy1[1,i]
        x2, y2 = xy2[0,i], xy2[1,i]

        A = np.vstack((x1*P1[2,:] - P1[0,:], y1*P1[2,:] - P1[1,:], \
            x2*P2[2,:] - P2[0,:], y2*P2[2,:] - P2[1,:]))
        
        _,_,vt = np.linalg.svd(A)

        X[:,i] = vt.T[:,-1]
    X /= X[-1]
    return X # Placeholder, replace with your implementation
