import numpy as np

def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))

    for i in range(n):
        x1, y1 = xy1[0,i], xy1[1,i]
        x2, y2 = xy2[0,i], xy2[1,i]
        A[i,:] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    _,_,vt = np.linalg.svd(A)
    return np.reshape(vt.T[:,-1], (3,3))


