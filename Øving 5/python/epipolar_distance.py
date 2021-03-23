import numpy as np

def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    e1 = np.zeros(n) # Placeholder, replace with your implementation
    e2 = np.zeros(n)

    for i in range(n):
        e1_num = uv2[:,i].T @ F @ uv1[:,i]
        e2_num = e1_num.T

        Fu1 = F @ uv1[:,i]
        Ftu2 = F.T @ uv2[:,i]

        e1_denum = np.sqrt(Fu1[0]*Fu1[0] + Fu1[1]*Fu1[1])
        e2_denum = np.sqrt(Ftu2[0]*Ftu2[0] + Ftu2[1]*Ftu2[1])

        e1[i] = e1_num/e1_denum
        e2[i] = e2_num/e2_denum

    e = (e1 + e2)/2

    return e
