import numpy as np
import numpy.linalg as nl


def power_method(matA:np.ndarray, vecX:np.ndarray=None, epsilon:float=1e-7, n_iter_max:int=100000):
    n = matA.shape[0]
    
    if vecX is None:
        vecX = np.ones(n)

    for i in range(n_iter_max):

        vecY = matA @ vecX
        lam = abs(vecY).max()
        vecY *= 1.0 / lam

        norm = nl.norm(vecX - vecY)
        if norm < epsilon:
            break

        vecX = vecY


    return lam, vecY, i
