import numpy as np
import numpy.linalg as nl
import numpy.random as nr


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


def get_a_random_matrix(n:int=2) -> np.ndarray:
  matA = nr.randint(1, 10, size=(n,n))
  for p in range(n):
      matA[p, p] = nr.randint(10, 20)
  matA = (matA * matA.T) ** 0.5
  return matA


def main():
  matA = get_a_random_matrix()

  print(f"matA =\n{matA}")

  lam, vecX, n = power_method(matA)

  print(f'lam = {lam}')
  print(f'vecX = {vecX}')
  print(f'counter = {n}')

  Ax = matA @ vecX
  print(f"Ax = matA @ vecX {Ax}")
  print(f"Ax/Ax.max() = {Ax / Ax.max()}")


if "__main__" == __name__:
  main()
