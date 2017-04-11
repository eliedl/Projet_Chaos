import numpy as np
from math import log
from projet_core import Core

def delta(array, i, j):
    return abs(array[i] - array[j])

def calculate_liapunov(time_series, eps):
    N = len(time_series)
    liap_list = [[] for i in range(N)]
    n = 0
    for i in range(N):
        print(i // N *100)
        for j in range(i+1, N):
            if delta(time_series, i, j) < eps:
                n += 1
      é          for k in range(min(N-i, N-j)):
                    liap_list[k].append(log(delta(time_series, i+k, j+k)))

    for i in range(len(liap_list)):
        if len(liap_list[i]):
            print>>sum(liap_list[i])/len(liap_list[i])


if __name__ == '__main__':
    core = Core()
    core.coordinates = np.array([[1, 1, 1], [1.+1e-1, 1+1e-1, 1+1e-1]])
    core.attractor = 'Lorenz'
    core.params = np.array([[10, 28, 8/3]])
    core.t = np.linspace(1, 100, 10001)
    core.solve_edo()

    L_exp = calculate_liapunov(core.time_series[:, 0], 0.3)
