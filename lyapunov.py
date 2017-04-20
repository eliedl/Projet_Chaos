import numpy as np
from math import log
from projet_core import Core
import matplotlib.pyplot as plt
from scipy import stats

def delta(array, i, j):
    return abs(array[i] - array[j])

def calculate_liapunov(time_series, eps):
    N = len(time_series)
    liap_list = [[] for i in range(N)]
    result = np.zeros(N)
    n = 0
    for i in range(N):
        for j in range(i+1, N):
            if delta(time_series, i, j) < eps:
                n += 1
                for k in range(min(N-i, N-j)):
                    liap_list[k].append(log(delta(time_series, i+k, j+k)))

    for i in range(N):
        if len(liap_list[i]):
           result[i] = sum(liap_list[i])/len(liap_list[i])

    return result

if __name__ == '__main__':
    core = Core()
    core.coordinates = np.array([[1, 1, 1], [1.+1e-1, 1+1e-1, 1+1e-1]])
    core.attractor = 'Rössler'
    core.params = np.array([[0.2, 0.2, 14]])
    core.t = np.linspace(1, 100, 10001)
    core.solve_edo()

    r = np.sqrt(core.time_series[:, 0]**2 + core.time_series[:, 1]**2 + core.time_series[:, 2]**2)

    L_exp = calculate_liapunov(r, 0.00001)
    np.savetxt('liapunov_r_Rössler.txt', L_exp)
