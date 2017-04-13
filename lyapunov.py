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
    core.attractor = 'Lorenz'
    core.params = np.array([[10, 28, 8/3]])
    core.t = np.linspace(1, 100, 10001)
    core.solve_edo()

    L_exp = calculate_liapunov(core.time_series[:, 0], 0.00001)
    np.savetxt('liapunov.txt', L_exp)

    k = np.arange(0, 30)
    dk = L_exp[:30]

    slope, intercept, r_value, p_value, std_err = stats.linregress(k[:5], dk[: 5])
    line = slope * k[:5]
    print(slope)

    plt.plot(k[:5], line)


    plt.plot(L_exp[:30])
    plt.show()