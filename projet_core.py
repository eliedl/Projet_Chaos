import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class Core:
    def __init__(self):
        self.attractor = ''
        self.coordinates = []
        self.params = np.array([])
        self.time_series = np.array([0, 0, 0])
        self.t = np.array([])

    @staticmethod
    def lorentz(l, t, sigma ,rho, beta):

        x, y, z  = l

        dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
        return dldt

    @staticmethod
    def roessler(l, t, sigma, rho, beta):
        x, y, z = l

        dldt = [-y - z, x + sigma*y, rho + z*(x - beta)]
        return dldt


    def solve_edo(self):
        if self.attractor == 'Lorentz':
            for i in range(np.shape(self.coordinates)[0]):

                sol = odeint(self.lorentz, self.coordinates[i, :], self.t,
                             args=(self.params[0, 0], self.params[0, 1], self.params[0, 2]))

                if i == 0:
                    self.time_series = sol
                else:
                    self.time_series = np.hstack((self.time_series, sol))

        if self.attractor == 'Roessler':
            for i in range(np.shape(self.coordinates)[0]):

                sol = odeint(self.roessler, self.coordinates[i, :], self.t,
                             args=(self.params[0, 0], self.params[0, 1], self.params[0, 2]))
                if i == 0:
                    self.time_series = sol
                else:
                    self.time_series = np.hstack((self.time_series, sol))

if __name__ == '__main__':
    core_1 = Core()
    core_1.coordinates = np.array([[1, 1, 1], [1+1e-5, 1+1e-5, 1+1e-5]])
    core_1.attractor = 'Roessler'
    core_1.params = np.array([[0.2, 0.2, 5.7]])
    core_1.t = np.linspace(1, 100, 10001)
    core_1.solve_edo()

    print(np.shape(core_1.t))
    print(np.shape(core_1.time_series))
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #x = core_1.time_series[:, 0]
    #y = core_1.time_series[:, 1]
    #z = core_1.time_series[:, 2]

    x = core_1.time_series[:, 0] - core_1.time_series[:, 3]
    y = core_1.time_series[:, 1] - core_1.time_series[:, 4]
    z = core_1.time_series[:, 2] - core_1.time_series[:, 5]

    r = np.sqrt(x**2 + y**2 + z**2)
    ax.plot(x, y, z, label='Lorentz attractor', marker = '.', markersize = 2, ls='none')
    ax.legend()
    plt.show()

    plt.plot(core_1.t, r)
    plt.show()