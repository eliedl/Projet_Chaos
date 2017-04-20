import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class Core:
    def __init__(self):
        self.attractor   = ''
        self.coordinates = []
        self.params      = np.array([])
        self.time_series = np.array([0, 0, 0])
        self.t           = np.array([])

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

    @staticmethod
    def unknown(l, t, sigma, rho, beta):
        x, y, z = l

        dldt = [x*y - x**2, -sigma*x**2 + sigma*z, -y*z + x**2 - rho*z]
        return dldt

    @staticmethod
    def  fabinovich_fabrikant(l, t, sigma, rho, beta):
        x, y, z = l

        dldt = [y*(z - 1 + x**2) + rho*x, x*(3*z + 1 - x**2) + rho*y, -2*z*(sigma + x*y)]
        return dldt

    def solve_edo(self):
        if self.attractor == 'Lorenz':
            func = self.lorentz
        elif self.attractor == 'Rössler':
            func = self.roessler
        elif self.attractor == 'Unknown':
            func = self.unknown
        elif self.attractor == 'Fabinovich-Fabrikant':
            func = self.fabinovich_fabrikant

        for i in range(np.shape(self.coordinates)[0]):
            sol = odeint(func, self.coordinates[i, :], self.t,
                         args=(self.params[0, 0], self.params[0, 1], self.params[0, 2]))

            if i == 0:
                self.time_series = sol
            else:
                self.time_series = np.hstack((self.time_series, sol))

if __name__ == '__main__':
    core_1 = Core()
    core_1.coordinates = np.array([[0.1, 0.1, 0.1]])
    core_1.attractor = 'Fabinovich-Fabrikant'
    core_1.params = np.array([[ 0.1, 0.2715, 8/3]])
    core_1.t = np.linspace(1, 100, 100001)
    core_1.solve_edo()

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = core_1.time_series[:, 0]
    y = core_1.time_series[:, 1]
    z = core_1.time_series[:, 2]

    ax.plot(x, y, z, marker = '.', ls= 'none')
    ax.legend()
    plt.show()
