import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation

def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt

sigma, rho, beta = 10, 28, 8/3
t = np.linspace(1, 100, 10001)
y0 = [0.5, 0.5, 0.5]
y1 = [1, 0.5, 0.5]

sol = odeint(pend, y0, t, args=(sigma, rho, beta))
sol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))

sol_copy = np.copy(sol[:, 0])
sol_1_copy = np.copy(sol_1[:, 0])
print(np.size(sol[:, 0]) == np.size(sol_copy))
x_n = np.delete(sol[:, 0], -1)
x_nn = np.delete(sol_copy, 0)


x_n_ = np.delete(sol_1[:, 0], -1)
x_nn_ = np.delete(sol_1_copy, 0)

print(len(x_n), len(x_nn))

plt.plot(x_n, x_nn)
plt.plot(x_n_, x_nn_)
plt.show()

plt.plot(t, sol[:, 0], 'b', label='x_0')
plt.plot(t, sol_1[:, 0], 'g', label='x_1')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.plot(x_n, x_n_)
plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
x = sol[:, 0]
y = sol[:, 1]
z = sol[:, 2]
ax.plot(x, y, z, label='Lorentz attractor')
ax.legend()


plt.show()