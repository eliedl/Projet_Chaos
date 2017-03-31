__author__ = 'Charles'

import numpy as np
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

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

Acc_11 = sol[:,0]
Acc_12 = sol[:,1]
Acc_13 = sol[:,2]

# Scatter plot
fig = plt.figure(figsize = (5,5))
ax = p3.Axes3D(fig)
ax.set_xlim(min(Acc_11), max(Acc_11))
ax.set_ylim(min(Acc_12), max(Acc_12))

point, = ax.plot([Acc_11[0]],[Acc_12[0]],sol[0,2], 'go')

def ani(coords):
    print(coords)
    point.set_data([coords[0]],[coords[1]],[coords[2]])
    #point.sed_3d_properties([coords[0]],[coords[1]],[coords[2]])
    return point

def frames():
    for acc_11_pos, acc_12_pos, acc_13_pos in zip(Acc_11, Acc_12, Acc_13):
        yield acc_11_pos, acc_12_pos, acc_13_pos

plt.plot(sol[:,0], sol[:,1],sol[:,2],lw = 0.1)

ani = FuncAnimation(fig, ani, frames=frames, interval=10)

plt.show()