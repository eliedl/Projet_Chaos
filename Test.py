
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

Acc_11 = sol[:,0]
Acc_12 = sol[:,1]
Acc_21 = sol_1[:,0]
Acc_22 = sol_1[:,1]

# Scatter plot
fig = plt.figure(figsize = (5,5))
axes = fig.add_subplot(111)
axes.set_xlim(min(Acc_11), max(Acc_11))
axes.set_ylim(min(Acc_12), max(Acc_12))

point, = axes.plot([Acc_11[0]],[Acc_12[0]], 'go')
point_1, =axes.plot([Acc_21[0]],[Acc_22[0]], 'go')


def ani(i):
    point.set_data(Acc_11[i],Acc_12[i])
    point_1.set_data(Acc_21[i],Acc_22[i])
    return point




plt.plot(sol[:,0], sol[:,1],lw = 0.1)

ani = FuncAnimation(fig, ani, frames=5000, interval=1)


plt.show()