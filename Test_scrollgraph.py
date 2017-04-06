
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

ssol = odeint(pend, y0, t, args=(sigma, rho, beta))
ssol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def scrollgraph(fig, ax, matrix1, matrix2):
    sol = matrix1
    sol_1 = matrix2
    diff = np.sqrt(abs((sol_1[:,0] - sol[:,0]))**2 + (sol_1[:,1] - sol[:,1])**2 + (sol_1[:,2] - sol[:,2])**2)
    Acc_11 = sol[:,0]
    Acc_12 = sol[:,1]

    axes = ax
    print(min(diff),max(diff))
    axes.set_ylim(min(diff), max(diff))

    line, = axes.plot([Acc_11[0]],[Acc_12[0]])



    def ani(step):
        i = 3*step
        x = t[:i]
        y = diff[:i]
        line.set_data(x,y)

        axes.set_xlim(t[i]-10, t[i]+1)
        return line




    return  FuncAnimation(fig, ani, frames=5000, interval=50)


if __name__ == "__main__":
    fig = plt.figure()
    axes = fig.add_subplot(111)

    ani = scrollgraph(fig, axes, ssol, ssol_1)

    plt.show()

