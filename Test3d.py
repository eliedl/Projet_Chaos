
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
y1 = [4, 6, 18]

ssol = odeint(pend, y0, t, args=(sigma, rho, beta))
ssol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

def animate3d(fig, axes, matrix1,matrix2):
    sol = matrix1
    sol_1 = matrix2

    Acc_11 = sol[:,0]
    Acc_12 = sol[:,1]
    Acc_13 = sol[:,2]

    Acc_21 = sol_1[:,0]
    Acc_22 = sol_1[:,1]
    Acc_23 = sol_1[:,2]

    # Scatter plot
    #fig = plt.figure(figsize = (5,5))
    # axes = p3.Axes3D(fig)
    axes.set_xlim(min(Acc_11), max(Acc_11))
    axes.set_ylim(min(Acc_12), max(Acc_12))
    axes.set_zlim(min(Acc_13), max(Acc_13))



    point, = axes.plot([Acc_11[0]],[Acc_12[0]], 'bo')
    point_1, =axes.plot([Acc_21[0]],[Acc_22[0]], 'ro')
    line, = axes.plot([],[], lw=0.3, color ='blue')
    line_1, = axes.plot([],[], lw=0.3, color ='red')

    trainee, = axes.plot([],[], lw=1, color ='blue')
    trainee_1, = axes.plot([],[], lw=1, color ='red')

    def ani(step):
        i = 3*step

        line.set_data(sol[:i,0],sol[:i,1])
        line.set_3d_properties(sol[:i,2])
        trainee.set_data(sol[i-10:i+1,0],sol[i-10:i+1,1])
        trainee.set_3d_properties(sol[i-10:i+1,2])

        line_1.set_data(sol_1[:i,0],sol_1[:i,1])
        line_1.set_3d_properties(sol_1[:i,2])
        trainee_1.set_data(sol_1[i-10:i+1,0],sol_1[i-10:i+1,1])
        trainee_1.set_3d_properties(sol_1[i-10:i+1,2])

        point.set_data(Acc_11[i],Acc_12[i])
        point.set_3d_properties(Acc_13[i])

        point_1.set_data(Acc_21[i],Acc_22[i])
        point_1.set_3d_properties(Acc_23[i])
        return point, point_1, line, line_1, trainee, trainee_1



   # plt.plot(sol[:,0], sol[:,1], sol[:,2], '--',lw = 0.3, color = 'lime')

    return FuncAnimation(fig, ani, frames=5000, interval=30, blit= True)


    plt.show()
if __name__ == "__main__":
    fig = plt.figure(figsize = (5,5))
    axes = p3.Axes3D(fig)

    ani = animate3d(fig, axes, ssol, ssol_1)

    plt.show()