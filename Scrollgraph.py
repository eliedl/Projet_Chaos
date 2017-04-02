__author__ = 'Charles'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def scrollgraph(fig, ax, matrix1, matrix2):
    sol = matrix1
    sol_1 = matrix2
    diff = np.sqrt(abs((sol_1[:,0] - sol[:,0]))**2 + (sol_1[:,1] - sol[:,1])**2 + (sol_1[:,2] - sol[:,2])**2)
    Acc_11 = sol[:,0]
    Acc_12 = sol[:,1]

    #fig = plt.figure(figsize = (5,5))
    #axes = fig.add_subplot(111)
    axes = ax
    axes.set_ylim(min(diff), max(diff))

    line, = axes.plot([Acc_11[0]],[Acc_12[0]])



    def ani(step):
        i = 3*step

        x = t[:i]
        y = diff[:i]
        line.set_data(x,y)

        axes.set_xlim(t[i]-10, t[i]+1)
        return line




    ani = FuncAnimation(fig, ani, frames=5000, interval=25)
