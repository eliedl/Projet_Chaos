
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from utils import*

def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt



#Génère l'animation 2d de l'évolution des conditions initiales
def animate2d(matrix1, matrix2):

    #Fonction qui construit la "frame", retourne une image pour un certain temps "step" donné.
    def ani(step):
        i = 3*step
        line.set_data(sol[:i,0],sol[:i,1])
        trainee.set_data(sol[i-10:i+1,0],sol[i-10:i+1,1])

        line_1.set_data(sol_1[:i,0],sol_1[:i,1])
        trainee_1.set_data(sol_1[i-10:i+1,0],sol_1[i-10:i+1,1])

        point.set_data(Acc_11[i],Acc_12[i])
        point_1.set_data(Acc_21[i],Acc_22[i])
        return point, line, point_1, line_1

    #On initialise les composantes x y des deux trajectoires
    sol = matrix1
    sol_1 = matrix2
    Acc_11 = sol[:,0]
    Acc_12 = sol[:,1]
    Acc_21 = sol_1[:,0]
    Acc_22 = sol_1[:,1]

    #On initialise la figure
    fig = plt.figure(figsize = (5,5))
    axes = fig.add_subplot(111)
    axes.set_xlim(min(Acc_11), max(Acc_11))
    axes.set_ylim(min(Acc_12), max(Acc_12))

    #On initialise les points et les lignes
    point, = axes.plot([Acc_11[0]],[Acc_12[0]], 'bo')
    point_1, = axes.plot([Acc_21[0]],[Acc_22[0]], 'ro')
    line, = axes.plot([],[], lw=0.3, color ='blue')
    line_1, = axes.plot([],[], lw=0.3, color ='red')

    #On initialise les trainées
    trainee, = axes.plot([],[], lw=1, color ='blue')
    trainee_1, = axes.plot([],[], lw=1, color ='red')

    #On active l'animations et on la montre à l'écran
    ani = FuncAnimation(fig, ani, frames=5000, interval=15)

    plt.show()

if __name__ == "__main__":
    #Initialisation des conditions initiales
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    y0 = [0.5, 0.5, 0.5]
    y1 = [1, 0.5, 0.5]
    #calcul des  trajectoires par résolution d'équations différentielles
    ssol = odeint(pend, y0, t, args=(sigma, rho, beta))
    ssol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))

    #On génère et fait jouer l'animation
    animate2d(ssol, ssol_1)