__author__ = 'Charles'
__author__ = 'Charles'
__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import projet_core
from utils import*

#initialisation des équations différentielles pour l'attracteur de Rossler
def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]


def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt


if __name__ == "__main__":
    #on commence par initialiser l'attracteur
    sigma, rho, beta = 10, 28, 8/3
    #on définis l'espace temporel
    t = np.linspace(1, 100, 10001)
    #On définit la magnitude de perturbation
    espacement = 1e-2
    #on définit les conditions initiales avec et sans perturbations
    y0 = [1+ espacement, 1+ espacement, 1+ espacement]
    y1 = [1,1,1]

    # on résoud les équations différentielles de l'attracteur de Rossler
    core_1 = projet_core.Core()
    core_1.coordinates = np.array([[1, 1, 1], [1+ espacement, 1+ espacement, 1+ espacement]])
    core_1.attractor = 'Rössler'
    core_1.params = np.array([[0.2, 0.2, 14]])
    core_1.t = np.linspace(1, 200, 10001)
    core_1.solve_edo()

    #Les solutions sont gardées en mémoire
    sol =  core_1.time_series[:,:3]
    sol_1 =  core_1.time_series[:,3:]

    #On calcule la similarité cosinus des deux trajectoires résolues
    sim =np.degrees(np.arccos(cosine_similarity(sol,sol_1)))

    #On imprime le graphique après avoir transformé l'indice du temps de corrélation en temps de corrélation.
    plt.plot(t[:-1], sim)
    plt.title("conditions initiales: {} espacement: {}".format(y1, espacement))
    count = 0
    threshhold = 0
    for i in sim:
        if i >= 90:
            threshhold = t[count]
            break
        count += 1
    print("Temps de corrélation : ", threshhold," s")
    plt.show()
    print("done")