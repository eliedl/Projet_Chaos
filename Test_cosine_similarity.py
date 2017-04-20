__author__ = 'Charles'
__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from utils import*


def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt


if __name__ == "__main__":
#initialisation de l'Attracteur
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)

#génération des conditions initiales
    espacement = 1e-1
    x=1
    y0 = [1+ espacement, 1+ espacement, 1+ espacement]
    y1 = [1,1,1]
#résolution des EDOS
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    sol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))
#calcul de la similarité cosinus
    sim =np.degrees(np.arccos(cosine_similarity(sol,sol_1)))
    plt.plot(t[:-1], sim)
#On calcule de temps de similarité à l'aide d'un threshhold prédéterminé
    count = 0
    threshhold = 0
    for i in sim:
        if i >= 90:
            threshhold = t[count]
            break
        count += 1

    plt.title("conditions initiales: {} espacement: {}".format(y1, espacement))

    print("Temps de corrélation : ", threshhold," s")
    plt.show()
    print("done")