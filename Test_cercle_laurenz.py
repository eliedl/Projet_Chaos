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

def generate_data(y0, t):
    sigma, rho, beta = 10, 28, 8/3
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))

    return sol
#initialisation des points pour l'animation
def init():
    for pt in pts:
        pt.set_data([], [])
        pt.set_3d_properties([])
    return pts

#Fonction qui construit la "frame", retourne une image pour un certain temps "step" donné.
def animate(i):
    count = 0
    #on attend un peu avant de commencer l'animation
    wait_time = 50
    if i <= wait_time and i > 1:
            return pts
    if i == 1:
        wait_time = 0
    #on déplace chaque point pour chaque frame
    for pt in pts:
        ind = liste_ind[count]
        ind = ind[0]
        if (i + ind) >= (sol.shape[0]):
            pt.set_data(100, 100)
            pt.set_3d_properties(100)
            count += 1
            continue
        j = i - wait_time
        x = sol[j+ind,0]
        y = sol[j+ind,1]
        z = sol[j+ind,2]
        pt.set_data(x, y)
        pt.set_3d_properties(z)
        count += 1
    #on dessine les points déplacés
    fig.canvas.draw()
    return pts


if __name__ == "__main__":
    #initialisation de l'Attracteur
    sigma, rho, beta = 10, 28, 8/3

    t = np.linspace(1, 1000, 100001)
    y0 = [1,1,1]
    t2 = np.linspace(1,100,100001)
    #Résolution des equations différentielles
    sol = generate_data(y0, t)
    sol2 = generate_data(y0, t2)
    #initialisation des figures
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

#Différentes conditions initiales générant des figures intéressantes

    #centre = [-17, -10, 42] #top
    #centre = [-17, -10, 43] #demi-cercle top
    #centre = [-8, -8, 27] #centre
    #centre = [-10, -9, 32] #en haut du centre
    #centre = [-12, -9, 34] #spirale
    #centre = [-5, -7, 13] # en bas du centre
    centre = [0, 0, 20] #ejection

#paramètres des step-sims
    nombre = 4
    espace = 13
    begin = 0
#paramètres du cercle
    arrete = 10
    epaisseur = 3

#des conditions initiales possibles, on isole les points d'intéret (dans l'ensemble)
    valid_matrix = points_in_circle(sol, centre, arrete, epaisseur)
    liste_sols = []
    liste_ind = []
    last = 0

#on trouve les indices dans la matrice initiale des points trouvés dans l'ensemble
    for i in range(0, valid_matrix.shape[0]):
        point = [valid_matrix[i,0],valid_matrix[i,1],valid_matrix[i,2]]
        liste_ind.append(np.where(sol == point)[0])

        #imporession du pourcentage de complétion 1-10
        pourcen = 10*i//valid_matrix.shape[0]
        if (pourcen != last):
            print(pourcen)
            last = pourcen

    #initialisation des différents points
    pts = sum([ax.plot([], [], [], 'ro', markersize = 1)for i in range(0,valid_matrix.shape[0])], [])

#mise en forme de la figure
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.plot(sol2[:,0] , sol2[:,1], sol2[:,2], lw = 0.1)

#animation et sauvegarde

    #anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                           frames=2000, interval=30, blit=True)
    #anim.save('ejection.mp4', fps=30, writer = "ffmpeg")

#Génération de figure de type "freezeframe", pour présenter des animations dans un article.
freezeframe = True
if freezeframe:
    for i in np.arange(begin*espace,nombre*espace,espace+1):
         count = 0
         #imporession du pourcentage de complétion (10- 100)
         print(90*i//(nombre*espace) + 10)
         for pt in pts:
            ind = liste_ind[count]
            ind = ind[0]
            if (i + ind) >= (sol.shape[0]):
                pt.set_data(100, 100)
                pt.set_3d_properties(100)
                count += 1
                continue
            j = i
            x = sol[j+ind,0]
            y = sol[j+ind,1]
            z = sol[j+ind,2]
            ax.scatter(x, y, z, c = "red", s = 0.21)
            count += 1

    print(100)
#mise en forme des axes de la figure et affichage de celle-ci
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_zlabel("Z")
    plt.title("Initial : {}, espacement : {}, nombre : {}, begin : {} ".format(centre,espace,nombre, begin), y = 1.08)
    plt.show()
