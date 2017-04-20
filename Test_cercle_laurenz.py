__author__ = 'Charles'

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


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

def isinbox(position_point, centre_boite, arrete_boite, epaisseur):
    x = position_point[0]
    y = position_point[1]
    z = position_point[2]

    xb = centre_boite[0]
    yb = centre_boite[1]
    zb = centre_boite[2]

    valide = False
    if x > xb - arrete_boite and x < xb + arrete_boite:
        if y > yb - arrete_boite and y < yb + arrete_boite:
            if z > zb - arrete_boite and z < zb + arrete_boite:
                valide = True

    if x > xb - arrete_boite + epaisseur and x < xb + arrete_boite - epaisseur:
        if y > yb - arrete_boite + epaisseur and y < yb + arrete_boite - epaisseur:
            if z > zb - arrete_boite + epaisseur and z < zb + arrete_boite - epaisseur:
                valide = False

    return valide

def isincircle(position_point, centre_cercle, rayon, epaisseur):
    x = position_point[0]
    y = position_point[1]
    z = position_point[2]

    xb = centre_cercle[0]
    yb = centre_cercle[1]
    zb = centre_cercle[2]
    valide = False

    if (np.abs(x-xb)**2 + np.abs(y-yb)**2 + np.abs(z-zb)**2) < rayon:
        valide = True

    if (np.abs(x-xb)**2 + np.abs(y-yb)**2 + np.abs(z-zb)**2) < rayon - epaisseur:
        valide = False

    return valide

def points_in_box(mat, centre, arrete, epaisseur):
        thresh_matrix = np.zeros((1, 3))
        for i in range(0, mat.shape[0]):
            point = [mat[i,0] , mat[i,1], mat[i,2]]
            if isinbox(point, centre, arrete, epaisseur):
                thresh_matrix = np.vstack((thresh_matrix, point))
        thresh_matrix = thresh_matrix[1:,:]

        return thresh_matrix

def points_in_circle(mat, centre, rayon, epaisseur):
        thresh_matrix = np.zeros((1, 3))
        for i in range(0, mat.shape[0]):
            point = [mat[i,0] , mat[i,1], mat[i,2]]
            if isincircle(point, centre, rayon, epaisseur):
                thresh_matrix = np.vstack((thresh_matrix, point))
        thresh_matrix = thresh_matrix[1:,:]

        return thresh_matrix

def init():
    for pt in pts:
        pt.set_data([], [])
        pt.set_3d_properties([])
    return pts

def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    count = 0
    wait_time = 50
    if i <= wait_time and i > 1:
            return pts
    if i == 1:
        wait_time = 0
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
    fig.canvas.draw()
    return pts


if __name__ == "__main__":

    sigma, rho, beta = 10, 28, 8/3

    t = np.linspace(1, 1000, 100001)
    y0 = [1,1,1]
    t2 = np.linspace(1,100,100001)

    sol = generate_data(y0, t)
    sol2 = generate_data(y0, t2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
    valid_matrix = points_in_circle(sol, centre, arrete, epaisseur)
    liste_sols = []
    liste_ind = []
    last = 0
    for i in range(0, valid_matrix.shape[0]):
        point = [valid_matrix[i,0],valid_matrix[i,1],valid_matrix[i,2]]
        liste_ind.append(np.where(sol == point)[0])

        pourcen = 10*i//valid_matrix.shape[0]
        if (pourcen != last):
            print(pourcen)
            last = pourcen


    pts = sum([ax.plot([], [], [], 'ro', markersize = 1)for i in range(0,valid_matrix.shape[0])], [])


    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.plot(sol2[:,0] , sol2[:,1], sol2[:,2], lw = 0.1)

    #anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                           frames=2000, interval=30, blit=True)
    #anim.save('ejection.mp4', fps=30, writer = "ffmpeg")

    for i in np.arange(begin*espace,nombre*espace,espace+1):
         count = 0
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_zlabel("Z")
    plt.title("Initial : {}, espacement : {}, nombre : {}, begin : {} ".format(centre,espace,nombre, begin), y = 1.08)
    plt.show()
