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

def cube_corners(centre, arrete):
    coin_1 = [centre[0]-arrete, centre[1] - arrete, centre[2] +arrete]

    coin_2 = [centre[0]-arrete, centre[1] - arrete, centre[2] -arrete]

    coin_3 = [centre[0]-arrete, centre[1] + arrete, centre[2] +arrete]

    coin_4 = [centre[0]-arrete, centre[1] + arrete, centre[2] -arrete]

    coin_5 = [centre[0]+arrete, centre[1] - arrete, centre[2] +arrete]

    coin_6 = [centre[0]+arrete, centre[1] - arrete, centre[2] -arrete]

    coin_7 = [centre[0]+arrete, centre[1] + arrete, centre[2] +arrete]

    coin_8 = [centre[0]+arrete, centre[1] + arrete, centre[2] -arrete]

    return [[coin_1], [coin_2], [coin_3], [coin_4], [coin_5], [coin_6], [coin_7], [coin_8]]


def init():
    for pt in pts:
        pt.set_data([], [])
        pt.set_3d_properties([])
    return pts

def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    count = 0
    for pt in pts:
        values = liste_sols[count]
        x = values[i,0]
        y = values[i,1]
        z = values[i,2]
        pt.set_data(x, y)
        pt.set_3d_properties(z)
        count += 1
    #truc.set_data(i/10, 0 ,0)
    fig.canvas.draw()
    return pts


if __name__ == "__main__":

    sigma, rho, beta = 10, 28, 8/3

    t = np.linspace(1, 100, 10001)
    y0 = [1,1,1]

    sol = generate_data(y0, t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    centre = [-4.108202576766243119e+00, -7.212142429485874473e+00, 2.024074651651810086e+01]
    arrete = 8
    epaisseur = 1.5
    valid_matrix = points_in_circle(sol, centre, arrete, epaisseur)
    liste_sols = []

    for i in range(0, valid_matrix.shape[0]):
        point = [valid_matrix[i,0],valid_matrix[i,1],valid_matrix[i,2]]
        lasol = generate_data(point,t)
        liste_sols.append(lasol)
        print(100*i//valid_matrix.shape[0])

    pts = sum([ax.plot([], [], [], 'ro', markersize = 1)for i in range(0,valid_matrix.shape[0])], [])
    #truc = ax.plot(0,0,0)

    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.plot(sol[:,0] , sol[:,1], sol[:,2], lw = 0.1)
    #ax.scatter(cube_corners(centre,arrete))
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=30, blit=False)
    fig.canvas.draw()

    plt.show()
