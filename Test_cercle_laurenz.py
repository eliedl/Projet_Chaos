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
    #centre = [-12, -9, 34] #spirale
    #centre = [-5, -7, 13] # en bas du centre
    centre = [0, 0, 20] #ejection

    arrete = 10
    epaisseur = 3
    valid_matrix = points_in_circle(sol, centre, arrete, epaisseur)
    liste_sols = []
    liste_ind = []

    for i in range(0, valid_matrix.shape[0]):
        point = [valid_matrix[i,0],valid_matrix[i,1],valid_matrix[i,2]]
        liste_ind.append(np.where(sol == point)[0])
        #lasol = generate_data(point,t)
        #liste_sols.append(lasol)
        print(100*i//valid_matrix.shape[0])

    pts = sum([ax.plot([], [], [], 'ro', markersize = 1)for i in range(0,valid_matrix.shape[0])], [])
    #truc = ax.plot(0,0,0)

    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))
    ax.plot(sol2[:,0] , sol2[:,1], sol2[:,2], lw = 0.1)
    #ax.scatter(cube_corners(centre,arrete))
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=30, blit=True)
    anim.save('ejection.mp4', fps=30, writer = "ffmpeg")
    fig.canvas.draw()

    plt.show()
