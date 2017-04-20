
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from utils import *

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


if __name__ == "__main__":
#initialisation de l'attracteur
    sigma, rho, beta = 10, 28, 8/3

#types de matrices avec limitations sur les conditions à prendre en compte
    lam_mat =lambda Y,Z:([[- sigma, (rho + sigma -Z)/2, Y/2]
                        , [(rho + sigma -Z)/2, -1, 0]
                        ,[ Y/2, 0, -beta]])
    lam_mat =lambda Y,Z:([[- sigma, (rho + sigma -Z)/2, 0]
                        , [(rho + sigma -Z)/2, -1, 0]
                        ,[ 0, 0, -beta]])
    t = np.linspace(1, 1000, 100001)
    y0 = [1.547679936204874984e+00, 2.123422213176072049e+00, 2.018664314318483122e+01]
    initial_values = generate_data(y0,t)
    thresh_matrix = np.zeros((1, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#résolution -> nombre de points simulés
    resolution = 10000
#résolutino des équations aux valeurs propres et sélection des lambdas
    for i in range(0,resolution,1):
        pos = [initial_values[i,0], initial_values[i,1], initial_values[i,2]]
        Y = 0
        Z = pos[2]
        w = np.linalg.eigvalsh(np.array(lam_mat(Y,Z)))
        liapunov = max(w)
        lam3 = (-(sigma+1) + np.sqrt((rho + sigma - Z)**2 + (sigma+1)**2))/2
        liapunov = lam3
        local = np.array([pos[0], pos[1], pos[2], liapunov])
        thresh_matrix = np.vstack((thresh_matrix, local))
        print((i * 100)//resolution)

#construction des matrices composant la figure
    xs = thresh_matrix[1:,0]
    ys = thresh_matrix[1:,1]
    zs = thresh_matrix[1:,2]
    c = thresh_matrix[1:,3]


    print(c)

    np.savetxt("last_liapunov", thresh_matrix)

    p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
    h = fig.colorbar(p)
    h.set_label(r'$\lambda_3(z)$')
    fig.canvas.draw()
    plt.show()

