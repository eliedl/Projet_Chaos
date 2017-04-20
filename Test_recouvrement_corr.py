__author__ = 'Charles'
__author__ = 'Charles'
__author__ = 'Charles'
__author__ = 'Charles'
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

def fit(mat,length):

    from scipy.optimize import curve_fit
    corr = autocorr(mat,10,length)
    y = np.linspace(10,length/100,length-10)
    popt, pcov = curve_fit(func, y, corr)
    return y, corr, popt, pcov

def func(x,a,b,c):
        return a*np.exp(-b*x)

if __name__ == "__main__":
#initialisation de l'atracteur, des conditions initiales et de la figure
    t = np.linspace(1, 100, 10001)
    y0 = [1.547679936204874984e+00, 2.123422213176072049e+00, 2.018664314318483122e+01]
    initial_values = generate_data(y0,t)
    thresh_matrix = np.zeros((1, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

#calcul de l'autocorrélation, du fit et prise en mémoire du temps de similarité
#pour un nombre "résolution" de points.
    resolution = 5000
    for i in range(0,resolution,5):
        mat1 = initial_values[i:]
        y = [initial_values[i,0], initial_values[i,1], initial_values[i,2]]

        garbage, corr, popt, pcov = fit(mat1[:,2],50)

        threshhold = -1/popt[1]
        local = np.array([y[0], y[1], y[2], threshhold])
        thresh_matrix = np.vstack((thresh_matrix, local))
        print((i * 100)//resolution)
#construction des matrices de la figure
    xs = thresh_matrix[1:,0]
    ys = thresh_matrix[1:,1]
    zs = thresh_matrix[1:,2]
    c = thresh_matrix[1:,3]
    count = 0

    print(c)

    np.savetxt("last_cossim_cover.txt", thresh_matrix)

    p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
    fig.colorbar(p)
    plt.show()