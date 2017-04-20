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

#calcul du moment ou il y a perte de similarité
def calculate_threshhold(mat, temps):
    count = 0
    for i in mat:
        if i <= 0.5:
            return temps[count]
        count += 1
    return temps[count]

if __name__ == "__main__":
#initialisation de l'attracteur, des conditions initiales et de la figure
    t = np.linspace(1, 100, 10001)
    y0 = [1.5, 2, 2]
    initial_values = generate_data(y0,t)
    thresh_matrix = np.zeros((1, 4))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

#sélection de la résolution (nombre de points à simuler) et de la magnitude de perturbation
    resolution = 5000
    espacement = 1e-8
    for i in range(0,resolution,1):
        mat1 = initial_values[i:]
#simulation des trajectoires perturbées et calcul de la similarité cosinus entre chaque trajectoire et la trajectoire originelle
        y = [initial_values[i,0] +espacement, initial_values[i,1], initial_values[i,2]]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold1 = calculate_threshhold(coeff,t)

        y = [initial_values[i,0] , initial_values[i,1], initial_values[i,2]+espacement]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold2 = calculate_threshhold(coeff,t)

        y = [initial_values[i,0] , initial_values[i,1]+espacement, initial_values[i,2]]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold3 = calculate_threshhold(coeff,t)

        y = [initial_values[i,0] -espacement, initial_values[i,1], initial_values[i,2]]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold4 = calculate_threshhold(coeff,t)

        y = [initial_values[i,0] , initial_values[i,1], initial_values[i,2]-espacement]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold5 = calculate_threshhold(coeff,t)

        y = [initial_values[i,0] , initial_values[i,1]-espacement, initial_values[i,2]]
        mat2 = generate_data(y,t[i:])
        coeff = cosine_similarity(mat1,mat2)
        threshhold6 = calculate_threshhold(coeff,t)

        threshhold = (threshhold1+threshhold2+threshhold3+threshhold4+threshhold5+threshhold6)/6
        local = np.array([y[0], y[1], y[2], threshhold])
        thresh_matrix = np.vstack((thresh_matrix, local))
        #imporession de pourcentage de complétion
        print((i * 100)//resolution)

#construction de la matrice de points finale
    xs = thresh_matrix[50:,0]
    ys = thresh_matrix[50:,1]
    zs = thresh_matrix[50:,2]
    c = thresh_matrix[50:,3]
    count = 0
#sauvegarde de la série de données.
    np.savetxt("-8_cossim_cover.txt", thresh_matrix)
#affichage de la figure.
    p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
    fig.colorbar(p)
    plt.show()