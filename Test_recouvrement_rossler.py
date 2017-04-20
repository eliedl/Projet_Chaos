__author__ = 'Charles'
__author__ = 'Charles'
__author__ = 'Charles'
__author__ = 'Charles'
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import projet_core


def f(t, y, sigma, rho, beta):
    return [sigma*(y[1] - y[0]), rho*y[0] - y[1] - y[0]*y[2], y[0]*y[1] - beta*y[2]]

def pend(l, t, sigma, rho, beta):
     x, y, z  = l
     dldt = [sigma*(y - x) , rho*x - y - x*z, x*y - beta*z]
     return dldt

if __name__ == "__main__":


    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    #y0 = [10 + 1e-5, 10+ 1e-5, 10+ 1e-5]
    #y1 = [10, 10, 10]

    #y0 = [-5 + 1e-2, 0.5+ 1e-2, 0.5+ 1e-2]
    #y1 = [-5, 0.5, 0.5]

    #y0 = [1+ 1e-5, 1+ 1e-5, 1+ 1e-5]
    #y1 = [1+1e-10,1+1e-10,1+1e-10]

    #sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    #sol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))


def generate_data(y0, t):
    sigma, rho, beta = 10, 28, 8/3
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))

    return sol

def cosine_similarity(mat1, mat2):
    cssim = []
    for i in range(0,mat1.size//3 -1):
        x1 = mat1[i,0]
        y1 = mat1[i,1]
        z1 = mat1[i,2]

        x2 = mat2[i,0]
        y2 = mat2[i,1]
        z2 = mat2[i,2]

        x1p = mat1[i+1,0]
        y1p = mat1[i+1,1]
        z1p = mat1[i+1,2]

        x2p = mat2[i+1,0]
        y2p = mat2[i+1,1]
        z2p = mat2[i+1,2]

        dx1 =  x1p - x1
        dy1 =  y1p - y1
        dz1 =  z1p - z1

        dx2 =  x2p - x2
        dy2 =  y2p - y2
        dz2 =  z2p - z2

        coeff = (dx1*dx2 + dy1*dy2 + dz1*dz2)/ (np.sqrt((dx1**2 + dy1**2 +dz1**2 )*(dx2**2 + dy2**2 +dz2**2 )))
        cssim.append(coeff)
    return cssim

def calculate_threshhold(mat, temps):

    count = 0
    for i in mat:
        if i <= 0.5:
            return temps[count]
        count += 1
    return temps[count]

def run_one():
    t = np.linspace(1, 100, 10001)
    y0 = [-5 + 1e-2, 0.5+ 1e-2, 0.5+ 1e-2]
    y1 = [-5, 0.5, 0.5]
    sol, sol_1 = generate_data(y0,t)
    sim = np.degrees(np.arccos(cosine_similarity(sol,sol_1)))
    plt.plot(t, sim)
    #plt.title(y1)

    print("Temps de corrélation : ", calculate_threshhold(sim,t)," s")
    plt.show()
    print("done")


if __name__ == "__main__":

    t = np.linspace(1, 1000, 10001)
    espacement = 1e-2
    x=1
    y0 = [1+ espacement, 1+ espacement, 1+ espacement]
    y1 = [1,1,1]
    core_1 = projet_core.Core()
    core_1.coordinates = np.array([y1, [1+ espacement, 1+ espacement, 1+ espacement]])
    core_1.attractor = 'Rössler'
    core_1.params = np.array([[0.2, 0.2, 14]])
    core_1.t = t
    core_1.solve_edo()

    initial_values =  core_1.time_series[:,:3]
    thresh_matrix = np.zeros((1, 4))

    resolution = 5000
    espacement = 1e-7
    for i in range(0,resolution,1):

        core_1 = projet_core.Core()

        mat1 = initial_values[i:]

        y1 = [initial_values[i,0] +espacement, initial_values[i,1], initial_values[i,2]]
        y2 = [initial_values[i,0] , initial_values[i,1], initial_values[i,2]+espacement]
        y3 = [initial_values[i,0] , initial_values[i,1]+espacement, initial_values[i,2]]
        y4 = [initial_values[i,0] -espacement, initial_values[i,1], initial_values[i,2]]
        y5 = [initial_values[i,0] , initial_values[i,1], initial_values[i,2]-espacement]
        y6 = [initial_values[i,0] , initial_values[i,1]-espacement, initial_values[i,2]]

        core_1.coordinates = np.array([y1,y2,y3,y4,y5,y6])
        core_1.attractor = 'Rössler'
        core_1.params = np.array([[0.2, 0.2, 14]])
        core_1.t = t
        core_1.solve_edo()

        ymat1 = core_1.time_series[:,:3]
        ymat2 = core_1.time_series[:,3:6]
        ymat3 = core_1.time_series[:,6:9]
        ymat4 = core_1.time_series[:,9:12]
        ymat5 = core_1.time_series[:,12:15]
        ymat6 = core_1.time_series[:,15:18]

        coeff = cosine_similarity(mat1,ymat1)
        threshhold1 = calculate_threshhold(coeff,t)

        coeff = cosine_similarity(mat1,ymat2)
        threshhold2 = calculate_threshhold(coeff,t)
        coeff = cosine_similarity(mat1,ymat3)
        threshhold3 = calculate_threshhold(coeff,t)
        coeff = cosine_similarity(mat1,ymat4)
        threshhold4 = calculate_threshhold(coeff,t)
        coeff = cosine_similarity(mat1,ymat5)
        threshhold5 = calculate_threshhold(coeff,t)
        coeff = cosine_similarity(mat1,ymat6)
        threshhold6 = calculate_threshhold(coeff,t)

        threshhold = (threshhold1+threshhold2+threshhold3+threshhold4+threshhold5+threshhold6)/6
        local = np.array([y1[0], y1[1], y1[2], threshhold])
        thresh_matrix = np.vstack((thresh_matrix, local))
        print((i * 100)//resolution)


    xs = thresh_matrix[50:,0]
    ys = thresh_matrix[50:,1]
    zs = thresh_matrix[50:,2]
    c = thresh_matrix[50:,3]
    count = 0

    #print(c)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    np.savetxt("-7_cossim_cover.txt", thresh_matrix)

    p =ax.scatter(xs, ys, zs, c=c, cmap='plasma', marker = 'o')
    fig.colorbar(p)
    plt.show()