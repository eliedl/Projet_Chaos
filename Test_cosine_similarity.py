__author__ = 'Charles'
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




if __name__ == "__main__":
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    espacement = 1e-1
    x=1
    y0 = [1+ espacement, 1+ espacement, 1+ espacement]
    y1 = [1,1,1]

    sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    sol_1 = odeint(pend, y1, t, args=(sigma, rho, beta))
    sim =np.degrees(np.arccos(cosine_similarity(sol,sol_1)))
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