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


def autocorr( mat, min_step, max_step):
    corr = []
    count = 0
    for step in range(min_step,max_step):
        somme = 0
        for i in range(0,mat.size - step):
                somme +=mat[i]*mat[i+step]
        corr.append(somme/(100))
        count += 1
    return corr

def fit(mat,length):

    from scipy.optimize import curve_fit
    corr = autocorr(mat,10,length)
    y = np.linspace(10,length,length-10)

    popt, pcov = curve_fit(func, y, corr)
    return y, corr, popt, pcov

def func(x,a,b,c):
        return a*np.exp(-b*x)


if __name__ == "__main__":
    sigma, rho, beta = 10, 28, 8/3
    t = np.linspace(1, 100, 10001)
    y0 = [1, 1, 1]
    sol = odeint(pend, y0, t, args=(sigma, rho, beta))
    mat = sol

    y1, corr1, popt1, pcov1 = fit(sol[:,0],50)
    y2, corr2, popt2, pcov2 = fit(sol[:,1],50)
    y3, corr3, popt3, pcov3 = fit(sol[:,2],50)

    plt.plot(corr3)

    print(popt1[0],' * ',"e^(-",popt1[1],") + ", popt1[2], " avec ",1/popt1[1])
    print(popt2[0],' * ',"e^(-",popt2[1],") + ", popt2[2], " avec ",1/popt2[1])
    print(popt3[0],' * ',"e^(-",popt3[1],") + ", popt3[2], " avec ",1/popt3[1])

    coeff = (1/popt1[1] + 1/popt2[1] + 1/popt3[1])/3

    print("Le coeff est alors: ",coeff)
    plt.show()
    print("done")